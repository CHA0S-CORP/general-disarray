"""Regression tests for the API / queue / verification hardening pass.

Pure-logic tier: Host allowlist, CSRF helper, rate-limiter bounds, SSRF IP
policy, retry attempt handling, verification lockout + TOTP replay, and the
virtual-number store's corrupt-file handling.
"""
import ipaddress
import json
import time

import pyotp
import pytest

import api
from api import RateLimiter, host_header_allowed, _resolve_allowed_ips
from identity_verification import IdentityVerifier, VerificationStore

pytestmark = pytest.mark.unit


# --- Host allowlist (DNS-rebinding defense, tokenless mode) ---------------------

@pytest.mark.parametrize("host", [
    None, "localhost", "localhost:8080", "127.0.0.1:8180", "[::1]:8080", "::1",
    "sip-agent:8080", "host.docker.internal:8080", "dgx-spark.local",
    "app.localhost", "box.home.arpa", "svc.internal", "10.0.0.5", "testserver",
])
def test_host_allowed_defaults(host):
    assert host_header_allowed(host) is True


@pytest.mark.parametrize("host", [
    "evil.example.com", "attacker.com:8080", "localhost.attacker.com",
    "127.0.0.1.nip.io",
])
def test_host_rejected_defaults(host):
    assert host_header_allowed(host) is False


def test_host_extra_allowlist():
    assert host_header_allowed("agent.example.net", ["agent.example.net"]) is True
    assert host_header_allowed("a.corp.example", ["*.corp.example"]) is True
    assert host_header_allowed("corp.example.evil", ["*.corp.example"]) is False
    assert host_header_allowed("anything.example.com", ["*"]) is True


# --- rate limiter bounds ----------------------------------------------------------

def test_rate_limiter_bucket_map_is_bounded(monkeypatch):
    monkeypatch.setattr(RateLimiter, "MAX_BUCKETS", 50)
    limiter = RateLimiter(rpm=1, burst=5)
    # Every new key consumes a token, so none are "fully refilled" — the old
    # prune kept them all. The LRU eviction must still bound the map.
    for i in range(1000):
        limiter.allow(f"k{i}")
    assert len(limiter._buckets) <= 50


def test_rate_limiter_still_limits_hot_key(monkeypatch):
    monkeypatch.setattr(RateLimiter, "MAX_BUCKETS", 50)
    limiter = RateLimiter(rpm=1, burst=1)
    assert limiter.allow("hot") is True
    for i in range(10):
        limiter.allow(f"other{i}")
    assert limiter.allow("hot") is False


# --- SSRF IP policy ------------------------------------------------------------------

@pytest.mark.parametrize("addr", [
    "100.64.0.1",          # CGNAT / Tailscale
    "100.100.100.100",     # Tailscale MagicDNS
    "64:ff9b::a00:1",      # NAT64 -> 10.0.0.1
    "64:ff9b::7f00:1",     # NAT64 -> 127.0.0.1
    "::ffff:10.0.0.1",     # IPv4-mapped private
    "198.18.0.1",          # benchmarking range
    "127.0.0.1", "10.1.2.3", "169.254.169.254", "::1", "fe80::1",
])
def test_ssrf_rejects_non_global(addr):
    with pytest.raises(ValueError):
        _resolve_allowed_ips(addr)


@pytest.mark.parametrize("addr", ["8.8.8.8", "64:ff9b::808:808", "2606:4700:4700::1111"])
def test_ssrf_allows_global(addr):
    assert _resolve_allowed_ips(addr) == [str(ipaddress.ip_address(addr))]


# --- retry attempts --------------------------------------------------------------------

async def test_retry_zero_attempts_still_calls_once(config_factory, monkeypatch):
    from retry_utils import retry_async, RetryError
    cfg = config_factory(api_retry_attempts=0)
    calls = []

    async def ok():
        calls.append(1)
        return "done"

    assert await retry_async(ok, config=cfg) == "done"
    assert calls == [1]

    async def boom():
        calls.append(2)
        raise RuntimeError("x")

    with pytest.raises(RetryError):
        await retry_async(boom, config=cfg, max_attempts=0)
    assert calls == [1, 2]


async def test_retry_explicit_zero_delay_is_honoured(config_factory, monkeypatch):
    from retry_utils import retry_async
    cfg = config_factory(api_retry_base_delay_s=5)
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("retry_utils.asyncio.sleep", fake_sleep)
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("transient")
        return "ok"

    assert await retry_async(flaky, config=cfg, max_attempts=3, base_delay=0) == "ok"
    assert sleeps == [0, 0]


# --- verification lockout + TOTP replay ------------------------------------------------

def _verifier(config_factory, tmp_path, **overrides):
    cfg = config_factory(data_dir=str(tmp_path), **overrides)
    return IdentityVerifier(cfg, VerificationStore(cfg))


def test_lockout_after_failures_blocks_even_correct_pin(config_factory, tmp_path):
    ver = _verifier(config_factory, tmp_path, verify_pin="4321",
                    verify_lockout_failures=3, verify_lockout_s=900)
    for _ in range(3):
        assert ver.verify_pin("1001", "0000") is False
    assert ver.is_locked_out("1001")
    assert ver.lockout_remaining_s("1001") > 0
    # Correct PIN refused while locked (every path shares the guard).
    assert ver.verify_pin("1001", "4321") is False
    assert ver.verify("1001", "4321") == (False, None)
    assert ver.verify_factors("1001", pin="4321") == (False, None)
    # Other callers are unaffected.
    assert ver.verify_pin("2002", "4321") is True


def test_lockout_expires(config_factory, tmp_path, monkeypatch):
    ver = _verifier(config_factory, tmp_path, verify_pin="4321",
                    verify_lockout_failures=2, verify_lockout_s=60)
    clock = [1000.0]
    monkeypatch.setattr("identity_verification.time.monotonic", lambda: clock[0])
    ver.verify_pin("1001", "0")
    ver.verify_pin("1001", "1")
    assert ver.is_locked_out("1001")
    clock[0] += 61
    assert not ver.is_locked_out("1001")
    assert ver.verify_pin("1001", "4321") is True


def test_success_resets_failure_count(config_factory, tmp_path):
    ver = _verifier(config_factory, tmp_path, verify_pin="4321",
                    verify_lockout_failures=3)
    ver.verify_pin("1001", "0")
    ver.verify_pin("1001", "1")
    assert ver.verify_pin("1001", "4321") is True
    ver.verify_pin("1001", "2")
    ver.verify_pin("1001", "3")
    assert not ver.is_locked_out("1001")


def test_blank_candidate_is_not_a_strike(config_factory, tmp_path):
    ver = _verifier(config_factory, tmp_path, verify_pin="4321",
                    verify_lockout_failures=1)
    assert ver.verify_pin("1001", "") is False
    assert not ver.is_locked_out("1001")


def test_verify_factors_counts_one_strike(config_factory, tmp_path):
    secret = pyotp.random_base32()
    ver = _verifier(config_factory, tmp_path, verify_pin="4321",
                    verify_totp_secret=secret, verify_lockout_failures=2)
    assert ver.verify_factors("1001", pin="0000", otp="000000") == (False, None)
    assert not ver.is_locked_out("1001")


def test_lockout_disabled_with_zero(config_factory, tmp_path):
    ver = _verifier(config_factory, tmp_path, verify_pin="4321",
                    verify_lockout_failures=0)
    for _ in range(20):
        ver.verify_pin("1001", "0")
    assert ver.verify_pin("1001", "4321") is True


def test_totp_code_cannot_be_replayed(config_factory, tmp_path):
    secret = pyotp.random_base32()
    ver = _verifier(config_factory, tmp_path, verify_totp_secret=secret)
    code = pyotp.TOTP(secret).now()
    assert ver.verify_totp("1001", code) is True
    assert ver.verify_totp("1001", code) is False
    # Not even under another caller id sharing the global secret.
    assert ver.verify_totp("2002", code) is False


def test_totp_older_step_rejected_after_newer_accepted(config_factory, tmp_path):
    secret = pyotp.random_base32()
    ver = _verifier(config_factory, tmp_path, verify_totp_secret=secret,
                    verify_totp_window=1)
    totp = pyotp.TOTP(secret)
    now = time.time()
    assert ver.verify_totp("1001", totp.at(now)) is True
    # The previous step's code is inside the skew window but older than the
    # step just used — accepting it would allow a replay of an observed code.
    assert ver.verify_totp("1001", totp.at(now - 30)) is False


def test_explicit_totp_replay_rejected(config_factory, tmp_path):
    ver = _verifier(config_factory, tmp_path)
    secret = pyotp.random_base32()
    code = pyotp.TOTP(secret).now()
    assert ver.verify_explicit(code, totp_secret=secret) == (True, "otp")
    assert ver.verify_explicit(code, totp_secret=secret) == (False, None)


# --- virtual numbers store durability -----------------------------------------------------

def test_corrupt_virtual_number_store_is_backed_up(config_factory, tmp_path):
    from virtual_numbers import VirtualNumberRegistry
    cfg = config_factory(VIRTUAL_NUMBERS_ENABLED="true", data_dir=str(tmp_path))
    store = tmp_path / "virtual_numbers.json"
    store.write_text("{not json")

    reg = VirtualNumberRegistry(cfg)
    backups = list(tmp_path.glob("virtual_numbers.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == "{not json"

    # A later persist writes a fresh store but leaves the backup intact.
    reg.create(number="7301", purpose="pizza")
    assert json.loads(store.read_text())[0]["number"] == "7301"
    assert backups[0].read_text() == "{not json"


# --- webhook delivery id ----------------------------------------------------------------------

async def test_webhook_id_stable_across_retries(config_factory, monkeypatch):
    import httpx
    cfg = config_factory(webhook_allow_private="true", api_retry_attempts=3,
                         api_retry_base_delay_s=0)
    seen = []

    def handler(request):
        seen.append(request.headers.get("X-Webhook-Id"))
        return httpx.Response(503 if len(seen) < 3 else 200)

    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(api.httpx, "AsyncClient", fake_client)
    assert await api.deliver_webhook("http://127.0.0.1:9/hook", {"a": 1}, cfg) is True
    assert len(seen) == 3
    assert seen[0] and len(set(seen)) == 1
    first_id = seen[0]

    # A new delivery gets a new id.
    seen.clear()
    await api.deliver_webhook("http://127.0.0.1:9/hook", {"a": 1}, cfg)
    assert seen[-1] and seen[-1] != first_id
