"""Unit tests for the virtual-number registry (ephemeral inbound extensions)."""
import time

import pytest

from virtual_numbers import (VirtualNumber, VirtualNumberRegistry,
                             VirtualNumberError, extension_from_uri)

pytestmark = pytest.mark.unit


def make_registry(config_factory, tmp_path, **overrides):
    cfg = config_factory(
        VIRTUAL_NUMBERS_ENABLED="true",
        data_dir=str(tmp_path),
        **overrides,
    )
    return VirtualNumberRegistry(cfg)


# --- dialed-URI extraction ---------------------------------------------------

@pytest.mark.parametrize("uri,expected", [
    ("sip:7301@pbx.local", "7301"),
    ("<sip:7301@pbx.local>", "7301"),
    # '*'/'#' extensions are valid virtual numbers and must survive
    # extraction (caller_id_from_uri would reject them).
    ("sip:*77@pbx.local", "*77"),
    ("sip:12#4@pbx.local;transport=udp", "12#4"),
    ("sips:assistant@pbx.local", "assistant"),
    ("", None),
    ("sip:@pbx.local", None),
])
def test_extension_from_uri(uri, expected):
    assert extension_from_uri(uri) == expected


def test_star_number_claimable_via_extraction(config_factory, tmp_path):
    """End-to-end of the call-path matching: create '*77', extract the dialed
    user part the way main.py does, and claim it."""
    reg = make_registry(config_factory, tmp_path)
    entry = reg.create(number="*77", purpose="star code")
    dialed = extension_from_uri("sip:*77@pbx.local")
    assert reg.claim(dialed) is entry


# --- allocation --------------------------------------------------------------

def test_explicit_number(config_factory, tmp_path):
    reg = make_registry(config_factory, tmp_path)
    entry = reg.create(number="7301", purpose="pizza order")
    assert entry.number == "7301"
    assert reg.get(entry.id) is entry
    assert entry.expires_at > time.time()


def test_auto_allocation_lowest_free(config_factory, tmp_path):
    reg = make_registry(config_factory, tmp_path, VIRTUAL_NUMBER_RANGE="7300-7302")
    a = reg.create(purpose="a")
    b = reg.create(purpose="b")
    assert (a.number, b.number) == ("7300", "7301")
    # Explicitly taking the last one exhausts the range.
    reg.create(number="7302", purpose="c")
    with pytest.raises(VirtualNumberError) as exc:
        reg.create(purpose="d")
    assert exc.value.status_code == 503


def test_collision_rejected(config_factory, tmp_path):
    reg = make_registry(config_factory, tmp_path)
    reg.create(number="7301", purpose="x")
    with pytest.raises(VirtualNumberError) as exc:
        reg.create(number="7301", purpose="y")
    assert exc.value.status_code == 409


def test_sip_user_collision_rejected(config_factory, tmp_path):
    reg = make_registry(config_factory, tmp_path, SIP_USER="7350")
    with pytest.raises(VirtualNumberError) as exc:
        reg.create(number="7350", purpose="x")
    assert exc.value.status_code == 409
    # Auto-allocation skips the agent's own identity too.
    reg2 = make_registry(config_factory, tmp_path / "b", SIP_USER="7300",
                         VIRTUAL_NUMBER_RANGE="7300-7301")
    assert reg2.create(purpose="x").number == "7301"


@pytest.mark.parametrize("bad", ["x", "1", "abc", "73 01", "7" * 33])
def test_bad_number_shapes_rejected(config_factory, tmp_path, bad):
    reg = make_registry(config_factory, tmp_path)
    with pytest.raises(VirtualNumberError) as exc:
        reg.create(number=bad, purpose="x")
    assert exc.value.status_code == 400


def test_max_active_cap(config_factory, tmp_path):
    reg = make_registry(config_factory, tmp_path, VIRTUAL_NUMBER_MAX_ACTIVE="2")
    reg.create(purpose="a")
    reg.create(purpose="b")
    with pytest.raises(VirtualNumberError) as exc:
        reg.create(purpose="c")
    assert exc.value.status_code == 503


def test_ttl_clamped_to_max(config_factory, tmp_path):
    reg = make_registry(config_factory, tmp_path, VIRTUAL_NUMBER_MAX_TTL_S="100")
    entry = reg.create(purpose="x", ttl_s=10_000)
    assert entry.expires_at <= time.time() + 101


# --- claim / consume ---------------------------------------------------------

def test_claim_and_consume(config_factory, tmp_path):
    reg = make_registry(config_factory, tmp_path)
    entry = reg.create(number="7301", purpose="x")

    assert reg.claim("9999") is None          # unknown number
    claimed = reg.claim("7301")
    assert claimed is entry and claimed.claimed
    assert reg.claim("7301") is None          # already claimed

    consumed = reg.consume(entry.id)
    assert consumed is entry
    assert reg.consume(entry.id) is None      # idempotent
    assert reg.get(entry.id) is None


def test_claim_disabled_config(config_factory, tmp_path):
    cfg = config_factory(VIRTUAL_NUMBERS_ENABLED="false", data_dir=str(tmp_path))
    reg = VirtualNumberRegistry(cfg)
    assert reg.claim("7301") is None


def test_release_unclaims(config_factory, tmp_path):
    reg = make_registry(config_factory, tmp_path)
    entry = reg.create(number="7301", purpose="x")
    reg.claim("7301")
    reg.release(entry.id)
    assert reg.claim("7301") is entry


# --- sweep -------------------------------------------------------------------

def test_sweep_removes_only_unclaimed_expired(config_factory, tmp_path):
    reg = make_registry(config_factory, tmp_path)
    expired = reg.create(number="7301", purpose="a", ttl_s=1)
    live = reg.create(number="7302", purpose="b", ttl_s=3600)
    claimed = reg.create(number="7303", purpose="c", ttl_s=1)
    reg.claim("7303")

    expired.expires_at = time.time() - 1
    claimed.expires_at = time.time() - 1
    reg._sweep()

    assert reg.get(expired.id) is None        # expired + unclaimed -> gone
    assert reg.get(live.id) is live           # not expired -> stays
    assert reg.get(claimed.id) is claimed     # claimed (call live) -> exempt


async def test_expiry_webhook_payload(config_factory, tmp_path, monkeypatch):
    import api as api_module
    sent = []

    async def fake_deliver(url, payload, config, api_name="webhook"):
        sent.append((url, payload))
        return True

    monkeypatch.setattr(api_module, "deliver_webhook", fake_deliver)
    reg = make_registry(config_factory, tmp_path)
    entry = reg.create(number="7301", purpose="pizza",
                       callback_url="https://example.com/hook", ttl_s=1)
    entry.expires_at = time.time() - 1
    reg._sweep()
    # Let the fire-and-forget task run.
    import asyncio
    await asyncio.gather(*reg._webhook_tasks)

    assert len(sent) == 1
    url, payload = sent[0]
    assert url == "https://example.com/hook"
    assert payload["event"] == "virtual_number.expired"
    assert payload["status"] == "expired"
    assert payload["number"] == "7301"
    assert payload["purpose"] == "pizza"


# --- persistence -------------------------------------------------------------

def test_persistence_round_trip(config_factory, tmp_path):
    cfg = config_factory(VIRTUAL_NUMBERS_ENABLED="true", data_dir=str(tmp_path))
    reg = VirtualNumberRegistry(cfg)
    entry = reg.create(number="7301", purpose="pizza", greeting="hi",
                       callback_url="https://example.com/hook")
    reg.claim("7301")

    reloaded = VirtualNumberRegistry(cfg)
    loaded = reloaded.get(entry.id)
    assert loaded is not None
    assert loaded.number == "7301"
    assert loaded.purpose == "pizza"
    assert loaded.greeting == "hi"
    # A claimed entry from before a crash reloads as active (unclaimed).
    assert loaded.claimed is False


def test_expired_entries_dropped_on_load(config_factory, tmp_path):
    cfg = config_factory(VIRTUAL_NUMBERS_ENABLED="true", data_dir=str(tmp_path))
    reg = VirtualNumberRegistry(cfg)
    entry = reg.create(number="7301", purpose="x", ttl_s=1)
    # Rewrite the store with an already-expired timestamp.
    entry.expires_at = time.time() - 10
    reg._persist()

    reloaded = VirtualNumberRegistry(cfg)
    assert reloaded.get(entry.id) is None
    assert [e.id for e in reloaded._expired_on_load] == [entry.id]
