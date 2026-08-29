"""Unit tests for optional caller identity verification (PIN + TOTP)."""
import pyotp
import pytest

from identity_verification import (
    IdentityVerifier,
    VerificationStore,
    is_safe_caller_id,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def store(config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path))
    return VerificationStore(cfg), cfg


# --- caller id validation ------------------------------------------------------

def test_is_safe_caller_id():
    assert is_safe_caller_id("1001")
    assert is_safe_caller_id("alice.smith+1")
    assert not is_safe_caller_id("")
    assert not is_safe_caller_id("../etc/passwd")
    assert not is_safe_caller_id("a" * 65)


# --- store round-trip ----------------------------------------------------------

def test_set_get_delete_round_trip(store):
    st, _ = store
    view = st.set_credentials("1001", pin="1234", generate_totp=True)
    assert view["has_pin"] and view["has_totp"]

    record = st.get("1001")
    assert record["pin_hash"] and record["pin_salt"] and record["totp_secret"]
    # The raw PIN is never persisted.
    assert "1234" not in str(record)

    assert st.delete("1001") is True
    assert st.get("1001") is None
    assert st.delete("1001") is False  # already gone


def test_set_credentials_rejects_empty_enrollment(store):
    st, _ = store
    assert st.set_credentials("1001") is None
    assert st.get("1001") is None


def test_set_credentials_rejects_bad_base32(store):
    st, _ = store
    assert st.set_credentials("1001", totp_secret="not base32!") is None


def test_set_credentials_preserves_other_factor(store):
    st, _ = store
    st.set_credentials("1001", pin="1234")
    st.set_credentials("1001", generate_totp=True)  # add TOTP, keep PIN
    record = st.get("1001")
    assert record["pin_hash"] and record["totp_secret"]


def test_public_view_hides_secrets(store):
    st, _ = store
    st.set_credentials("1001", pin="1234", generate_totp=True)
    view = st.public_view("1001")
    assert set(view) == {"caller_id", "has_pin", "has_totp", "updated_at"}


def test_provisioning_uri(store):
    st, cfg = store
    assert st.provisioning_uri("1001", cfg.verify_issuer) is None  # no secret yet
    st.set_credentials("1001", generate_totp=True)
    uri = st.provisioning_uri("1001", cfg.verify_issuer)
    assert uri.startswith("otpauth://totp/")
    assert "1001" in uri


# --- PIN verification ----------------------------------------------------------

def test_global_pin(config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path), verify_pin="4321")
    ver = IdentityVerifier(cfg, VerificationStore(cfg))
    assert ver.verify_pin("1001", "4321") is True
    assert ver.verify_pin("1001", "0000") is False
    assert ver.verify_pin("1001", "") is False


def test_per_caller_pin_overrides_global(config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path), verify_pin="4321")
    store = VerificationStore(cfg)
    store.set_credentials("1001", pin="1234")
    ver = IdentityVerifier(cfg, store)
    # Enrolled caller uses their own PIN; the global PIN no longer works for them.
    assert ver.verify_pin("1001", "1234") is True
    assert ver.verify_pin("1001", "4321") is False
    # A different, un-enrolled caller still falls back to the global PIN.
    assert ver.verify_pin("2002", "4321") is True


def test_no_pin_configured_rejects(config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path))
    ver = IdentityVerifier(cfg, VerificationStore(cfg))
    assert ver.verify_pin("1001", "1234") is False


# --- TOTP verification ---------------------------------------------------------

def test_global_totp(config_factory, tmp_path):
    secret = pyotp.random_base32()
    cfg = config_factory(data_dir=str(tmp_path), verify_totp_secret=secret)
    ver = IdentityVerifier(cfg, VerificationStore(cfg))
    assert ver.verify_totp("1001", pyotp.TOTP(secret).now()) is True
    assert ver.verify_totp("1001", "000000") is False


def test_per_caller_totp(config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path))
    store = VerificationStore(cfg)
    store.set_credentials("1001", generate_totp=True)
    ver = IdentityVerifier(cfg, store)
    secret = store.get("1001")["totp_secret"]
    assert ver.verify_totp("1001", pyotp.TOTP(secret).now()) is True


def test_totp_window_rejects_far_code(config_factory, tmp_path):
    secret = pyotp.random_base32()
    cfg = config_factory(data_dir=str(tmp_path), verify_totp_secret=secret,
                         verify_totp_window=1)
    ver = IdentityVerifier(cfg, VerificationStore(cfg))
    # A code generated for a timestamp 5 minutes ago is far outside +/-1 step.
    import time
    stale = pyotp.TOTP(secret).at(int(time.time()) - 300)
    assert ver.verify_totp("1001", stale) is False


def test_current_otp(config_factory, tmp_path):
    secret = pyotp.random_base32()
    cfg = config_factory(data_dir=str(tmp_path), verify_totp_secret=secret)
    ver = IdentityVerifier(cfg, VerificationStore(cfg))
    result = ver.current_otp("1001")
    assert result is not None
    code, remaining = result
    assert ver.verify_totp("1001", code) is True
    assert 0 < remaining <= 30


# --- combined / configured -----------------------------------------------------

def test_verify_auto_accepts_either_factor(config_factory, tmp_path):
    secret = pyotp.random_base32()
    cfg = config_factory(data_dir=str(tmp_path))
    store = VerificationStore(cfg)
    store.set_credentials("1001", pin="1234", totp_secret=secret)
    ver = IdentityVerifier(cfg, store)

    ok, method = ver.verify("1001", "1234", method="auto")
    assert ok and method == "pin"
    ok, method = ver.verify("1001", pyotp.TOTP(secret).now(), method="auto")
    assert ok and method == "otp"
    ok, method = ver.verify("1001", "9999", method="auto")
    assert not ok and method is None


def test_can_verify_and_is_configured(config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path))
    store = VerificationStore(cfg)
    ver = IdentityVerifier(cfg, store)
    assert ver.is_configured() is False
    assert ver.can_verify("1001") is False
    store.set_credentials("1001", pin="1234")
    assert ver.has_any_credentials("1001") is True
    assert ver.can_verify("1001") is True


# --- explicit (per-request) credentials ----------------------------------------

def test_verify_explicit_pin_and_otp(config_factory, tmp_path):
    """Ad-hoc factors are checked directly, with no store/global lookup."""
    secret = pyotp.random_base32()
    cfg = config_factory(data_dir=str(tmp_path))
    ver = IdentityVerifier(cfg, VerificationStore(cfg))  # empty store

    ok, method = ver.verify_explicit("1234", pin="1234")
    assert ok and method == "pin"

    ok, method = ver.verify_explicit(pyotp.TOTP(secret).now(), totp_secret=secret)
    assert ok and method == "otp"

    ok, method = ver.verify_explicit("0000", pin="1234", totp_secret=secret)
    assert not ok and method is None


def test_verify_explicit_respects_forced_method(config_factory, tmp_path):
    secret = pyotp.random_base32()
    cfg = config_factory(data_dir=str(tmp_path))
    ver = IdentityVerifier(cfg, VerificationStore(cfg))

    # method='pin' ignores the (correct) OTP; method='otp' ignores the PIN.
    ok, _ = ver.verify_explicit(pyotp.TOTP(secret).now(), pin="1234",
                                totp_secret=secret, method="pin")
    assert not ok
    ok, _ = ver.verify_explicit("1234", pin="1234", totp_secret=secret, method="otp")
    assert not ok


def test_verify_explicit_empty_candidate(config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path))
    ver = IdentityVerifier(cfg, VerificationStore(cfg))
    assert ver.verify_explicit("", pin="1234") == (False, None)


# --- configurable TOTP parameters ----------------------------------------------

def test_configurable_totp_digits_period_algorithm(config_factory, tmp_path):
    """Stored-secret verification honours the configured digits/period/algorithm."""
    import hashlib
    secret = pyotp.random_base32()
    cfg = config_factory(data_dir=str(tmp_path), verify_totp_digits=8,
                         verify_totp_period=60, verify_totp_algorithm="SHA256")
    store = VerificationStore(cfg)
    store.set_credentials("1001", totp_secret=secret)
    ver = IdentityVerifier(cfg, store)

    matching = pyotp.TOTP(secret, digits=8, digest=hashlib.sha256, interval=60).now()
    assert ver.verify_totp("1001", matching) is True
    # A default 6-digit/SHA1/30s code must NOT verify under the custom config.
    assert ver.verify_totp("1001", pyotp.TOTP(secret).now()) is False


def test_current_otp_uses_configured_params(config_factory, tmp_path):
    import hashlib
    secret = pyotp.random_base32()
    cfg = config_factory(data_dir=str(tmp_path), verify_totp_digits=8,
                         verify_totp_period=60, verify_totp_algorithm="SHA512")
    store = VerificationStore(cfg)
    store.set_credentials("1001", totp_secret=secret)
    ver = IdentityVerifier(cfg, store)

    code, remaining = ver.current_otp("1001")
    assert code == pyotp.TOTP(secret, digits=8, digest=hashlib.sha512, interval=60).now()
    assert len(code) == 8
    assert 0 < remaining <= 60


def test_provisioning_uri_embeds_custom_params(config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path), verify_totp_digits=8,
                         verify_totp_period=60, verify_totp_algorithm="SHA256")
    store = VerificationStore(cfg)
    store.set_credentials("1001", generate_totp=True)
    uri = store.provisioning_uri("1001", cfg.verify_issuer)
    assert "digits=8" in uri and "period=60" in uri and "algorithm=SHA256" in uri


def test_verify_explicit_totp_param_overrides(config_factory, tmp_path):
    import hashlib
    secret = pyotp.random_base32()
    cfg = config_factory(data_dir=str(tmp_path))  # defaults 6/30/SHA1
    ver = IdentityVerifier(cfg, VerificationStore(cfg))

    code = pyotp.TOTP(secret, digits=8, digest=hashlib.sha256, interval=60).now()
    ok, method = ver.verify_explicit(code, totp_secret=secret, totp_digits=8,
                                     totp_period=60, totp_algorithm="SHA256")
    assert ok and method == "otp"
    # Without the overrides, the same code fails under the SHA1/6/30 defaults.
    ok, _ = ver.verify_explicit(code, totp_secret=secret)
    assert not ok


# --- review regressions --------------------------------------------------------

def test_non_ascii_pin_is_a_mismatch_not_an_error(config_factory, tmp_path):
    """hmac.compare_digest(str, str) raises on non-ASCII; must be a plain False."""
    cfg = config_factory(data_dir=str(tmp_path), verify_pin="1234")
    v = IdentityVerifier(cfg, VerificationStore(cfg))
    assert v.verify_pin("9999", "１２３４") is False
    assert v.verify_explicit("１２", pin="12") == (False, None)
    # per-caller hash path too
    v.store.set_credentials("1001", pin="2468")
    assert v.verify_pin("1001", "２４") is False


def test_store_file_is_owner_only(store):
    import os
    import stat
    s, _ = store
    s.set_credentials("1001", pin="1234")
    mode = stat.S_IMODE(os.stat(s.path).st_mode)
    assert mode == 0o600


def test_current_otp_code_and_expiry_agree(config_factory, tmp_path):
    import time
    from identity_verification import build_totp, totp_params
    secret = pyotp.random_base32()
    cfg = config_factory(data_dir=str(tmp_path), verify_totp_secret=secret)
    v = IdentityVerifier(cfg, VerificationStore(cfg))
    code, remaining = v.current_otp("anyone")
    now = int(time.time())
    assert code == build_totp(secret, totp_params(cfg)).at(now)
    assert remaining == 30 - now % 30


def test_has_own_totp_secret_ignores_global(config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path), verify_totp_secret=pyotp.random_base32())
    v = IdentityVerifier(cfg, VerificationStore(cfg))
    assert v.has_own_totp_secret("1001") is False
    v.store.set_credentials("1001", generate_totp=True)
    assert v.has_own_totp_secret("1001") is True


def test_async_facades(config_factory, tmp_path):
    import asyncio
    cfg = config_factory(data_dir=str(tmp_path), verify_pin="1234")
    v = IdentityVerifier(cfg, VerificationStore(cfg))
    assert asyncio.run(v.averify("x", "1234")) == (True, "pin")
    assert asyncio.run(v.averify_pin("x", "0000")) is False
    assert asyncio.run(v.averify_explicit("77", pin="77")) == (True, "pin")
