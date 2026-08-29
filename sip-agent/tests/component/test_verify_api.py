"""Component tests for the identity-verification REST endpoints."""
import asyncio

import pyotp
import pytest

pytestmark = pytest.mark.component


# --- enrollment CRUD -----------------------------------------------------------

def test_enroll_get_delete_round_trip(client):
    r = client.post("/verify/credentials",
                    json={"caller_id": "1001", "pin": "1234", "generate_totp": True})
    assert r.status_code == 200
    body = r.json()
    assert body["caller_id"] == "1001"
    assert body["has_pin"] and body["has_totp"]
    assert body["provisioning_uri"].startswith("otpauth://totp/")

    g = client.get("/verify/credentials/1001")
    assert g.status_code == 200
    assert g.json()["has_totp"] is True

    d = client.delete("/verify/credentials/1001")
    assert d.status_code == 200 and d.json()["success"] is True
    assert client.get("/verify/credentials/1001").status_code == 404
    assert client.delete("/verify/credentials/1001").status_code == 404


def test_enroll_requires_a_factor(client):
    r = client.post("/verify/credentials", json={"caller_id": "1001"})
    assert r.status_code == 400


def test_enroll_rejects_bad_caller_id(client):
    r = client.post("/verify/credentials",
                    json={"caller_id": "../etc/passwd", "pin": "1234"})
    assert r.status_code == 400


# --- verification --------------------------------------------------------------

def test_verify_pin_and_otp(client, assistant):
    secret = pyotp.random_base32()
    client.post("/verify/credentials",
                json={"caller_id": "1001", "pin": "1234", "totp_secret": secret})

    ok = client.post("/verify", json={"caller_id": "1001", "pin": "1234"})
    assert ok.json() == {"caller_id": "1001", "verified": True, "method": "pin"}

    code = pyotp.TOTP(secret).now()
    ok = client.post("/verify", json={"caller_id": "1001", "otp": code})
    assert ok.json()["verified"] is True and ok.json()["method"] == "otp"

    bad = client.post("/verify", json={"caller_id": "1001", "pin": "0000"})
    assert bad.json()["verified"] is False and bad.json()["method"] is None


def test_verify_requires_a_factor(client):
    r = client.post("/verify", json={"caller_id": "1001"})
    assert r.status_code == 422  # pydantic model validator


def test_verify_global_pin_fallback(make_client, config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path), verify_pin="4321")
    c, _ = make_client(cfg)
    r = c.post("/verify", json={"caller_id": "9999", "pin": "4321"})
    assert r.json()["verified"] is True


# --- current OTP ---------------------------------------------------------------

def test_get_current_otp(client):
    secret = pyotp.random_base32()
    client.post("/verify/credentials",
                json={"caller_id": "1001", "totp_secret": secret})
    r = client.get("/verify/otp/1001")
    assert r.status_code == 200
    body = r.json()
    assert body["otp"] == pyotp.TOTP(secret).now()
    assert 0 < body["expires_in_s"] <= 30


def test_get_current_otp_missing(client):
    assert client.get("/verify/otp/nobody").status_code == 404


# --- call and verify (outbound) ------------------------------------------------

def test_call_verify_rejects_uncredentialed_caller(make_client, config_factory, tmp_path):
    """No PIN/TOTP for the caller and no global factor -> nothing to check.

    Isolated data dir so no other test's enrollment leaks in.
    """
    c, _ = make_client(config_factory(data_dir=str(tmp_path)))
    r = c.post("/verify/call", json={"caller_id": "1001"})
    assert r.status_code == 400


def test_call_verify_rejects_bad_caller_id(client):
    r = client.post("/verify/call", json={"caller_id": "../etc/passwd"})
    assert r.status_code == 400


class _FakeCall:
    def __init__(self):
        self.is_active = True
        self.media_ready = True


class _FakeSip:
    """Minimal SIP double that answers immediately and replays queued DTMF."""

    def __init__(self, digits):
        self._digits = list(digits)
        self.call = _FakeCall()

    async def make_call(self, uri, caller_name=None):
        return self.call

    async def hangup_call(self, call_info):
        call_info.is_active = False

    async def send_audio(self, call_info, audio, tag=None):
        pass

    def clear_dtmf(self, call_info):
        pass

    def get_dtmf_digit(self, call_info):
        return self._digits.pop(0) if self._digits else None


class _FakeAudio:
    def __init__(self):
        self.said = []

    async def synthesize(self, text):
        self.said.append(text)
        return b"\x00\x00"

    def new_session_state(self):
        return {}


def _verify_handler(assistant, digits):
    from api import OutboundCallHandler
    assistant.sip_handler = _FakeSip(digits)
    assistant.audio_pipeline = _FakeAudio()
    return OutboundCallHandler(assistant, call_queue=None)


def test_call_verify_success_with_correct_pin(client, assistant):
    from api import VerifyCallRequest
    client.post("/verify/credentials", json={"caller_id": "1001", "pin": "2468"})
    handler = _verify_handler(assistant, ["2", "4", "6", "8", "#"])

    resp = asyncio.run(handler.run_verify_call(
        VerifyCallRequest(caller_id="1001", extension="1001")))

    assert resp.verified is True
    assert resp.method == "pin"
    assert resp.status.value == "completed"
    # The call was torn down.
    assert handler.assistant.sip_handler.call.is_active is False


def test_call_verify_fails_with_wrong_code(client, assistant, comp_config):
    from api import VerifyCallRequest
    client.post("/verify/credentials", json={"caller_id": "1001", "pin": "2468"})
    # One wrong entry per allowed attempt, each terminated by '#'.
    digits = ["9"] * comp_config.verify_max_attempts
    queued = []
    for d in digits:
        queued += [d, "#"]
    handler = _verify_handler(assistant, queued)

    resp = asyncio.run(handler.run_verify_call(
        VerifyCallRequest(caller_id="1001", extension="1001")))

    assert resp.verified is False
    assert resp.method is None
    assert resp.attempts == comp_config.verify_max_attempts


def test_call_verify_uses_spoken_message_overrides(client, assistant):
    from api import VerifyCallRequest
    client.post("/verify/credentials", json={"caller_id": "1001", "pin": "2468"})
    handler = _verify_handler(assistant, ["2", "4", "6", "8", "#"])

    resp = asyncio.run(handler.run_verify_call(VerifyCallRequest(
        caller_id="1001", extension="1001",
        prompt="Key in your secret code now.",
        success_phrase="You are in. Bye.")))

    assert resp.verified is True
    said = handler.assistant.audio_pipeline.said
    assert "Key in your secret code now." in said
    assert "You are in. Bye." in said
    # The configured defaults were overridden, not spoken.
    assert assistant.config.verify_call_prompt not in said


def test_call_verify_with_inline_pin_no_enrollment(make_client, config_factory, tmp_path):
    """A per-request PIN verifies a caller with no stored/global credentials."""
    from api import VerifyCallRequest, OutboundCallHandler
    c, a = make_client(config_factory(data_dir=str(tmp_path)))
    a.sip_handler = _FakeSip(["2", "4", "6", "8", "#"])
    a.audio_pipeline = _FakeAudio()
    handler = OutboundCallHandler(a, call_queue=None)

    resp = asyncio.run(handler.run_verify_call(
        VerifyCallRequest(caller_id="1001", extension="1001", pin="2468")))

    assert resp.verified is True
    assert resp.method == "pin"


def test_call_verify_with_inline_totp_secret(make_client, config_factory, tmp_path):
    from api import VerifyCallRequest, OutboundCallHandler
    secret = pyotp.random_base32()
    code = pyotp.TOTP(secret).now()
    c, a = make_client(config_factory(data_dir=str(tmp_path)))
    a.sip_handler = _FakeSip(list(code) + ["#"])
    a.audio_pipeline = _FakeAudio()
    handler = OutboundCallHandler(a, call_queue=None)

    resp = asyncio.run(handler.run_verify_call(
        VerifyCallRequest(caller_id="1001", extension="1001", totp_secret=secret)))

    assert resp.verified is True
    assert resp.method == "otp"


def test_call_verify_endpoint_accepts_inline_pin(make_client, config_factory, tmp_path, monkeypatch):
    """The endpoint takes pin/totp_secret and reaches the call path (no 400)."""
    from api import OutboundCallHandler
    c, a = make_client(config_factory(data_dir=str(tmp_path)))

    captured = {}

    async def fake_run(self, request):
        captured["pin"] = request.pin
        captured["totp_secret"] = request.totp_secret
        from api import VerifyCallResponse, CallStatus
        return VerifyCallResponse(call_id="x", status=CallStatus.COMPLETED,
                                  verified=True, method="pin")

    monkeypatch.setattr(OutboundCallHandler, "run_verify_call", fake_run)
    r = c.post("/verify/call",
               json={"caller_id": "1001", "pin": "2468", "totp_secret": "JBSWY3DPEHPK3PXP"})
    assert r.status_code == 200
    assert r.json()["verified"] is True
    assert captured == {"pin": "2468", "totp_secret": "JBSWY3DPEHPK3PXP"}


def test_call_verify_blank_caller_id_with_inline_pin(make_client, config_factory, tmp_path):
    """No caller_id needed when an inline PIN and an extension are supplied."""
    from api import VerifyCallRequest, OutboundCallHandler
    c, a = make_client(config_factory(data_dir=str(tmp_path)))
    a.sip_handler = _FakeSip(["2", "4", "6", "8", "#"])
    a.audio_pipeline = _FakeAudio()
    handler = OutboundCallHandler(a, call_queue=None)

    resp = asyncio.run(handler.run_verify_call(
        VerifyCallRequest(extension="1001", pin="2468")))

    assert resp.verified is True


def test_call_verify_requires_caller_id_or_extension(client, assistant):
    from api import VerifyCallRequest, RequestRejected, OutboundCallHandler
    handler = OutboundCallHandler(assistant, call_queue=None)
    with pytest.raises(RequestRejected):
        asyncio.run(handler.run_verify_call(VerifyCallRequest(pin="2468")))


def test_call_verify_rejects_bad_inline_totp_secret(client, assistant):
    from api import VerifyCallRequest, RequestRejected, OutboundCallHandler
    handler = OutboundCallHandler(assistant, call_queue=None)
    with pytest.raises(RequestRejected):
        asyncio.run(handler.run_verify_call(
            VerifyCallRequest(caller_id="1001", totp_secret="not base32!")))


def test_call_verify_keypress_mutes_prompt(make_client, config_factory, tmp_path):
    """The caller's first keypress barges in — the still-playing prompt is
    flushed via the playlist player's clear()."""
    from api import VerifyCallRequest, OutboundCallHandler

    class _Player:
        def __init__(self):
            self.cleared = 0

        def clear(self):
            self.cleared += 1

    c, a = make_client(config_factory(data_dir=str(tmp_path)))
    a.sip_handler = _FakeSip(["2", "4", "6", "8", "#"])
    a.audio_pipeline = _FakeAudio()
    player = _Player()
    a.sip_handler.get_playlist_player = lambda call_info: player
    handler = OutboundCallHandler(a, call_queue=None)

    resp = asyncio.run(handler.run_verify_call(
        VerifyCallRequest(caller_id="1001", extension="1001", pin="2468")))

    assert resp.verified is True
    assert player.cleared >= 1  # prompt was muted on the first keypress


def test_call_verify_auto_submits_without_pound(make_client, config_factory, tmp_path):
    """The caller need not press '#': entry auto-submits after the inter-digit gap
    (so a time-based OTP isn't left to expire waiting out the full window)."""
    from api import VerifyCallRequest, OutboundCallHandler
    c, a = make_client(config_factory(data_dir=str(tmp_path),
                                      verify_dtmf_interdigit_s="0.2"))
    a.sip_handler = _FakeSip(["2", "4", "6", "8"])  # NB: no trailing '#'
    a.audio_pipeline = _FakeAudio()
    handler = OutboundCallHandler(a, call_queue=None)

    resp = asyncio.run(handler.run_verify_call(
        VerifyCallRequest(caller_id="1001", extension="1001", pin="2468")))

    assert resp.verified is True
    assert resp.attempts == 1


def test_call_verify_no_answer(client, assistant):
    from api import VerifyCallRequest
    client.post("/verify/credentials", json={"caller_id": "1001", "pin": "2468"})
    handler = _verify_handler(assistant, [])
    handler.assistant.sip_handler.call.is_active = False  # never answers

    resp = asyncio.run(handler.run_verify_call(
        VerifyCallRequest(caller_id="1001", extension="1001", ring_timeout=1)))

    assert resp.verified is False
    assert resp.status.value == "no_answer"


def test_call_verify_endpoint_requires_auth(make_client, config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path), api_auth_token="s3cret")
    c, _ = make_client(cfg)
    assert c.post("/verify/call", json={"caller_id": "1001"}).status_code == 401


# --- auth ----------------------------------------------------------------------

def test_verify_endpoints_require_auth_when_token_set(make_client, config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path), api_auth_token="s3cret")
    c, _ = make_client(cfg)

    # No credentials -> 401 on the mutating endpoints.
    assert c.post("/verify", json={"caller_id": "1001", "pin": "1"}).status_code == 401
    assert c.post("/verify/credentials",
                  json={"caller_id": "1001", "pin": "1234"}).status_code == 401

    # With the key -> succeeds.
    ok = c.post("/verify/credentials",
                json={"caller_id": "1001", "pin": "1234"},
                headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200


# --- review regressions --------------------------------------------------------

def test_get_current_otp_never_falls_back_to_global_for_unknown_caller(
        make_client, config_factory, tmp_path):
    secret = pyotp.random_base32()
    c, _ = make_client(config_factory(data_dir=str(tmp_path), verify_totp_secret=secret))
    assert c.get("/verify/otp/does-not-exist").status_code == 404
    assert c.get("/verify/otp/bad%20id").status_code == 400
    r = c.get("/verify/otp/global")
    assert r.status_code == 200
    assert r.json()["otp"] == pyotp.TOTP(secret).now()


def test_get_current_otp_global_missing(make_client, config_factory, tmp_path):
    c, _ = make_client(config_factory(data_dir=str(tmp_path)))
    assert c.get("/verify/otp/global").status_code == 404


def test_verify_endpoint_non_ascii_pin_is_false_not_500(make_client, config_factory, tmp_path):
    c, _ = make_client(config_factory(data_dir=str(tmp_path), verify_pin="1234"))
    r = c.post("/verify", json={"caller_id": "1", "pin": "１２３４"})
    assert r.status_code == 200 and r.json()["verified"] is False


def test_call_verify_caller_id_defaults_to_extension(make_client, config_factory, tmp_path):
    """Enrolled extension, no global PIN, caller_id omitted: the extension's own
    credentials are consulted (docs promise caller_id defaults to extension)."""
    from api import VerifyCallRequest
    c, a = make_client(config_factory(data_dir=str(tmp_path)))
    a.verify_store.set_credentials("1001", pin="2468")
    handler = _verify_handler(a, ["2", "4", "6", "8", "#"])
    resp = asyncio.run(handler.run_verify_call(VerifyCallRequest(extension="1001")))
    assert resp.verified is True and resp.method == "pin"


def test_call_verify_hangup_before_code_is_not_a_completed_attempt(
        make_client, config_factory, tmp_path):
    from api import VerifyCallRequest, CallStatus

    class _HangupSip(_FakeSip):
        def get_dtmf_digit(self, call_info):
            call_info.is_active = False  # caller hangs up during the prompt
            return None

    c, a = make_client(config_factory(data_dir=str(tmp_path)))
    a.sip_handler = _HangupSip([])
    a.audio_pipeline = _FakeAudio()
    from api import OutboundCallHandler
    handler = OutboundCallHandler(a, call_queue=None)
    resp = asyncio.run(handler.run_verify_call(
        VerifyCallRequest(caller_id="1001", extension="1001", pin="2468")))
    assert resp.verified is False
    assert resp.attempts == 0
    assert resp.status == CallStatus.HANGUP


def test_call_verify_empty_entry_does_not_burn_an_attempt(make_client, config_factory, tmp_path):
    """First prompt times out with nothing keyed, second gets the code: one attempt."""
    from api import VerifyCallRequest, OutboundCallHandler
    c, a = make_client(config_factory(data_dir=str(tmp_path), verify_dtmf_timeout_s="0.2",
                                      verify_dtmf_interdigit_s="0.2"))

    class _LateSip(_FakeSip):
        def __init__(self, digits):
            super().__init__(digits)
            self.calls = 0

        def get_dtmf_digit(self, call_info):
            self.calls += 1
            if self.calls < 8:  # ~0.35s of silence: first prompt window expires
                return None
            return super().get_dtmf_digit(call_info)

    a.sip_handler = _LateSip(["2", "4", "6", "8", "#"])
    a.audio_pipeline = _FakeAudio()
    handler = OutboundCallHandler(a, call_queue=None)
    resp = asyncio.run(handler.run_verify_call(
        VerifyCallRequest(caller_id="1001", extension="1001", pin="2468")))
    assert resp.verified is True
    assert resp.attempts == 1


# --- REST tool endpoints honour VERIFY_REQUIRED_TOOLS ---------------------------

def test_rest_tool_execute_is_gated(make_client, config_factory, tmp_path):
    c, a = make_client(config_factory(data_dir=str(tmp_path), verify_required_tools="CALC"))
    r = c.post("/tools/CALC/execute", json={"params": {"expression": "2+2"}})
    assert r.status_code == 403
    # a verified live session unlocks it
    from call_session import CallSession
    sess = CallSession(call_info=object(), direction="inbound", transcript_id="c1")
    sess.verified = True
    a.sessions = {"c1": sess}
    r = c.post("/tools/CALC/execute", json={"params": {"expression": "2+2"}, "call_id": "c1"})
    assert r.status_code == 200, r.text
    # ungated tools are unaffected
    r = c.post("/tools/JOKE/execute", json={"params": {}})
    assert r.status_code != 403


def test_rest_tool_call_is_gated(make_client, config_factory, tmp_path):
    c, a = make_client(config_factory(data_dir=str(tmp_path), verify_required_tools="CALC"))
    r = c.post("/tools/CALC/call", json={"extension": "1001", "params": {"expression": "2+2"}})
    assert r.status_code == 403
