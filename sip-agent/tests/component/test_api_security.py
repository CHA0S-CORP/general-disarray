"""Component regressions for the REST hardening pass (real create_api app)."""
import asyncio

import pytest

pytestmark = pytest.mark.component


def _spy_tool(monkeypatch, assistant, name):
    """Replace a tool's execute with a counting stub; returns the call list."""
    calls = []
    tool = assistant.tool_manager.get_tool(name)
    real = tool.execute

    async def spy(params):
        calls.append(params)
        return await real(params)

    monkeypatch.setattr(tool, "execute", spy)
    return calls


# --- VERIFY_REQUIRED_TOOLS on every tool-running path -------------------------------

def test_webhook_call_is_gated(make_client, config_factory, tmp_path, monkeypatch):
    c, a = make_client(config_factory(data_dir=str(tmp_path), verify_required_tools="CALC"))
    calls = _spy_tool(monkeypatch, a, "CALC")
    r = c.post("/webhook/call", json={"extension": "1001", "tool": "CALC",
                                      "params": {"expression": "2+2"}})
    assert r.status_code == 403
    assert calls == []


def test_schedule_of_gated_tool_is_refused(make_client, config_factory, tmp_path):
    c, a = make_client(config_factory(data_dir=str(tmp_path), verify_required_tools="CALC"))
    r = c.post("/schedule", json={"extension": "1001", "tool": "calc",
                                  "delay_seconds": 60})
    assert r.status_code == 403
    assert not a.tool_manager.scheduled_tasks


# --- request validation runs BEFORE the tool --------------------------------------------

def test_tool_call_validates_before_executing(client, assistant, monkeypatch):
    calls = _spy_tool(monkeypatch, assistant, "CALC")
    r = client.post("/tools/CALC/call", json={
        "extension": "sip:x@evil.example", "params": {"expression": "2+2"}})
    assert r.status_code == 400
    r = client.post("/tools/CALC/call", json={
        "extension": "1001", "params": {"expression": "2+2"},
        "call_id": "bad id with spaces"})
    assert r.status_code == 400
    assert calls == []


def test_webhook_call_validates_before_executing(client, assistant, monkeypatch):
    calls = _spy_tool(monkeypatch, assistant, "CALC")
    r = client.post("/webhook/call", json={
        "extension": "1001", "tool": "CALC", "params": {"expression": "2+2"},
        "callback_url": "http://127.0.0.1/internal"})
    assert r.status_code == 400
    assert calls == []


def test_tool_call_duplicate_call_id_rejected_before_executing(client, assistant, monkeypatch):
    from api import OutboundCallRequest
    calls = _spy_tool(monkeypatch, assistant, "CALC")
    handler = client.app.state.handler
    handler.pending_calls["busy-1"] = OutboundCallRequest(message="x", extension="1001")
    try:
        r = client.post("/tools/CALC/call", json={
            "extension": "1001", "params": {"expression": "2+2"}, "call_id": "busy-1"})
        assert r.status_code == 409
        assert calls == []
    finally:
        handler.pending_calls.pop("busy-1", None)


# --- CSRF on every mutating route + Host allowlist ----------------------------------------

def test_csrf_applies_to_all_mutating_routes(client):
    cross = {"Sec-Fetch-Site": "cross-site"}
    assert client.post("/speak", params={"message": "hi"}, headers=cross).status_code == 403
    assert client.post("/tools/CALC/execute", json={"params": {"expression": "1+1"}},
                       headers=cross).status_code == 403
    assert client.delete("/schedule/nope", headers=cross).status_code == 403
    # Origin mismatch (older browsers without fetch metadata).
    r = client.post("/tools/CALC/execute", json={"params": {"expression": "1+1"}},
                    headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    # Reads and header-free (non-browser) clients are unaffected.
    assert client.get("/tools", headers=cross).status_code == 200
    r = client.post("/tools/CALC/execute", json={"params": {"expression": "1+1"}})
    assert r.status_code == 200
    r = client.post("/tools/CALC/execute", json={"params": {"expression": "1+1"}},
                    headers={"Sec-Fetch-Site": "same-origin"})
    assert r.status_code == 200


def test_tokenless_rejects_rebinding_host(client):
    r = client.get("/calls", headers={"Host": "attacker.example.com"})
    assert r.status_code == 400
    assert client.get("/calls", headers={"Host": "localhost:8080"}).status_code == 200
    assert client.get("/calls", headers={"Host": "sip-agent:8080"}).status_code == 200


def test_host_not_checked_when_token_set(make_client, config_factory):
    c, _ = make_client(config_factory(api_auth_token="s3cret"))
    r = c.get("/calls", headers={"Host": "agent.example.com",
                                 "Authorization": "Bearer s3cret"})
    assert r.status_code == 200


# --- auth -----------------------------------------------------------------------------------

def test_call_status_requires_auth(make_client, config_factory):
    c, _ = make_client(config_factory(api_auth_token="s3cret"))
    assert c.get("/call/x").status_code == 401
    r = c.get("/call/x", headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200 and r.json()["status"] == "not_found"


def test_non_ascii_token_is_401_not_500(make_client, config_factory):
    c, _ = make_client(config_factory(api_auth_token="s3cret"))
    # Raw (latin-1) header bytes decode to non-ASCII str server-side.
    r = c.get("/calls", headers={"X-API-Key": "s3cr\u00e9t".encode("utf-8")})
    assert r.status_code == 401


def test_rate_limit_keyed_on_ip_when_tokenless(make_client, config_factory):
    c, _ = make_client(config_factory(rate_limit_rpm=1, rate_limit_burst=1))
    body = {"params": {"expression": "1+1"}}
    assert c.post("/tools/CALC/execute", json=body,
                  headers={"Authorization": "Bearer a"}).status_code == 200
    # Rotating a (meaningless) auth header must not mint a fresh bucket.
    r = c.post("/tools/CALC/execute", json=body, headers={"Authorization": "Bearer b"})
    assert r.status_code == 429


# --- /schedule validation -------------------------------------------------------------------

def test_schedule_unknown_timezone_is_400_and_not_persisted(client, assistant):
    r = client.post("/schedule", json={"extension": "1001", "message": "hi",
                                       "delay_seconds": 60, "timezone": "Mars/Base"})
    assert r.status_code == 400
    assert not assistant.tool_manager.scheduled_tasks


@pytest.mark.parametrize("delay", [-5, 400 * 24 * 3600])
def test_schedule_delay_bounds(client, delay):
    r = client.post("/schedule", json={"extension": "1001", "message": "hi",
                                       "delay_seconds": delay})
    assert r.status_code == 422


def test_schedule_rejects_unsupported_recurrence(client, assistant):
    r = client.post("/schedule", json={"extension": "1001", "message": "hi",
                                       "at_time": "07:00", "recurring": "0 7 * * *"})
    assert r.status_code == 422
    assert not assistant.tool_manager.scheduled_tasks


def test_schedule_normalizes_recurrence(client, assistant):
    r = client.post("/schedule", json={"extension": "1001", "message": "hi",
                                       "at_time": "07:00", "recurring": "Weekdays"})
    assert r.status_code == 200, r.text
    assert r.json()["recurring"] == "weekdays"
    client.delete(f"/schedule/{r.json()['schedule_id']}")


# --- request size bounds --------------------------------------------------------------------

def test_oversized_fields_rejected(client):
    assert client.post("/call", json={"extension": "1001",
                                      "message": "x" * 6000}).status_code == 422
    r = client.post("/call", json={
        "extension": "1001", "message": "hi", "callback_url": "https://e.x/hook",
        "choice": {"prompt": "?", "options": [
            {"value": "yes", "synonyms": ["y"] * 51}]}})
    assert r.status_code == 422


# --- /verify brute-force lockout -------------------------------------------------------------

def test_verify_endpoint_locks_out(make_client, config_factory, tmp_path):
    c, a = make_client(config_factory(data_dir=str(tmp_path), verify_pin="4321",
                                      verify_lockout_failures=3, verify_lockout_s=900))
    for _ in range(3):
        r = c.post("/verify", json={"caller_id": "1001", "pin": "0000"})
        assert r.status_code == 200 and r.json()["verified"] is False
    r = c.post("/verify", json={"caller_id": "1001", "pin": "4321"})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0
    # The verify-call path refuses to dial a locked-out caller too.
    from api import OutboundCallHandler, RequestRejected, VerifyCallRequest
    handler = OutboundCallHandler(a, call_queue=None)
    with pytest.raises(RequestRejected) as exc:
        asyncio.run(handler.run_verify_call(VerifyCallRequest(caller_id="1001")))
    assert exc.value.status_code == 429


# --- outbound call: webhook off the slot, sanitized errors ---------------------------------

def test_execute_call_does_not_wait_for_webhook_and_hides_internals(assistant, monkeypatch):
    from types import SimpleNamespace
    from api import CallStatus, OutboundCallHandler, OutboundCallRequest

    async def boom(text):
        raise RuntimeError("redis://:hunter2@internal-host leaked")

    assistant.audio_pipeline = SimpleNamespace(synthesize=boom)
    handler = OutboundCallHandler(assistant, call_queue=None)
    released = asyncio.Event()
    delivered = []

    async def slow_webhook(url, payload):
        await released.wait()
        delivered.append(payload)

    monkeypatch.setattr(handler, "_send_webhook", slow_webhook)

    async def run():
        req = OutboundCallRequest(message="hi", extension="1001",
                                  callback_url="https://example.com/hook")
        status, error = await asyncio.wait_for(handler._execute_call("c1", req), 2)
        assert status is CallStatus.FAILED
        assert "hunter2" not in error
        assert not delivered          # returned without awaiting the webhook
        assert handler._tasks         # ...which is still held strongly
        released.set()
        await asyncio.gather(*list(handler._tasks))
        assert delivered and "hunter2" not in (delivered[0].error or "")

    asyncio.run(run())
