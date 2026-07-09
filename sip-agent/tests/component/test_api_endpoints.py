"""Component tests for the FastAPI surface via Starlette's TestClient.

Exercises the real create_api() app against a fake assistant with the real tool
manager. No SIP, no phone, no external services for these paths.
"""
import pytest

pytestmark = pytest.mark.component


# --- health / tools listing ------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["sip_registered"] is False


def test_call_transcript_endpoint(client, assistant):
    assistant.transcripts.start("t-1", "inbound", "sip:1001@host")
    assistant.transcripts.add_turn("t-1", "user", "hello there")
    assistant.transcripts.add_turn("t-1", "assistant", "hi, how can I help?")

    r = client.get("/call/t-1/transcript")   # live call
    assert r.status_code == 200
    assert [t["content"] for t in r.json()["turns"]] == ["hello there", "hi, how can I help?"]

    assistant.transcripts.end("t-1")
    r = client.get("/call/t-1/transcript")   # finished call (LRU/disk)
    assert r.status_code == 200
    assert r.json()["ended_at"] is not None

    assert client.get("/call/nope/transcript").status_code == 404
    # Path-traversal-shaped ids must not read arbitrary files.
    assert client.get("/call/..%2F..%2Fetc%2Fpasswd/transcript").status_code == 404


def test_health_deep_reports_dependencies(client):
    r = client.get("/health", params={"deep": "true"})
    assert r.status_code == 200
    body = r.json()
    # Always reports all three dependencies; up/down depends on the environment.
    assert set(body["dependencies"]) == {"vllm", "speaches", "redis"}
    assert body["status"] in ("healthy", "degraded")


def test_list_tools(client):
    r = client.get("/tools")
    assert r.status_code == 200
    names = {t["name"] for t in r.json()}
    assert {"CALC", "SIMON_SAYS", "JOKE", "DATETIME"} <= names


def test_get_tool(client):
    r = client.get("/tools/CALC")
    assert r.status_code == 200
    assert r.json()["name"] == "CALC"


def test_get_unknown_tool(client):
    assert client.get("/tools/NOPE").status_code == 404


# --- tool execution (deterministic CALC) -----------------------------------

def test_execute_calc(client):
    r = client.post("/tools/CALC/execute", json={"params": {"expression": "2+2"}})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["tool"] == "CALC"
    assert "4" in body["message"]
    assert body["data"]["result"] == 4


def test_execute_unknown_tool(client):
    r = client.post("/tools/NOPE/execute", json={"params": {}})
    assert r.status_code == 404


def test_execute_tool_path_body_mismatch(client):
    r = client.post("/tools/CALC/execute", json={"tool": "JOKE", "params": {}})
    assert r.status_code == 400


# --- /call request validation ----------------------------------------------

def test_call_rejects_raw_sip_uri(client):
    # Default config: outbound_allow_sip_uri False -> 400 RequestRejected.
    r = client.post("/call", json={"message": "hi", "extension": "sip:x@evil.example"})
    assert r.status_code == 400


def test_call_missing_message_is_422(client):
    r = client.post("/call", json={"extension": "1001"})
    assert r.status_code == 422  # pydantic validation


def test_call_choice_without_callback_is_422(client):
    r = client.post("/call", json={
        "message": "confirm?",
        "extension": "1001",
        "choice": {"prompt": "yes?", "options": [{"value": "yes"}]},
    })
    assert r.status_code == 422  # model_validator: callback_url required for choice


# --- /schedule CRUD --------------------------------------------------------

def test_schedule_crud(client):
    # Create (far enough out that it never fires during the test).
    r = client.post("/schedule", json={
        "extension": "1001", "message": "wake up", "delay_seconds": 3600,
    })
    assert r.status_code == 200
    schedule_id = r.json()["schedule_id"]

    # List includes it.
    r = client.get("/schedule")
    assert r.status_code == 200
    assert any(s["schedule_id"] == schedule_id for s in r.json())

    # Get single.
    r = client.get(f"/schedule/{schedule_id}")
    assert r.status_code == 200
    assert r.json()["extension"] == "1001"

    # Delete, then it is gone.
    r = client.delete(f"/schedule/{schedule_id}")
    assert r.status_code in (200, 204)
    assert client.get(f"/schedule/{schedule_id}").status_code == 404


def test_schedule_rejects_bad_extension(client):
    r = client.post("/schedule", json={
        "extension": "sip:x@evil.example", "message": "hi", "delay_seconds": 60,
    })
    assert r.status_code == 400


# --- auth ------------------------------------------------------------------

def test_rate_limit_enforced(make_client, config_factory):
    cfg = config_factory(rate_limit_rpm=60, rate_limit_burst=2)
    c, _ = make_client(cfg)

    body = {"params": {"expression": "1+1"}}
    assert c.post("/tools/CALC/execute", json=body).status_code == 200
    assert c.post("/tools/CALC/execute", json=body).status_code == 200
    # Burst of 2 exhausted -> 429.
    assert c.post("/tools/CALC/execute", json=body).status_code == 429
    # Read-only endpoints are not rate limited.
    assert c.get("/health").status_code == 200


def test_auth_enforced_when_token_set(make_client, config_factory):
    cfg = config_factory(api_auth_token="s3cret")
    c, _ = make_client(cfg)

    # Mutating endpoint without credentials -> 401.
    assert c.post("/tools/CALC/execute", json={"params": {"expression": "1+1"}}).status_code == 401
    # With the right key -> 200.
    ok = c.post(
        "/tools/CALC/execute",
        json={"params": {"expression": "1+1"}},
        headers={"X-API-Key": "s3cret"},
    )
    assert ok.status_code == 200
    # Read-only health stays open.
    assert c.get("/health").status_code == 200


# --- reformat_for_speech -----------------------------------------------------

def test_call_reformats_message_when_flagged(client, assistant):
    r = client.post("/call", json={
        "message": "ALERT: p99=340ms", "extension": "1001",
        "reformat_for_speech": True,
    })
    assert r.status_code == 200
    assert assistant.llm_engine.reformat_calls == ["ALERT: p99=340ms"]


def test_call_skips_reformat_by_default(client, assistant):
    r = client.post("/call", json={"message": "plain text", "extension": "1001"})
    assert r.status_code == 200
    assert assistant.llm_engine.reformat_calls == []


def test_schedule_stores_reformat_flag(client, assistant):
    r = client.post("/schedule", json={
        "extension": "1001", "message": "wake up", "delay_seconds": 3600,
        "reformat_for_speech": True,
    })
    assert r.status_code == 200
    task = assistant.tool_manager.scheduled_tasks[r.json()["schedule_id"]]
    assert task.metadata["reformat_for_speech"] is True
    client.delete(f"/schedule/{r.json()['schedule_id']}")
