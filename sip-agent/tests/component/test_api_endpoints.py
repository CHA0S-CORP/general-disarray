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


def test_transcript_endpoint_requires_auth_when_token_set(make_client, config_factory):
    """Transcripts are verbatim recordings of what callers said, and call_ids
    are guessable (prefix-<unix_second>-<n>) -- unlike the other read
    endpoints this one must not be world-readable."""
    cfg = config_factory(api_auth_token="s3cret")
    c, a = make_client(cfg)
    a.transcripts.start("t-1", "inbound", "sip:1001@host")
    a.transcripts.add_turn("t-1", "user", "my pin is 1234")

    assert c.get("/call/t-1/transcript").status_code == 401
    r = c.get("/call/t-1/transcript", headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200
    assert r.json()["turns"][0]["content"] == "my pin is 1234"


# --- /play (audio upload into the active call) -------------------------------

def _wav_bytes(sample_rate=8000, duration_s=0.25, freq=440.0):
    import io
    import wave
    import numpy as np
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    pcm = (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_play_audio_into_active_call(client, assistant):
    from types import SimpleNamespace

    sent = []

    async def send_audio(call, audio):
        sent.append((call, audio))

    assistant.sip_handler.send_audio = send_audio
    assistant.current_call = SimpleNamespace(call_id="c-1")

    r = client.post("/play", content=_wav_bytes())
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["duration_s"] == pytest.approx(0.25, abs=0.05)

    # Decoded to 16-bit mono PCM at the configured call rate.
    _, audio = sent[0]
    expected = int(assistant.config.sample_rate * 0.25) * 2
    assert abs(len(audio) - expected) <= expected * 0.1


def test_play_call_id_mismatch_is_404(client, assistant):
    from types import SimpleNamespace
    assistant.current_call = SimpleNamespace(call_id="c-1")
    r = client.post("/play", params={"call_id": "someone-else"},
                    content=_wav_bytes())
    assert r.status_code == 404


def test_play_without_active_call_is_404(client):
    assert client.post("/play", content=_wav_bytes()).status_code == 404


def test_play_undecodable_audio_is_400(client, assistant):
    from types import SimpleNamespace
    assistant.current_call = SimpleNamespace(call_id="c-1")
    r = client.post("/play", content=b"this is not audio at all")
    assert r.status_code == 400
    assert client.post("/play", content=b"").status_code == 400


def test_play_oversized_body_is_413(make_client, config_factory):
    client, assistant = make_client(config_factory(PLAY_AUDIO_MAX_BYTES="1000"))
    r = client.post("/play", content=b"\0" * 2000)
    assert r.status_code == 413


# --- /virtual-numbers (ephemeral inbound extensions) --------------------------

@pytest.fixture
def vn_client(make_client, config_factory, tmp_path):
    """Client with virtual numbers enabled and an isolated data dir."""
    return make_client(config_factory(
        VIRTUAL_NUMBERS_ENABLED="true", data_dir=str(tmp_path)))


def test_virtual_number_crud(vn_client):
    client, assistant = vn_client

    # Create (auto-allocated from the default 7300-7399 range)
    r = client.post("/virtual-numbers", json={"purpose": "pizza order pickup"})
    assert r.status_code == 200
    body = r.json()
    vn_id = body["id"]
    assert body["number"] == "7300"
    assert body["sip_uri"].startswith("sip:7300@")
    assert body["status"] == "active"

    # List + get
    r = client.get("/virtual-numbers")
    assert [e["id"] for e in r.json()] == [vn_id]
    assert client.get(f"/virtual-numbers/{vn_id}").json()["purpose"] == "pizza order pickup"

    # Delete, then 404
    assert client.delete(f"/virtual-numbers/{vn_id}").status_code == 200
    assert client.get(f"/virtual-numbers/{vn_id}").status_code == 404
    assert client.delete(f"/virtual-numbers/{vn_id}").status_code == 404


def test_virtual_number_disabled_is_403(client):
    # Default comp_config has VIRTUAL_NUMBERS_ENABLED unset (false).
    assert client.post("/virtual-numbers", json={"purpose": "x"}).status_code == 403
    assert client.get("/virtual-numbers").status_code == 403


def test_virtual_number_duplicate_is_409(vn_client):
    client, _ = vn_client
    assert client.post("/virtual-numbers",
                       json={"purpose": "a", "number": "7311"}).status_code == 200
    r = client.post("/virtual-numbers", json={"purpose": "b", "number": "7311"})
    assert r.status_code == 409


def test_virtual_number_bad_inputs_are_400_422(vn_client):
    client, _ = vn_client
    # Malformed extension
    assert client.post("/virtual-numbers",
                       json={"purpose": "x", "number": "not-a-number"}).status_code == 400
    # Missing purpose
    assert client.post("/virtual-numbers", json={}).status_code == 422
    # SSRF-blocked callback URL (private address, default config)
    r = client.post("/virtual-numbers",
                    json={"purpose": "x", "callback_url": "http://127.0.0.1/hook"})
    assert r.status_code == 400


def test_virtual_number_claim_hides_from_reuse(vn_client):
    client, assistant = vn_client
    r = client.post("/virtual-numbers", json={"purpose": "x", "number": "7322"})
    vn_id = r.json()["id"]

    entry = assistant.virtual_numbers.claim("7322")
    assert entry is not None
    assert client.get(f"/virtual-numbers/{vn_id}").json()["status"] == "claimed"

    # Consuming (call ended) removes it from the registry and the API.
    assistant.virtual_numbers.consume(vn_id)
    assert client.get(f"/virtual-numbers/{vn_id}").status_code == 404
