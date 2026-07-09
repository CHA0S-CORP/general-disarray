"""Component tests for per-call session isolation.

A replaced call's leftover tasks must write into their own (dead) session,
never into the next call's conversation — the bug class that motivated the
CallSession refactor.
"""
import asyncio
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.component


def _call(uri):
    return SimpleNamespace(is_active=True, remote_uri=uri, media_ready=False)


@pytest.fixture
def real_assistant(comp_config):
    # The real orchestrator; SIPHandler falls back to its mock without pjsua2.
    from main import SIPAIAssistant
    return SIPAIAssistant(comp_config)


async def test_new_session_gets_fresh_state(real_assistant):
    a = real_assistant
    s1 = a._begin_session(_call("sip:1001@host"), "inbound", "sip:1001@host")
    s1.conversation_history.append({"role": "user", "content": "old call"})

    assert a.current_call is s1.call_info
    assert a.conversation_history == s1.conversation_history

    await a._teardown_session()
    s2 = a._begin_session(_call("sip:1002@host"), "inbound", "sip:1002@host")

    assert a.conversation_history == []
    assert s2.transcript_id != s1.transcript_id
    # The old session's history is untouched but detached.
    assert s1.conversation_history[0]["content"] == "old call"
    await a._teardown_session()


async def test_stale_turn_cannot_leak_into_next_call(real_assistant):
    a = real_assistant
    s1 = a._begin_session(_call("sip:1001@host"), "inbound", "sip:1001@host")

    async def slow_turn():
        await asyncio.sleep(5)
        s1.conversation_history.append({"role": "assistant", "content": "stale write"})

    s1.turn_task = asyncio.create_task(slow_turn())
    await asyncio.sleep(0)  # let the task start

    # Replacing the session cancels the stale turn before the new call begins.
    await a._teardown_session()
    s2 = a._begin_session(_call("sip:1002@host"), "inbound", "sip:1002@host")

    assert s1.turn_task is None or s1.turn_task.cancelled() or s1.turn_task.done()
    assert s2.conversation_history == []
    assert all(m.get("content") != "stale write" for m in s1.conversation_history)
    await a._teardown_session()


async def test_teardown_persists_transcript(real_assistant):
    a = real_assistant
    session = a._begin_session(_call("sip:1001@host"), "inbound", "sip:1001@host")
    a.transcripts.add_turn(session.transcript_id, "user", "hello")

    await a._teardown_session()

    record = a.transcripts.get(session.transcript_id)
    assert record is not None
    assert record["ended_at"] is not None
    assert record["turns"][0]["content"] == "hello"
    assert a.session is None


# --- Call-lifecycle event webhooks ------------------------------------------

@pytest.fixture
def webhook_capture(monkeypatch):
    """Capture deliver_webhook calls; _emit_call_event's lazy `from api import
    deliver_webhook` resolves against module `api`, so patching there takes."""
    captured = []

    async def fake_deliver(url, payload, config, api_name="webhook"):
        captured.append((url, payload))
        return True

    monkeypatch.setattr("api.deliver_webhook", fake_deliver)
    return captured


def _events_assistant(config_factory, speaches_url, vllm_url, **overrides):
    from main import SIPAIAssistant
    cfg = config_factory(
        speaches_api_url=speaches_url,
        llm_base_url=f"{vllm_url}/v1",
        llm_model="mock-model",
        call_event_webhook_url="http://127.0.0.1:9/hook",
        **overrides,
    )
    return SIPAIAssistant(cfg)


async def test_call_events_emitted_on_begin_and_teardown(
        config_factory, speaches_url, vllm_url, webhook_capture):
    a = _events_assistant(config_factory, speaches_url, vllm_url)
    session = a._begin_session(_call("sip:1001@host"), "inbound", "sip:1001@host")
    a.transcripts.add_turn(session.transcript_id, "user", "hello")
    await asyncio.sleep(0)  # let the fire-and-forget task run

    assert len(webhook_capture) == 1
    url, started = webhook_capture[0]
    assert url == "http://127.0.0.1:9/hook"
    assert started["event"] == "call.started"
    assert started["call_id"] == session.transcript_id
    assert started["direction"] == "inbound"

    await a._teardown_session()
    await asyncio.sleep(0)

    assert len(webhook_capture) == 2
    _, ended = webhook_capture[1]
    assert ended["event"] == "call.ended"
    assert ended["call_id"] == session.transcript_id
    assert isinstance(ended["duration_seconds"], float)
    assert ended["transcript"]["turns"][0]["content"] == "hello"
    assert ended["transcript"]["ended_at"] is not None


async def test_call_ended_emitted_once(
        config_factory, speaches_url, vllm_url, webhook_capture):
    a = _events_assistant(config_factory, speaches_url, vllm_url)
    session = a._begin_session(_call("sip:1001@host"), "inbound", "sip:1001@host")

    # Simulate the audio-loop tail firing first, then the forced teardown.
    a.transcripts.end(session.transcript_id)
    a._emit_call_event("call.ended", session)
    await a._teardown_session()
    await asyncio.sleep(0)

    ended = [p for _, p in webhook_capture if p["event"] == "call.ended"]
    assert len(ended) == 1


async def test_call_events_filtering(
        config_factory, speaches_url, vllm_url, webhook_capture):
    a = _events_assistant(config_factory, speaches_url, vllm_url,
                          call_events="call.ended")
    a._begin_session(_call("sip:1001@host"), "inbound", "sip:1001@host")
    await a._teardown_session()
    await asyncio.sleep(0)

    assert [p["event"] for _, p in webhook_capture] == ["call.ended"]


async def test_no_events_when_url_empty(real_assistant, webhook_capture):
    a = real_assistant
    a._begin_session(_call("sip:1001@host"), "inbound", "sip:1001@host")
    await a._teardown_session()
    await asyncio.sleep(0)

    assert webhook_capture == []
