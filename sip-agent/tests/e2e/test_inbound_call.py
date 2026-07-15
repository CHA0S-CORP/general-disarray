"""E2E: place a real inbound SIP call and assert the agent answers and speaks.

Uses the deterministic SIMON_SAYS path: the spoken question is "Simon says, the
eagle has landed", so the agent's reply content is caller-controlled and immune
to LLM phrasing variance.
"""
import pytest

pytestmark = pytest.mark.e2e

REQUIRED_EVENTS = ("sip_incoming_call", "sip_call_connected", "user_speech", "assistant_response")


def _texts_for(events, name):
    return [(e.get("data") or {}).get("text", "") for e in events if e.get("event") == name]


def test_inbound_simon_says(question_wav, place_inbound_call, assert_spoke,
                            transcribe, agent_events, event_names):
    fn = question_wav("simon_says_eagle.wav")
    captured, started_at = place_inbound_call(fn, duration=30)

    # Layer 0: the agent produced real, non-silent audio.
    assert_spoke(captured)

    # Layer 1 (primary gate): the pipeline stages fired, in the log.
    events = agent_events(started_at)
    names = event_names(events)
    for required in REQUIRED_EVENTS:
        assert required in names, f"missing event '{required}'; saw {sorted(set(names))}"

    # STT produced a non-empty transcript.
    assert any(t.strip() for t in _texts_for(events, "user_speech")), "no STT transcript logged"

    # Layer 2: the deterministic phrase comes back, via the logged reply text
    # and/or the transcribed captured audio (either is sufficient; STT on the
    # recording can be noisy).
    reply_text = " ".join(_texts_for(events, "assistant_response")).lower()
    transcript = transcribe(captured).lower()
    assert "eagle has landed" in reply_text or "eagle has landed" in transcript, (
        f"phrase not found.\n  reply_text={reply_text!r}\n  transcript={transcript!r}"
    )
