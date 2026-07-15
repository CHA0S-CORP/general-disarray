"""E2E: a real call that asks a math question and gets the correct spoken answer.

Asks "What is seventeen times three?" and expects 51 in the reply. The model may
either invoke the CALC tool or self-compute such small arithmetic — both are
correct, so we assert on the *answer* (deterministic) rather than requiring a
tool_call. The tool-execution path itself is covered deterministically by the
SIMON_SAYS inbound test, which the LLM reliably routes through the tool.
"""
import pytest

pytestmark = pytest.mark.e2e

# Accept digit or word forms; STT/LLM may render either.
ANSWER_FORMS = ("51", "fifty one", "fifty-one")


def _texts_for(events, name):
    return [(e.get("data") or {}).get("text", "") for e in events if e.get("event") == name]


def test_calc_over_call(question_wav, place_inbound_call, assert_spoke,
                        transcribe, agent_events, event_names):
    fn = question_wav("calc_17x3.wav")
    captured, started_at = place_inbound_call(fn, duration=30)

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    reply_text = " ".join(_texts_for(events, "assistant_response")).lower()
    transcript = transcribe(captured).lower()
    haystack = f"{reply_text} || {transcript}"
    assert any(form in haystack for form in ANSWER_FORMS), (
        f"answer 51 not found.\n  reply_text={reply_text!r}\n  transcript={transcript!r}"
    )
