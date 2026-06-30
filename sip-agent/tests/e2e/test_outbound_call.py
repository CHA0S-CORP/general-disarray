"""E2E: the agent places a REAL outbound call to a softphone and speaks.

Flow: POST /call -> agent dials `test-softphone` -> the resident softphone
auto-answers and records -> agent plays the TTS message -> hangs up.

Requires the stack to be started with OUTBOUND_ALLOW_SIP_URI=true (so the agent
accepts a raw `sip:` dial target); the test skips with guidance otherwise.
"""
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.e2e

OUTBOUND_EVENTS = (
    "outbound_call_initiated",
    "outbound_call_dialing",
    "outbound_call_answered",
    "outbound_call_message_played",
)
TARGET = "sip:tester@test-softphone:5060"
MESSAGE = "Simon says the eagle has landed"


def test_outbound_call(outbound_answerer, agent_post, assert_spoke, transcribe,
                       agent_events, event_names, wait_for_event):
    started_at = datetime.now(timezone.utc)
    resp = agent_post("/call", {"message": MESSAGE, "extension": TARGET})

    if resp.status_code == 401:
        pytest.skip("API_AUTH_TOKEN is set on the stack; configure auth to test outbound")
    if resp.status_code == 400:
        pytest.skip("agent rejected the sip: target; start the stack with OUTBOUND_ALLOW_SIP_URI=true")
    assert resp.status_code == 200, resp.text

    # Layer 1 (primary gate): wait for the full outbound sequence in the log.
    assert wait_for_event("outbound_call_message_played", started_at, timeout=90), \
        "agent never reported playing the outbound message"
    names = event_names(agent_events(started_at))
    for required in OUTBOUND_EVENTS:
        assert required in names, f"missing event '{required}'; saw {sorted(set(names))}"

    # Finalize the softphone recording, then Layers 0 + 2 on what it heard.
    outbound_answerer.stop_and_finalize()
    assert_spoke(outbound_answerer.captured)
    transcript = transcribe(outbound_answerer.captured).lower()
    assert "eagle has landed" in transcript, f"phrase not heard by softphone: {transcript!r}"
