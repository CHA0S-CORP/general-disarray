"""Component tests for the REST surface with 0, 1, or several active calls.

/speak (and hangup) defaulting rules: no call -> 404, one call -> routed to
it, several calls -> 409 listing the active call ids unless call_id picks one.
"""
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.component


def _mk_session(tid, remote):
    return SimpleNamespace(
        transcript_id=tid,
        direction="inbound",
        start_time=time.time(),
        call_info=SimpleNamespace(call_id=f"{tid}-sip", remote_uri=remote,
                                  is_active=True),
        conversation_history=[],
    )


def _wire_audio(assistant):
    """Give the FakeAssistant a speakable audio path; returns played calls."""
    sent = []

    async def synthesize(text):
        return b"\x00\x01" * 40

    async def send_audio(call_info, audio, tag=None):
        sent.append(call_info)

    assistant.audio_pipeline = SimpleNamespace(synthesize=synthesize)
    assistant.sip_handler.send_audio = send_audio
    return sent


# --- /speak -------------------------------------------------------------------

def test_speak_with_no_active_call_is_404(client, assistant):
    _wire_audio(assistant)
    r = client.post("/speak", params={"message": "hello"})
    assert r.status_code == 404


def test_speak_with_one_active_call_routes_to_it(client, assistant):
    sent = _wire_audio(assistant)
    s1 = _mk_session("in-1", "sip:42@host")
    assistant.sessions = {"a": s1}

    r = client.post("/speak", params={"message": "hello"})
    assert r.status_code == 200
    assert sent == [s1.call_info]


def test_speak_with_two_active_calls_is_409_with_ids(client, assistant):
    sent = _wire_audio(assistant)
    s1, s2 = _mk_session("in-1", "sip:42@host"), _mk_session("in-2", "sip:43@host")
    assistant.sessions = {"a": s1, "b": s2}

    r = client.post("/speak", params={"message": "hello"})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert sorted(detail["active_call_ids"]) == ["in-1", "in-2"]
    assert sent == []


def test_speak_with_call_id_routes_to_that_call(client, assistant):
    sent = _wire_audio(assistant)
    s1, s2 = _mk_session("in-1", "sip:42@host"), _mk_session("in-2", "sip:43@host")
    assistant.sessions = {"a": s1, "b": s2}

    r = client.post("/speak", params={"message": "hello", "call_id": "in-2"})
    assert r.status_code == 200
    assert sent == [s2.call_info]

    # The SIP-level call id works too.
    r = client.post("/speak", params={"message": "hi", "call_id": "in-1-sip"})
    assert r.status_code == 200
    assert sent == [s2.call_info, s1.call_info]


def test_speak_with_unknown_call_id_is_404(client, assistant):
    _wire_audio(assistant)
    assistant.sessions = {"a": _mk_session("in-1", "sip:42@host")}
    r = client.post("/speak", params={"message": "hello", "call_id": "nope"})
    assert r.status_code == 404


# --- /calls/active + hangup with several calls ---------------------------------

def test_active_calls_lists_all_sessions(client, assistant):
    s1, s2 = _mk_session("in-1", "sip:42@host"), _mk_session("in-2", "sip:43@host")
    assistant.sessions = {"a": s1, "b": s2}
    body = client.get("/calls/active").json()
    assert body["active"] is True
    assert body["count"] == 2
    assert sorted(c["call_id"] for c in body["calls"]) == ["in-1", "in-2"]


def test_hangup_with_two_calls_requires_call_id(client, assistant):
    hung = []

    async def hangup_call(call_info):
        hung.append(call_info)

    assistant.sip_handler.hangup_call = hangup_call
    s1, s2 = _mk_session("in-1", "sip:42@host"), _mk_session("in-2", "sip:43@host")
    assistant.sessions = {"a": s1, "b": s2}

    r = client.post("/calls/active/hangup")
    assert r.status_code == 409
    assert sorted(r.json()["detail"]["active_call_ids"]) == ["in-1", "in-2"]
    assert hung == []

    r = client.post("/calls/active/hangup", params={"call_id": "in-2"})
    assert r.status_code == 200
    assert r.json()["call_id"] == "in-2"
    assert hung == [s2.call_info]

    r = client.post("/calls/active/hangup", params={"call_id": "gone"})
    assert r.status_code == 404


# --- /play (same ambiguity rule as /speak) -------------------------------------

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


def test_play_with_two_active_calls_is_409_with_ids(client, assistant):
    sent = _wire_audio(assistant)
    s1, s2 = _mk_session("in-1", "sip:42@host"), _mk_session("in-2", "sip:43@host")
    assistant.sessions = {"a": s1, "b": s2}

    r = client.post("/play", content=_wav_bytes())
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert sorted(detail["active_call_ids"]) == ["in-1", "in-2"]
    assert sent == []


def test_play_with_call_id_routes_to_that_call(client, assistant):
    sent = _wire_audio(assistant)
    s1, s2 = _mk_session("in-1", "sip:42@host"), _mk_session("in-2", "sip:43@host")
    assistant.sessions = {"a": s1, "b": s2}

    r = client.post("/play", params={"call_id": "in-2"}, content=_wav_bytes())
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert sent == [s2.call_info]


# --- /tools/{name}/execute with speak_result ------------------------------------

def test_tool_execute_speak_result_ambiguity_downgrades_to_unspoken(client, assistant):
    """With two active calls and no call_id, the tool call itself must still
    succeed — ambiguity only means the result couldn't be spoken anywhere
    (spoken: false), not a 409/500."""
    sent = _wire_audio(assistant)
    s1, s2 = _mk_session("in-1", "sip:42@host"), _mk_session("in-2", "sip:43@host")
    assistant.sessions = {"a": s1, "b": s2}

    r = client.post("/tools/CALC/execute",
                    json={"params": {"expression": "2+2"}, "speak_result": True})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["spoken"] is False
    assert sent == []


def test_tool_execute_speak_result_with_call_id_speaks_to_that_call(client, assistant):
    sent = _wire_audio(assistant)
    s1, s2 = _mk_session("in-1", "sip:42@host"), _mk_session("in-2", "sip:43@host")
    assistant.sessions = {"a": s1, "b": s2}

    r = client.post("/tools/CALC/execute",
                    json={"params": {"expression": "2+2"},
                          "speak_result": True, "call_id": "in-2"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["spoken"] is True
    assert sent == [s2.call_info]
