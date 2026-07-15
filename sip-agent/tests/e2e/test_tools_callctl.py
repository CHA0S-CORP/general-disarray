"""E2E: call-control paths — farewell hangup, SET_TIMER, and TRANSFER.

Three real inbound calls, one question each:

  1. A pure farewell utterance ("Okay thanks, goodbye.") must take the
     deterministic hangup path: goodbye phrase + `farewell_hangup` event,
     then `call_end` — no gamble on the LLM invoking HANGUP.
  2. "Set a timer for two minutes" must route through the SET_TIMER tool.
     Timers outlive the call, so verification/cleanup happens over the REST
     API (STATUS shows the timer, CANCEL removes it) instead of a second
     call — keeps the test hermetic without cross-call state.
  3. "Transfer me to extension 405" must actually invoke the TRANSFER tool
     (or log a transfer_* event). The target doesn't exist, so the transfer
     is allowed to fail — what must NOT happen is the agent *claiming* a
     transfer with no TRANSFER tool_call behind it (a real, recently fixed
     bug).
"""
import os
import pathlib
import subprocess
import sys
import time
import uuid
import wave
from datetime import datetime, timezone

import numpy as np
import pytest

# Register this file's question fixtures with the shared TTS generator
# (gen_audio.py itself is not edited; parallel authors would conflict).
sys.path.insert(0, str(pathlib.Path(__file__).parent / "audio"))
import gen_audio  # noqa: E402

gen_audio.FIXTURES.update({
    "farewell_goodbye.wav": "Okay thanks, goodbye.",
    "timer_two_minutes.wav": "Set a timer for two minutes.",
    "transfer_ext_405.wav": "Transfer me to extension 405.",
})

pytestmark = pytest.mark.e2e

_E2E_DIR = pathlib.Path(__file__).resolve().parent
_AUDIO_DIR = _E2E_DIR / "audio"
_NETWORK = os.environ.get("E2E_NETWORK", "general-disarray_default")
_SIP_TARGET = os.environ.get("E2E_SIP_TARGET", "sip:ai-assistant@sip-agent:5060")

# Every default goodbye phrase contains "bye" ("Goodbye. Take care.",
# "Bye for now.", ...); accept close STT misreads of the reply audio too.
GOODBYE_FORMS = ("bye", "take care", "talk soon", "thanks for calling")


def _ensure_silence_wav(filename: str, seconds: float = 8.0) -> str:
    """Create a pure-silence question WAV (16 kHz mono PCM16, like gen_audio's
    fixtures) directly — no TTS involved. Used by the drain call below."""
    out = _AUDIO_DIR / filename
    if not out.exists():
        _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        samples = np.zeros(int(seconds * 16000), dtype=np.int16)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(samples.tobytes())
    return filename


@pytest.fixture
def place_transfer_call(softphone_image):
    """Like conftest's place_inbound_call, but dials with a fresh per-call
    caller identity (`--id sip:e2exfer<uuid>@tester`).

    Caller memory is keyed by the SIP URI user part and persists across calls
    and pytest runs. With the default anonymous dialer every test shares one
    caller (the softphone's IP), and repeated runs of this suite left that
    caller's memory poisoned with "frequently requests to be transferred to
    extension 405" plus a last-call summary claiming the transfer was
    "already processing" — observed live: the model then answered
    "I'm already transferring you. Please hold." on every turn without ever
    invoking TRANSFER. A unique user part per call guarantees empty memory,
    keeping the Layer-1 tool_call assert deterministic.
    """
    def _call(question_filename: str, duration: int = 30, capture_name: str = "captured.wav"):
        captured = _AUDIO_DIR / capture_name
        if captured.exists():
            captured.unlink()
        cname = "e2e-xfer-dialer"
        subprocess.run(["docker", "rm", "-f", cname], check=False, capture_output=True)
        caller_id = f"sip:e2exfer{uuid.uuid4().hex[:10]}@tester"
        started_at = datetime.now(timezone.utc)
        # Same stdin lifecycle as conftest.place_inbound_call: hold stdin OPEN
        # for the call window (pjsua quits on EOF), then close it so pjsua
        # quits cleanly, finalizes the WAV recorder, and exits the container.
        proc = subprocess.Popen(
            [
                "docker", "run", "--rm", "-i", "--name", cname,
                "--network", _NETWORK,
                "-v", f"{_AUDIO_DIR}:/audio",
                softphone_image,
                "--id", caller_id,
                "--rtp-port", "4000",
                "--auto-play", "--play-file", f"/audio/{question_filename}",
                "--auto-rec", "--rec-file", f"/audio/{capture_name}",
                "--duration", str(duration),
                "--stdout-no-buf",
                _SIP_TARGET,
            ],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(duration + 4)  # let the call run; it hangs up at --duration
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        subprocess.run(["docker", "rm", "-f", cname], check=False, capture_output=True)
        return captured, started_at
    return _call


def _texts_for(events, name):
    return [(e.get("data") or {}).get("text", "") for e in events if e.get("event") == name]


def _tool_calls(events):
    """Names of tools invoked via the `tool_call` event."""
    return [(e.get("data") or {}).get("tool", "")
            for e in events if e.get("event") == "tool_call"]


def test_farewell_hangup(question_wav, place_inbound_call, assert_spoke,
                         transcribe, agent_events, event_names, wait_for_event):
    fn = question_wav("farewell_goodbye.wav")
    captured, started_at = place_inbound_call(fn, duration=30)

    # Layer 0: the agent spoke (greeting + goodbye at minimum).
    assert_spoke(captured)

    # Layer 1 (primary gate): the deterministic farewell path fired, then the
    # agent — not the softphone's --duration timeout — ended the call.
    assert wait_for_event("farewell_hangup", started_at, timeout=60), \
        "farewell_hangup never fired for a pure-goodbye utterance"
    assert wait_for_event("call_end", started_at, timeout=60), \
        "call_end never followed farewell_hangup"

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 2: the reply is a goodbye phrase, via the logged response text
    # and/or the transcribed captured audio (either is sufficient).
    reply_text = " ".join(_texts_for(events, "assistant_response")).lower()
    transcript = transcribe(captured).lower()
    haystack = f"{reply_text} || {transcript}"
    assert any(form in haystack for form in GOODBYE_FORMS), (
        f"no goodbye phrase found.\n  reply_text={reply_text!r}\n  transcript={transcript!r}"
    )


def test_set_timer_and_rest_cleanup(question_wav, place_inbound_call, assert_spoke,
                                    agent_events, event_names, agent_post):
    # Pre-clean: drop any tasks leaked by earlier runs so STATUS is ours alone.
    agent_post("/tools/CANCEL/execute", {"params": {"task_type": "all"}})

    fn = question_wav("timer_two_minutes.wav")
    captured, started_at = place_inbound_call(fn, duration=30)

    assert_spoke(captured)

    # Layer 1 (primary gate): the LLM routed the request through SET_TIMER.
    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"
    assert "SET_TIMER" in _tool_calls(events), (
        f"no SET_TIMER tool_call; tools={_tool_calls(events)} events={sorted(set(names))}"
    )

    # Timers survive the call: verify + clean up over REST rather than a
    # second (stateful) call. CANCEL runs before the asserts so a failed
    # STATUS check can't leak a live 2-minute timer into later tests.
    status_resp = agent_post("/tools/STATUS/execute", {"params": {}})
    cancel_resp = agent_post("/tools/CANCEL/execute", {"params": {"task_type": "timer"}})

    assert status_resp.status_code == 200, status_resp.text
    status_msg = (status_resp.json().get("message") or "").lower()
    assert "timer" in status_msg, f"STATUS does not mention the timer: {status_msg!r}"

    assert cancel_resp.status_code == 200, cancel_resp.text
    cancel_body = cancel_resp.json()
    cancelled = (cancel_body.get("data") or {}).get("cancelled_count", 0)
    assert cancelled >= 1, f"CANCEL removed nothing: {cancel_body!r}"


def test_transfer_invokes_tool(question_wav, place_transfer_call, assert_spoke,
                               agent_events, event_names, agent_post):
    # Drain call: the agent's audio_pipeline VAD buffer is a singleton that is
    # NOT reset between calls, so a call that hangs up mid-utterance leaks its
    # last words into the next call's first turn (observed live: this call
    # heard "Set a timer for two minutes." from the previous test, and with
    # that prior exchange in context the LLM then answered every transfer
    # request with "I'll transfer you. Please hold." — never invoking the
    # tool). A short all-silence call absorbs any leaked buffer and, sending
    # only silence, ends with the buffer clean.
    drain = _ensure_silence_wav("drain_silence.wav")
    place_transfer_call(drain, duration=10, capture_name="drain_captured.wav")
    # A leaked "set a timer" utterance flushed into the drain call may have
    # started a stray timer — clear it so later tests see clean STATUS.
    agent_post("/tools/CANCEL/execute", {"params": {"task_type": "all"}})

    # Fresh caller id per call: the shared dialer's memory is poisoned with
    # transfer-was-already-processing facts (see place_transfer_call).
    fn = question_wav("transfer_ext_405.wav")
    captured, started_at = place_transfer_call(fn, duration=30)

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 1 (primary gate): real transfer evidence. Extension 405 doesn't
    # exist, so the transfer may fail (transfer_target_blocked or a failed
    # REFER are both fine) — but the TRANSFER tool must actually have been
    # invoked. An agent that merely *says* it transferred, with no tool_call
    # and no transfer_* event, is the exact bug this test guards against.
    transfer_events = [n for n in names if "transfer" in n.lower()]
    assert "TRANSFER" in _tool_calls(events) or transfer_events, (
        f"no TRANSFER tool_call and no transfer event; "
        f"tools={_tool_calls(events)} events={sorted(set(names))}"
    )
