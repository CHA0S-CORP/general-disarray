"""E2E: ops + knowledge tools over real calls (GPU_STATUS, ALERTS, KNOWLEDGE).

Each test places one inbound call, asks a question that routes to the tool, and
asserts Layer 0 (non-silent reply audio) + Layer 1 (a `tool_call` event naming
the tool). Content asserts (Layer 1.5/2) are kept loose where the backing
service's answer varies:

  - GPU_STATUS: numbers change, but the reply always carries a unit
    (percent utilization and/or degrees temperature).
  - ALERTS: Alertmanager may be unconfigured/empty on the test stack, so any
    graceful spoken answer is fine — only the tool_call is asserted.
  - KNOWLEDGE: anchored on a distinctive fact from
    data/knowledge/example-house-notes.md (trash pickup is Tuesday morning),
    which the model can only know via RAG.

Stabilizations learned from the live stack:

  - Questions explicitly ask for a check ("check the GPU status", "check
    whether any monitoring alerts...", "check your knowledge base...") — with
    vaguer phrasing the model was observed answering from thin air ("No
    alerts are currently firing", fabricated GPU numbers) without invoking
    any tool.
  - The KNOWLEDGE probe avoids the guest Wi-Fi SSID: the live model treats
    network names/credentials as private and refuses ("It's not available for
    sharing") instead of searching, even when told the KB has it. The trash
    schedule is a neutral fact it looks up willingly.
  - Calls are placed with a FRESH random caller id per call (local dialer
    below): the default dialer's identity resolves to its container IP, whose
    caller-memory file accumulates junk "facts" across every e2e run (40+
    calls observed) that get injected into the system prompt and bias the
    model into answering ops questions from memory instead of tools.
  - Each test first places a short benign "flush" call: the agent replays the
    previous call's last in-flight utterance as the *next* call's first turn
    (stale-transcript bleed, observed live even minutes apart), and a foreign
    stale question can poison the model into tool-less fabricated answers.
    The flush call absorbs the foreign bleed, and its own harmless "Hello
    there" is what bleeds into the real measured call.
"""
import os
import pathlib
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

import pytest

# Register this file's question fixtures without editing gen_audio.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent / "audio"))
import gen_audio  # noqa: E402

# NB: ensure_question_wav caches by filename, so reworded questions get NEW
# filenames (v2) to avoid replaying a stale cached WAV.
gen_audio.FIXTURES.update({
    "hello_flush.wav": "Hello there.",
    "gpu_status_question_v2.wav":
        "Please check the GPU status. What are the utilization and temperature right now?",
    "alerts_question_v2.wav":
        "Please check whether any monitoring alerts are currently firing.",
    "knowledge_trash.wav":
        "Check your knowledge base. Which day does the trash get picked up?",
})

pytestmark = pytest.mark.e2e

# Mirror the conftest dialer's environment (can't import conftest directly).
_AUDIO_DIR = pathlib.Path(__file__).resolve().parent / "audio"
_NETWORK = os.environ.get("E2E_NETWORK", "general-disarray_default")
_SIP_TARGET = os.environ.get("E2E_SIP_TARGET", "sip:ai-assistant@sip-agent:5060")


def _texts_for(events, name):
    return [(e.get("data") or {}).get("text", "") for e in events if e.get("event") == name]


def _tools_called(events):
    """Names of tools invoked via `tool_call` events (uppercased)."""
    tools = []
    for e in events:
        name = e.get("event") or (e.get("data") or {}).get("event")
        if name == "tool_call":
            tools.append(((e.get("data") or {}).get("tool") or "").upper())
    return tools


@pytest.fixture
def place_call_fresh_caller(softphone_image):
    """Like conftest's place_inbound_call, but dials with a FRESH random SIP
    identity per call (--id sip:e2eops<hex>@tester) so the agent's caller
    memory starts empty — see the module docstring for why that matters."""
    def _call(question_filename: str, duration: int = 30, capture_name: str = "captured.wav"):
        captured = _AUDIO_DIR / capture_name
        if captured.exists():
            captured.unlink()
        cname = "e2e-ops-dialer"
        subprocess.run(["docker", "rm", "-f", cname], check=False, capture_output=True)
        started_at = datetime.now(timezone.utc)
        # Same stdin lifecycle as the conftest dialer: keep stdin OPEN during
        # the call, close it afterwards so pjsua quits and finalizes the WAV.
        proc = subprocess.Popen(
            [
                "docker", "run", "--rm", "-i", "--name", cname,
                "--network", _NETWORK,
                "-v", f"{_AUDIO_DIR}:/audio",
                softphone_image,
                "--id", f"sip:e2eops{uuid.uuid4().hex[:10]}@tester",
                "--rtp-port", "4000",
                "--auto-play", "--play-file", f"/audio/{question_filename}",
                "--auto-rec", "--rec-file", f"/audio/{capture_name}",
                "--duration", str(duration),
                "--stdout-no-buf",
                _SIP_TARGET,
            ],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(duration + 4)
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


@pytest.fixture
def flush_stale_audio(question_wav, place_call_fresh_caller):
    """Absorb the previous call's stale-transcript bleed with a benign call.

    Deliberately NOT a farewell phrase ("thanks, goodbye" would bleed forward
    and trigger farewell_hangup at the start of the measured call)."""
    fn = question_wav("hello_flush.wav")
    place_call_fresh_caller(fn, duration=12, capture_name="flush_captured.wav")


def test_gpu_status_over_call(flush_stale_audio, question_wav, place_call_fresh_caller,
                              assert_spoke, transcribe, agent_events, event_names):
    fn = question_wav("gpu_status_question_v2.wav")
    captured, started_at = place_call_fresh_caller(fn, duration=30)

    # Layer 0: the agent produced real, non-silent audio.
    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 1: the GPU_STATUS tool actually ran.
    tools = _tools_called(events)
    assert "GPU_STATUS" in tools, f"GPU_STATUS not invoked; tools called: {tools}"

    # The numbers vary, but a real GPU report always carries a unit —
    # utilization (percent) and/or temperature (degrees/Celsius).
    unit_forms = ("percent", "%", "degree", "celsius")
    reply_text = " ".join(_texts_for(events, "assistant_response")).lower()
    transcript = transcribe(captured).lower()
    haystack = f"{reply_text} || {transcript}"
    assert any(form in haystack for form in unit_forms), (
        f"no percent/degrees unit in reply.\n  reply_text={reply_text!r}\n"
        f"  transcript={transcript!r}"
    )


def test_alerts_over_call(flush_stale_audio, question_wav, place_call_fresh_caller,
                          assert_spoke, agent_events, event_names):
    fn = question_wav("alerts_question_v2.wav")
    captured, started_at = place_call_fresh_caller(fn, duration=30)

    # Layer 0: the agent spoke *something* — Alertmanager may be unconfigured
    # or have zero firing alerts, and a graceful spoken answer is acceptable
    # either way, so no content assert here.
    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 1 (the real gate): the ALERTS tool was invoked for the question.
    tools = _tools_called(events)
    assert "ALERTS" in tools, f"ALERTS not invoked; tools called: {tools}"


def test_knowledge_over_call(flush_stale_audio, question_wav, place_call_fresh_caller,
                             assert_spoke, transcribe, agent_events, event_names):
    fn = question_wav("knowledge_trash.wav")
    captured, started_at = place_call_fresh_caller(fn, duration=30)

    # Layer 0: the agent produced real, non-silent audio.
    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 1: the KNOWLEDGE (RAG) tool actually ran.
    tools = _tools_called(events)
    assert "KNOWLEDGE" in tools, f"KNOWLEDGE not invoked; tools called: {tools}"

    # Layer 2: the pickup day from example-house-notes.md comes back. The doc
    # is the only place the agent can learn it, so its presence proves
    # retrieval worked end-to-end.
    reply_text = " ".join(_texts_for(events, "assistant_response")).lower()
    transcript = transcribe(captured).lower()
    haystack = f"{reply_text} || {transcript}"
    assert "tuesday" in haystack, (
        f"'Tuesday' (trash day from the knowledge doc) not found.\n"
        f"  reply_text={reply_text!r}\n  transcript={transcript!r}"
    )
