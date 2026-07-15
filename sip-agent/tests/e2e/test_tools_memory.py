"""E2E: caller memory (REMEMBER / recall / FORGET) across real calls.

Caller memory is keyed by the SIP URI user part, and the default dialer has no
fixed identity — so this file defines its own dialer fixture (same docker-run
pattern as conftest's place_inbound_call) with a stable `--id sip:e2emem@tester`.

Three sequential calls from that identity:
  1. "Please remember that my favorite color is teal."  -> REMEMBER tool, or the
     post-call extractor persists it (both paths end with 'teal' on disk)
  2. "What is my favorite color?"                       -> 'teal' in the reply
  3. "Forget my favorite color."                        -> FORGET tool, or 'teal'
     gone from the persisted file after the post-call memory rewrite

The persisted memory file (data/caller_memory/e2emem.json, root-owned inside
the container) is removed via `docker exec sip-agent` before and after the run.
"""
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Register this file's question fixtures without editing gen_audio.py.
sys.path.insert(0, str(Path(__file__).parent / "audio"))
import gen_audio  # noqa: E402

gen_audio.FIXTURES.update({
    "mem_remember_teal.wav": "Please remember that my favorite color is teal.",
    "mem_ask_color.wav": "What is my favorite color?",
    "mem_forget_color.wav": "Forget my favorite color.",
})

pytestmark = pytest.mark.e2e

_AUDIO_DIR = Path(__file__).resolve().parent / "audio"
NETWORK = os.environ.get("E2E_NETWORK", "general-disarray_default")
SIP_TARGET = os.environ.get("E2E_SIP_TARGET", "sip:ai-assistant@sip-agent:5060")

# Stable caller identity -> memory file data/caller_memory/e2emem.json
CALLER_ID = "sip:e2emem@tester"
MEMORY_FILE_IN_CONTAINER = "/app/data/caller_memory/e2emem.json"


def _texts_for(events, name):
    return [(e.get("data") or {}).get("text", "") for e in events if e.get("event") == name]


def _tool_calls(events):
    """(tool, params) for every tool_call event."""
    return [
        ((e.get("data") or {}).get("tool", ""), (e.get("data") or {}).get("params"))
        for e in events if e.get("event") == "tool_call"
    ]


def _read_memory_file() -> str:
    """Contents of the persisted memory file for the test caller ('' if absent)."""
    res = subprocess.run(
        ["docker", "exec", "sip-agent", "cat", MEMORY_FILE_IN_CONTAINER],
        check=False, capture_output=True, text=True,
    )
    return res.stdout if res.returncode == 0 else ""


def _rm_memory_file():
    # The file is root-owned inside the container's data volume; remove it via
    # docker exec rather than from the host.
    subprocess.run(
        ["docker", "exec", "sip-agent", "rm", "-f", MEMORY_FILE_IN_CONTAINER],
        check=False, capture_output=True,
    )


@pytest.fixture(scope="module", autouse=True)
def clean_e2emem_memory():
    """Start from (and leave behind) no persisted memory for the test caller."""
    _rm_memory_file()
    yield
    _rm_memory_file()


@pytest.fixture
def place_memory_call(softphone_image):
    """Dial the agent with a STABLE caller identity (sip:e2emem@tester), play a
    question WAV, record the reply. Same mechanism as conftest's
    place_inbound_call — see the comments there — plus `--id` so caller memory
    keys consistently across the three calls in this file."""
    def _call(question_filename: str, duration: int = 30, capture_name: str = "mem_captured.wav"):
        captured = _AUDIO_DIR / capture_name
        if captured.exists():
            captured.unlink()
        cname = "e2e-mem-dialer"
        subprocess.run(["docker", "rm", "-f", cname], check=False, capture_output=True)
        started_at = datetime.now(timezone.utc)
        # pjsua reads commands from stdin and quits on EOF: keep stdin open for
        # the call window, then close it so pjsua finalizes the WAV and exits.
        proc = subprocess.Popen(
            [
                "docker", "run", "--rm", "-i", "--name", cname,
                "--network", NETWORK,
                "-v", f"{_AUDIO_DIR}:/audio",
                softphone_image,
                "--id", CALLER_ID,
                "--rtp-port", "4000",
                "--auto-play", "--play-file", f"/audio/{question_filename}",
                "--auto-rec", "--rec-file", f"/audio/{capture_name}",
                "--duration", str(duration),
                "--stdout-no-buf",
                SIP_TARGET,
            ],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(duration + 4)  # let the call run; it hangs up at --duration
        try:
            if proc.stdin:
                proc.stdin.close()  # EOF -> pjsua quits, finalizes WAV, container exits
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


def test_remember_favorite_color(question_wav, place_memory_call, assert_spoke,
                                 agent_events, event_names, wait_for_event):
    fn = question_wav("mem_remember_teal.wav")
    captured, started_at = place_memory_call(fn, duration=30, capture_name="mem_captured_1.wav")

    # Layer 0: real, non-silent agent speech.
    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 1: the fact was persisted. Two legitimate paths: the LLM calls the
    # REMEMBER tool mid-call, OR it just acknowledges and the post-call memory
    # extractor stores the fact (caller_memory event, fire-and-forget after
    # call_end). Accept either, then require the persisted file to hold 'teal'.
    tools = [t.upper() for t, _ in _tool_calls(events)]
    if "REMEMBER" not in tools:
        assert wait_for_event("caller_memory", started_at, timeout=30), (
            f"no REMEMBER tool_call and no caller_memory extraction; tool calls were {tools}"
        )
    content = _read_memory_file()
    assert "teal" in content.lower(), (
        f"'teal' not in persisted memory (tools={tools}).\n  memory file: {content!r}"
    )


def test_recall_favorite_color(question_wav, place_memory_call, assert_spoke,
                               transcribe, agent_events, event_names):
    # Separate call, same caller identity: the fact stored by the previous test
    # must be injected from persisted memory and used to answer.
    fn = question_wav("mem_ask_color.wav")
    captured, started_at = place_memory_call(fn, duration=30, capture_name="mem_captured_2.wav")

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 2: the remembered fact comes back. Primary signal is the logged
    # reply text; the transcript of the captured audio is a fallback.
    reply_text = " ".join(_texts_for(events, "assistant_response")).lower()
    transcript = transcribe(captured).lower()
    assert "teal" in reply_text or "teal" in transcript, (
        f"'teal' not recalled.\n  reply_text={reply_text!r}\n  transcript={transcript!r}"
    )


def test_forget_favorite_color(question_wav, place_memory_call, assert_spoke,
                               agent_events, event_names, wait_for_event):
    fn = question_wav("mem_forget_color.wav")
    captured, started_at = place_memory_call(fn, duration=30, capture_name="mem_captured_3.wav")

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 1 (primary): the FORGET tool ran. Fallback (outcome-based): after
    # the post-call memory rewrite, 'teal' is actually gone from the persisted
    # file — never weaker than the tool assert, just a different observable.
    tools = [t.upper() for t, _ in _tool_calls(events)]
    if "FORGET" not in tools:
        wait_for_event("caller_memory", started_at, timeout=30)
        content = _read_memory_file()
        assert "teal" not in content.lower(), (
            f"no FORGET tool_call (tools={tools}) and 'teal' still persisted: {content!r}"
        )
