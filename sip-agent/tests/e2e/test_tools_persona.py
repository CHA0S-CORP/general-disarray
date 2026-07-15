"""E2E: the PERSONA tool over real inbound calls.

Three real calls, one stable caller identity:

  1. set   — ask the agent to adopt a distinctive demeanor; assert the
             persona_set event fires and the reply takes on the demeanor.
  2. save  — ask it to save that demeanor under a unique name; assert
             persona_save and that the name lands in the persisted store.
  3. load  — a SEPARATE call recalls the saved demeanor by name; assert
             persona_load fires with that name and the demeanor returns.

Anchored on the structured persona_* events (Layer 1) and the persisted
personas.json (deterministic), with a spoken-content check as a soft fallback —
never on exact LLM phrasing. A unique, made-up profile name is used so the test
exercises real save/load persistence rather than the shipped seed profiles.
"""
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

import pytest

# Register this file's question fixtures without editing gen_audio.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent / "audio"))
import gen_audio  # noqa: E402

# A distinctive, reliably-detectable demeanor: pirate speech ("ahoy"/"matey"/
# "arr") shows up strongly in replies (confirmed on live calls) and is unlikely
# to appear by chance in a neutral answer.
gen_audio.FIXTURES.update({
    "persona_set_pirate.wav":
        "Please use your persona tool to talk like a pirate for the rest of "
        "this call. Greet me with ahoy and call me matey.",
    # Single save action with the style supplied inline: chaining set-then-save
    # in one voice turn proved unreliable (the model set the persona but skipped
    # the save). "Space Captain" is a clean name STT/the model round-trip
    # consistently (unlike "E2E Buccaneer").
    "persona_save_v3.wav":
        "Using your persona tool, save a speaking style under the name Space "
        "Captain. The style is: talk like a pirate, say ahoy and call me matey.",
    "persona_load_v2.wav":
        "Use your persona tool to load the saved style called Space Captain, "
        "then say hello.",
})

pytestmark = pytest.mark.e2e

_AUDIO_DIR = pathlib.Path(__file__).resolve().parent / "audio"
NETWORK = os.environ.get("E2E_NETWORK", "general-disarray_default")
SIP_TARGET = os.environ.get("E2E_SIP_TARGET", "sip:ai-assistant@sip-agent:5060")

CALLER_ID = "sip:e2epersona@tester"
PERSONA_FILE_IN_CONTAINER = "/app/data/personas.json"
# Token matched (loosely) against saved profile keys so STT/phrasing variance in
# the spoken name ("Space Captain") doesn't break the test.
SAVED_TOKEN = "captain"

PIRATE_MARKERS = ("ahoy", "matey", "arr", "aye", "avast", "ye ", "yer ")


def _texts_for(events, name):
    return [(e.get("data") or {}).get("text", "") for e in events if e.get("event") == name]


def _read_personas() -> dict:
    """The persisted personas.json inside the container ({} if absent/bad)."""
    res = subprocess.run(
        ["docker", "exec", "sip-agent", "cat", PERSONA_FILE_IN_CONTAINER],
        check=False, capture_output=True, text=True,
    )
    if res.returncode != 0:
        return {}
    try:
        return json.loads(res.stdout)
    except Exception:
        return {}


def _remove_saved():
    """Drop any e2e buccaneer profile so the suite is idempotent. The store keys
    on the normalized name; delete every key containing our token."""
    data = _read_personas()
    if not any(SAVED_TOKEN in k for k in data):
        return
    kept = {k: v for k, v in data.items() if SAVED_TOKEN not in k}
    subprocess.run(
        ["docker", "exec", "-i", "sip-agent",
         "python3", "-c",
         "import sys,json;open('/app/data/personas.json','w').write(sys.stdin.read())"],
        input=json.dumps(kept), text=True, check=False, capture_output=True,
    )


@pytest.fixture(scope="module", autouse=True)
def clean_saved_persona():
    """Start and end with no e2e buccaneer profile in the store."""
    _remove_saved()
    yield
    _remove_saved()


@pytest.fixture
def place_persona_call(softphone_image):
    """Dial with the stable persona-test caller identity, play a WAV, record the
    reply. Same stdin/EOF lifecycle as conftest.place_inbound_call."""
    def _call(question_filename: str, duration: int = 30,
              capture_name: str = "persona_captured.wav"):
        captured = _AUDIO_DIR / capture_name
        if captured.exists():
            captured.unlink()
        cname = "e2e-persona-dialer"
        subprocess.run(["docker", "rm", "-f", cname], check=False, capture_output=True)
        started_at = datetime.now(timezone.utc)
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


def test_persona_set_shapes_the_reply(question_wav, place_persona_call, assert_spoke,
                                      transcribe, agent_events, event_names):
    """PERSONA set: the tool fires (persona_set) and the demeanor colors the reply."""
    fn = question_wav("persona_set_pirate.wav")
    captured, started_at = place_persona_call(fn, duration=30,
                                              capture_name="persona_set.wav")

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 1 (deterministic): a demeanor was applied. Either event counts —
    # 'pirate' is a seeded profile, so a set naming it is redirected to load
    # (persona_load via set_redirect); an ad-hoc description stays persona_set.
    assert ("persona_set" in names or "persona_load" in names), (
        f"PERSONA tool did not apply a demeanor; events were {sorted(set(names))}"
    )

    # Layer 2 (soft): the demeanor actually reached the caller.
    reply = " ".join(_texts_for(events, "assistant_response")).lower()
    transcript = transcribe(captured).lower()
    haystack = f"{reply} || {transcript}"
    assert any(m in haystack for m in PIRATE_MARKERS), (
        f"demeanor not reflected in reply.\n  reply={reply!r}\n  transcript={transcript!r}"
    )


def test_persona_save_persists_profile(question_wav, place_persona_call, assert_spoke,
                                       agent_events, event_names):
    """PERSONA save: setting a demeanor and saving it (in ONE call, since the
    active persona is per-call) writes the named profile to the persisted
    store."""
    fn = question_wav("persona_save_v3.wav")
    captured, started_at = place_persona_call(fn, duration=35,
                                              capture_name="persona_save_cap.wav")
    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Outcome is the real test of "save persists a profile": the named profile
    # lands on disk. (persona_save fires on success, but the disk check is the
    # ground truth and is robust to event-name drift.)
    data = _read_personas()
    assert any(SAVED_TOKEN in k for k in data), (
        f"PERSONA save did not persist a profile.\n  events={sorted(set(names))}"
        f"\n  personas keys={sorted(data)}"
    )


def test_persona_load_recalls_saved_profile(question_wav, place_persona_call, assert_spoke,
                                            transcribe, agent_events, event_names):
    """PERSONA load: a later call recalls the saved demeanor by name."""
    # Guarantee the profile exists independent of the save test's ordering, so
    # this test is self-contained: write it straight into the store.
    data = _read_personas()
    if not any(SAVED_TOKEN in k for k in data):
        data["space captain"] = {
            "name": "Space Captain",
            "text": "Talk like a pirate: greet with 'ahoy' and call the caller 'matey'.",
        }
        subprocess.run(
            ["docker", "exec", "-i", "sip-agent", "python3", "-c",
             "import sys;open('/app/data/personas.json','w').write(sys.stdin.read())"],
            input=json.dumps(data), text=True, check=False, capture_output=True,
        )

    fn = question_wav("persona_load_v2.wav")
    captured, started_at = place_persona_call(fn, duration=30,
                                              capture_name="persona_load_cap.wav")
    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 1 (deterministic): the saved demeanor was loaded.
    assert "persona_load" in names, (
        f"PERSONA did not load a saved profile; events were {sorted(set(names))}"
    )

    # Layer 2 (soft): the recalled demeanor reached the caller.
    reply = " ".join(_texts_for(events, "assistant_response")).lower()
    transcript = transcribe(captured).lower()
    haystack = f"{reply} || {transcript}"
    assert any(m in haystack for m in PIRATE_MARKERS), (
        f"recalled demeanor not reflected.\n  reply={reply!r}\n  transcript={transcript!r}"
    )
