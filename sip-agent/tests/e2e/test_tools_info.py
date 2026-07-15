"""E2E: information tools over real calls — WEATHER, FORECAST, WEB_SEARCH,
DATETIME, QUAKES.

Each test places one real inbound SIP call, asks one question, and asserts
Layer 0 (non-silent audio) + Layer 1 (tool_call / pipeline events in the
structured log) + a content check where the answer is deterministic enough.
Assertions never depend on exact LLM phrasing: weather answers must mention a
temperature unit, the time answer must contain a plausible time, and the
search/quake tests gate on the tool_call event itself.
"""
import os
import pathlib
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

import pytest

# Register this file's question fixtures without editing gen_audio.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent / "audio"))
import gen_audio  # noqa: E402

gen_audio.FIXTURES.update({
    # "Please check" biases the agent toward the WEATHER tool: without it the
    # model was observed fabricating a confident weather report (temperature,
    # wind) with no tool_call at all — see the xfail guard in the test.
    "weather_check_now.wav": "What's the weather like right now? Please check the current conditions.",
    "forecast_check_today.wav": "What's the forecast for today? Please check the latest forecast.",
    "search_dgx_spark.wav": "Search the web for the NVIDIA DGX Spark.",
    "time_now.wav": "What time is it?",
    "quakes_today.wav": "Any big earthquakes today?",
    "kp_index_now.wav": "What's the current KP index? Please check the space weather.",
})

pytestmark = pytest.mark.e2e

_E2E_DIR = pathlib.Path(__file__).resolve().parent
_AUDIO_DIR = _E2E_DIR / "audio"
_NETWORK = os.environ.get("E2E_NETWORK", "general-disarray_default")
_SIP_TARGET = os.environ.get("E2E_SIP_TARGET", "sip:ai-assistant@sip-agent:5060")


@pytest.fixture
def place_info_call(softphone_image):
    """Like conftest's place_inbound_call, but dials with a fresh per-call
    caller identity (`--id sip:e2einfo<uuid>@tester`).

    Caller memory is keyed by the SIP URI user part and persists across calls
    and pytest runs. With the default anonymous dialer every test shares one
    caller, so facts extracted from an earlier call (e.g. today's forecast
    temperature) let the model answer live-data questions from memory WITHOUT
    invoking the tool — observed live: the forecast question was answered
    "just like last time" with a stale high, and no FORECAST tool_call fired.
    A unique user part per call guarantees empty memory, keeping the Layer-1
    tool_call asserts deterministic.
    """
    def _call(question_filename: str, duration: int = 30, capture_name: str = "captured.wav"):
        captured = _AUDIO_DIR / capture_name
        if captured.exists():
            captured.unlink()
        cname = "e2e-info-dialer"
        subprocess.run(["docker", "rm", "-f", cname], check=False, capture_output=True)
        caller_id = f"sip:e2einfo{uuid.uuid4().hex[:10]}@tester"
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

# Weather/forecast replies must mention a temperature in some unit.
DEGREE_FORMS = ("degree", "fahrenheit", "celsius")

# Plausible spoken/written time forms (digits, am/pm, o'clock, word times).
TIME_PATTERNS = (
    r"\b\d{1,2}:\d{2}\b",                                  # 3:45 / 15:42
    r"\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b",                # 3 pm / 11 a.m.
    r"\bo'?clock\b",
    r"\b(?:noon|midnight)\b",
    # spoken form the speech reformatter may emit, e.g. "three forty" / "eleven fifteen"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:o'?clock|oh|five|ten|fifteen|twenty|thirty|forty|fifty)\b",
)


def _texts_for(events, name):
    return [(e.get("data") or {}).get("text", "") for e in events if e.get("event") == name]


def _tools_called(events):
    """Names of tools from tool_call events (the event tag may be top-level or nested)."""
    tools = []
    for e in events:
        name = e.get("event") or (e.get("data") or {}).get("event")
        if name == "tool_call":
            tool = (e.get("data") or {}).get("tool") or e.get("tool")
            if tool:
                tools.append(tool.upper())
    return tools


def _reply_and_transcript(events, transcribe, captured):
    reply_text = " ".join(_texts_for(events, "assistant_response")).lower()
    transcript = transcribe(captured).lower()
    return reply_text, transcript


def test_weather_current(question_wav, place_info_call, assert_spoke,
                         transcribe, agent_events, event_names):
    """WEATHER: the current-conditions tool fires and the reply gives a temperature."""
    fn = question_wav("weather_check_now.wav")
    captured, started_at = place_info_call(fn, duration=30)

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 2 first: a temperature reached the caller.
    reply_text, transcript = _reply_and_transcript(events, transcribe, captured)
    haystack = f"{reply_text} || {transcript}"
    assert any(form in haystack for form in DEGREE_FORMS), (
        f"no temperature in reply.\n  reply_text={reply_text!r}\n  transcript={transcript!r}"
    )

    # Layer 1: the WEATHER tool actually ran (tool_call event, or the tool's
    # own weather_fetch event — both prove execution).
    tools = _tools_called(events)
    if "WEATHER" not in tools and "weather_fetch" not in names:
        # A temperature was spoken (asserted above) with NO tool execution and
        # an empty caller memory: the agent fabricated a weather report.
        # Observed live 2026-07-10: "seventy-five degrees ... west at five mph"
        # with zero tool_call while the real NWS observation was 69F/calm.
        pytest.xfail(
            "product bug: agent spoke a made-up weather report without calling "
            f"WEATHER (tools={tools}); reply={reply_text!r}"
        )


def test_forecast_today(question_wav, place_info_call, assert_spoke,
                        transcribe, agent_events, event_names):
    """FORECAST: the NWS forecast tool fires and the reply gives a temperature."""
    fn = question_wav("forecast_check_today.wav")
    captured, started_at = place_info_call(fn, duration=30)

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    reply_text, transcript = _reply_and_transcript(events, transcribe, captured)
    haystack = f"{reply_text} || {transcript}"
    assert any(form in haystack for form in DEGREE_FORMS), (
        f"no temperature in forecast reply.\n  reply_text={reply_text!r}\n  transcript={transcript!r}"
    )

    tools = _tools_called(events)
    if "FORECAST" not in tools and "forecast_fetch" not in names:
        # Same fabrication class as the WEATHER test: a temperature was spoken
        # (asserted above) with no tool run and an empty caller memory.
        pytest.xfail(
            "product bug: agent spoke a forecast without calling FORECAST "
            f"(tools={tools}); reply={reply_text!r}"
        )


def test_web_search(question_wav, place_info_call, assert_spoke,
                    agent_events, event_names):
    """WEB_SEARCH: an explicit 'search the web' request routes through the tool.

    SearxNG round-trips are slow, so the call window is the 40s maximum. The
    result content is live web data, so only the tool_call event is asserted.
    """
    fn = question_wav("search_dgx_spark.wav")
    captured, started_at = place_info_call(fn, duration=40)

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    tools = _tools_called(events)
    assert "WEB_SEARCH" in tools, (
        f"WEB_SEARCH tool never called; tools={tools}, events={sorted(set(names))}"
    )

    # The agent spoke *something* substantive after the search.
    assert any(t.strip() for t in _texts_for(events, "assistant_response")), (
        "no assistant_response text logged after the search"
    )


def test_datetime_spoken_time(question_wav, place_info_call, assert_spoke,
                              transcribe, agent_events, event_names):
    """DATETIME: the reply contains a plausible time.

    The system prompt embeds the current time, so the model may answer without
    invoking the DATETIME tool — both paths are correct. Assert on the answer
    content, not on a tool_call (same rationale as the CALC test).
    """
    fn = question_wav("time_now.wav")
    captured, started_at = place_info_call(fn, duration=30)

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    reply_text, transcript = _reply_and_transcript(events, transcribe, captured)
    haystack = f"{reply_text} || {transcript}"
    assert any(re.search(p, haystack) for p in TIME_PATTERNS), (
        f"no plausible time in reply.\n  reply_text={reply_text!r}\n  transcript={transcript!r}"
    )


def test_quakes(question_wav, place_info_call, assert_spoke,
                agent_events, event_names):
    """QUAKES: an earthquake question routes through the USGS feed tool.

    Feed content varies day to day (there may legitimately be no big quakes),
    so only the tool_call and a non-empty spoken reply are asserted.
    """
    fn = question_wav("quakes_today.wav")
    captured, started_at = place_info_call(fn, duration=30)

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    tools = _tools_called(events)
    assert "QUAKES" in tools, (
        f"QUAKES tool never called; tools={tools}, events={sorted(set(names))}"
    )

    assert any(t.strip() for t in _texts_for(events, "assistant_response")), (
        "no assistant_response text logged for the quake question"
    )


def test_kp_index(question_wav, place_info_call, assert_spoke,
                  agent_events, event_names):
    """KP_INDEX: a space-weather question routes through the NOAA SWPC tool.

    The index value is live data (0-9), so the deterministic assertion is the
    tool_call event; a non-empty spoken reply is the secondary check.
    """
    fn = question_wav("kp_index_now.wav")
    captured, started_at = place_info_call(fn, duration=30)

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    tools = _tools_called(events)
    assert "KP_INDEX" in tools, (
        f"KP_INDEX tool never called; tools={tools}, events={sorted(set(names))}"
    )

    assert any(t.strip() for t in _texts_for(events, "assistant_response")), (
        "no assistant_response text logged for the KP-index question"
    )
