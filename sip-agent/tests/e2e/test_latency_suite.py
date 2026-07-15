"""E2E LATENCY SUITE: measure per-turn latency across representative real calls.

Places one real SIP call per row of CALLS (a tool turn, a plain chat turn, and a
WEATHER-tool turn), then a final summary test writes the collected numbers to
reports/latency.json + a human-readable reports/latency.md table.

Metrics are mined from the agent's structured JSON log (Layer 1):
  - e2e turn latency  = ts(assistant_response) - ts(user_speech), per turn
  - STT ms            from "STT (batch): <N>ms for <M>ms audio" msg lines
  - TTS ms            from "TTS: <N>ms for <K> chars" msg lines (first = first synth)
  - LLM turn          from agent_turn events (data.latency_ms, data.tool_rounds)
  - call setup        from "Call connected in <N>ms"

Assertions are deliberately GENEROUS budgets (order-of-magnitude guards) so the
suite stays stable across model/load variance; the reports carry the real data.
To add another call to the suite: register its question text in FIXTURES below
(if new) and append a row to CALLS — nothing else.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Register this module's question fixtures without editing gen_audio.py
# (parallel test authors each register their own texts at import time).
sys.path.insert(0, str(Path(__file__).parent / "audio"))
import gen_audio  # noqa: E402

gen_audio.FIXTURES.update({
    "latency_chat_how_are_you.wav": "How are you today?",
    "latency_weather_now.wav": "What is the weather right now?",
})

pytestmark = pytest.mark.e2e

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# --- generous budgets (ms): stability first, regressions of magnitude only ---
BUDGET_E2E_TURN_MS = 20_000   # user stops speaking -> assistant reply logged
BUDGET_STT_MS = 10_000        # single batch STT round-trip (spikes to ~4.5s
                              # observed under shared-GPU load; keep magnitude
                              # headroom — real numbers land in the reports)
BUDGET_TTS_FIRST_MS = 5_000   # first TTS synthesis of the reply

# --- log-line parsers --------------------------------------------------------
TS_FMT = "%Y-%m-%d %H:%M:%S,%f"  # JSONFormatter "ts", e.g. "2026-07-10 15:16:18,613"
STT_RE = re.compile(r"STT \(batch\): (\d+(?:\.\d+)?)ms for (\d+(?:\.\d+)?)ms audio")
TTS_RE = re.compile(r"TTS: (\d+(?:\.\d+)?)ms for (\d+) chars")
CONNECT_RE = re.compile(r"Call connected in (\d+(?:\.\d+)?)ms")

# One row per call: (id, wav fixture filename, call duration seconds).
# Reuses the existing calc fixture; the other two are registered above.
CALLS = [
    ("calc-tool-turn", "calc_17x3.wav", 30),
    ("chat-no-tool", "latency_chat_how_are_you.wav", 30),
    ("weather-tool", "latency_weather_now.wav", 35),
]

# Module-level accumulator: each call test appends its metrics dict here and
# the summary test (defined last, so it runs last) writes the reports.
COLLECTED = []


def _parse_ts(event):
    try:
        return datetime.strptime(event["ts"], TS_FMT)
    except (KeyError, ValueError):
        return None


def _extract_metrics(events):
    """Mine one call's latency metrics from its structured log window."""
    metrics = {
        "connect_ms": None,     # SIP answer -> media up
        "stt_ms": [],           # per STT round-trip
        "stt_audio_ms": [],     # utterance length fed to STT
        "tts_ms": [],           # per TTS synthesis (index 0 = first synth)
        "tts_chars": [],
        "agent_turns": [],      # {"latency_ms": ..., "tool_rounds": ...} per turn
        "e2e_turn_ms": [],      # ts(assistant_response) - ts(user_speech) per turn
        "ttfa_ms": [],          # turn start -> first reply audio enqueued, per turn
        "tools_called": [],
    }
    pending_user_ts = None
    for e in events:
        name = e.get("event") or (e.get("data") or {}).get("event")
        data = e.get("data") or {}
        msg = e.get("msg") or ""

        # (a) end-to-end turn latency: pair each user_speech with the next
        # assistant_response (acks/tool events in between are ignored).
        # NOTE: assistant_response now fires after playback DRAINS (truthful
        # history), so e2e_turn_ms includes reply playback time. The perceived
        # responsiveness number is ttfa_ms (turn start -> first audio enqueued),
        # carried on the assistant_response event itself.
        if name == "user_speech":
            pending_user_ts = _parse_ts(e)
        elif name == "assistant_response":
            if pending_user_ts is not None:
                ts = _parse_ts(e)
                if ts is not None:
                    metrics["e2e_turn_ms"].append((ts - pending_user_ts).total_seconds() * 1000.0)
                pending_user_ts = None
            if data.get("time_to_first_audio_ms") is not None:
                metrics["ttfa_ms"].append(float(data["time_to_first_audio_ms"]))

        # (d) LLM engine turn stats
        if name == "agent_turn":
            metrics["agent_turns"].append({
                "latency_ms": data.get("latency_ms"),
                "tool_rounds": data.get("tool_rounds"),
            })
        elif name == "tool_call" and data.get("tool"):
            metrics["tools_called"].append(data["tool"])

        # (b)/(c)/(e) latency msg lines
        m = STT_RE.search(msg)
        if m:
            metrics["stt_ms"].append(float(m.group(1)))
            metrics["stt_audio_ms"].append(float(m.group(2)))
        m = TTS_RE.search(msg)
        if m:
            metrics["tts_ms"].append(float(m.group(1)))
            metrics["tts_chars"].append(int(m.group(2)))
        m = CONNECT_RE.search(msg)
        if m and metrics["connect_ms"] is None:
            metrics["connect_ms"] = float(m.group(1))
    return metrics


@pytest.mark.parametrize("call_id,wav,duration", CALLS, ids=[c[0] for c in CALLS])
def test_latency_call(call_id, wav, duration, question_wav, place_inbound_call,
                      assert_spoke, agent_events, event_names):
    fn = question_wav(wav)
    captured, started_at = place_inbound_call(fn, duration=duration,
                                              capture_name=f"latency_{call_id}.wav")

    # Layer 0: the agent produced real, non-silent audio.
    assert_spoke(captured)

    # Layer 1: pipeline stages fired, and yield the latency numbers.
    events = agent_events(started_at)
    names = event_names(events)
    for required in ("user_speech", "assistant_response"):
        assert required in names, f"missing event '{required}'; saw {sorted(set(names))}"

    metrics = _extract_metrics(events)
    metrics["call"] = call_id
    metrics["question"] = gen_audio.FIXTURES[wav]
    COLLECTED.append(metrics)

    # Generous budgets only — the summary reports carry the real numbers.
    assert metrics["e2e_turn_ms"], "no user_speech -> assistant_response pair found"
    worst_turn = max(metrics["e2e_turn_ms"])
    assert worst_turn < BUDGET_E2E_TURN_MS, (
        f"e2e turn latency {worst_turn:.0f}ms exceeds {BUDGET_E2E_TURN_MS}ms budget"
    )
    if metrics["stt_ms"]:  # only logged in batch STT mode
        worst_stt = max(metrics["stt_ms"])
        assert worst_stt < BUDGET_STT_MS, (
            f"STT latency {worst_stt:.0f}ms exceeds {BUDGET_STT_MS}ms budget"
        )
    if metrics["tts_ms"]:  # greeting is pre-cached, so these are reply synths
        first_tts = metrics["tts_ms"][0]
        assert first_tts < BUDGET_TTS_FIRST_MS, (
            f"first TTS synth {first_tts:.0f}ms exceeds {BUDGET_TTS_FIRST_MS}ms budget"
        )


def _fmt(value):
    if value is None:
        return "-"
    return f"{value:.0f}"


def test_latency_summary_report():
    """Runs last (definition order): write reports/latency.{json,md}."""
    assert COLLECTED, "no latency metrics collected — did the call tests run first?"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "budgets_ms": {
            "e2e_turn": BUDGET_E2E_TURN_MS,
            "stt": BUDGET_STT_MS,
            "tts_first_synth": BUDGET_TTS_FIRST_MS,
        },
        "calls": COLLECTED,
    }
    (REPORTS_DIR / "latency.json").write_text(json.dumps(payload, indent=2))

    # Human-readable table: first-turn numbers per call (the question turn).
    lines = [
        "# E2E latency report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "| call | question | stt_ms | llm_ms | tts_ms | ttfa_ms | e2e_turn_ms | tool_rounds | tools | connect_ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in COLLECTED:
        first_turn = m["agent_turns"][0] if m["agent_turns"] else {}
        lines.append("| {call} | {q} | {stt} | {llm} | {tts} | {ttfa} | {e2e} | {rounds} | {tools} | {conn} |".format(
            call=m["call"],
            q=m["question"],
            stt=_fmt(m["stt_ms"][0] if m["stt_ms"] else None),
            llm=_fmt(first_turn.get("latency_ms")),
            tts=_fmt(m["tts_ms"][0] if m["tts_ms"] else None),
            ttfa=_fmt(m["ttfa_ms"][0] if m["ttfa_ms"] else None),
            e2e=_fmt(m["e2e_turn_ms"][0] if m["e2e_turn_ms"] else None),
            rounds=first_turn.get("tool_rounds", "-"),
            tools=",".join(m["tools_called"]) or "-",
            conn=_fmt(m["connect_ms"]),
        ))
    lines.append("")
    (REPORTS_DIR / "latency.md").write_text("\n".join(lines))
