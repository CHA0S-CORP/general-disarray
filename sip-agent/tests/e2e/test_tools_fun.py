"""E2E: fun tools (DICE, COIN, JOKE, TRIVIA) over real inbound calls.

Each test places one real SIP call, speaks a request, and asserts via the
standard layers: non-silent audio (Layer 0), structured log events proving the
tool fired (Layer 1), and — where the tool output is deterministic enough —
content in the logged reply and/or the transcribed capture (Layer 2).

DICE/COIN/JOKE are speak_result tools, so their ToolResult message is folded
verbatim into the spoken reply: number words for a d6 roll, heads/tails for a
coin, and the literal joke text from JokeTool.JOKES. Exact LLM phrasing is
never asserted.

Stability note: pjsua's --auto-play loops the question WAV for the whole call,
and the agent only routes a question to a tool the first time it hears it — on
repeats it answers from conversation history (improvising "results" without
the tool). Worse, an utterance still in flight at hangup gets transcribed
after call_end and leaks into the NEXT call's first turn. Both are avoided by
padding each question WAV with trailing silence past the call duration, so the
question is spoken exactly once and the hangup lands in silence.
"""
import pathlib
import sys
import wave

import pytest

# Register this file's question fixtures without editing gen_audio.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent / "audio"))
import gen_audio  # noqa: E402

gen_audio.FIXTURES.update({
    "fun_dice_d6.wav": "Roll a six sided dice.",
    "fun_coin_flip.wav": "Flip a coin.",
    # JOKE/TRIVIA name the tool explicitly: with natural phrasing the model
    # sometimes improvises a joke/question itself instead of calling the tool.
    "fun_joke_tech.wav": "Call your joke tool with category tech, and tell me the joke it returns.",
    "fun_trivia_tool.wav": "Please use your trivia tool to ask me a trivia question.",
})

pytestmark = pytest.mark.e2e

_AUDIO_DIR = pathlib.Path(__file__).parent / "audio"
CALL_DURATION = 30
# Question WAVs are padded to at least this long so the looping auto-play
# never gets a second playthrough in before the --duration hangup.
_MIN_WAV_S = CALL_DURATION + 5

# DICE spells results as words ("I rolled a three."); a d6 lands in one..six.
D6_WORDS = ("one", "two", "three", "four", "five", "six")

# Distinctive lowercase fragments from JokeTool.JOKES (src/plugins/joke_tool.py),
# spanning all three categories since the LLM may pass any `category` param.
JOKE_FRAGMENTS = (
    # general
    "trust atoms", "eyebrows", "scarecrow", "anti-gravity",
    "crack each other up", "fish without eyes", "grew on me",
    # tech
    "attracts bugs", "understand binary", "sql query", "his cache",
    "foo bar", "c sharp",
    # dad
    "days are numbered", "impasta", "it just waved", "got mugged",
    "gummy bear", "seafood diet",
)


def _pad_to_single_play(path: pathlib.Path, min_seconds: float = _MIN_WAV_S):
    """Append trailing silence so the WAV outlasts the call (idempotent)."""
    with wave.open(str(path), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    need = int(min_seconds * params.framerate) - params.nframes
    if need <= 0:
        return
    with wave.open(str(path), "wb") as w:
        w.setparams(params)
        w.writeframes(frames + b"\x00" * (need * params.sampwidth * params.nchannels))


@pytest.fixture
def single_play_wav(question_wav):
    """question_wav + trailing-silence padding: one utterance per call."""
    def _make(filename: str) -> str:
        fn = question_wav(filename)
        _pad_to_single_play(_AUDIO_DIR / fn)
        return fn
    return _make


def _texts_for(events, name):
    return [(e.get("data") or {}).get("text", "") for e in events if e.get("event") == name]


def _tools_called(events):
    return [(e.get("data") or {}).get("tool", "") for e in events if e.get("event") == "tool_call"]


def test_dice_over_call(single_play_wav, place_inbound_call, assert_spoke,
                        transcribe, agent_events, event_names):
    fn = single_play_wav("fun_dice_d6.wav")
    captured, started_at = place_inbound_call(fn, duration=CALL_DURATION)

    # Layer 0: the agent produced real, non-silent audio.
    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 1: the model can't roll dice itself — the DICE tool must fire.
    tools = _tools_called(events)
    assert "DICE" in tools, f"DICE tool never fired; tool_calls={tools}"

    # Layer 2: a d6 result word (speak_result output) reaches the caller.
    reply_text = " ".join(_texts_for(events, "assistant_response")).lower()
    transcript = transcribe(captured).lower()
    haystack = f"{reply_text} || {transcript}"
    assert any(word in haystack for word in D6_WORDS), (
        f"no d6 number word found.\n  reply_text={reply_text!r}\n  transcript={transcript!r}"
    )


def test_coin_over_call(single_play_wav, place_inbound_call, assert_spoke,
                        transcribe, agent_events, event_names):
    fn = single_play_wav("fun_coin_flip.wav")
    captured, started_at = place_inbound_call(fn, duration=CALL_DURATION)

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 1: a genuine flip requires the COIN tool.
    tools = _tools_called(events)
    assert "COIN" in tools, f"COIN tool never fired; tool_calls={tools}"

    # Layer 2: the outcome ("Heads."/"Tails.") is spoken back.
    reply_text = " ".join(_texts_for(events, "assistant_response")).lower()
    transcript = transcribe(captured).lower()
    haystack = f"{reply_text} || {transcript}"
    assert "heads" in haystack or "tails" in haystack, (
        f"no heads/tails found.\n  reply_text={reply_text!r}\n  transcript={transcript!r}"
    )


def test_joke_over_call(single_play_wav, place_inbound_call, assert_spoke,
                        transcribe, agent_events, event_names):
    fn = single_play_wav("fun_joke_tech.wav")
    captured, started_at = place_inbound_call(fn, duration=CALL_DURATION)

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 1: the JOKE tool must fire (the joke DB lives in the tool, and
    # speak_result guarantees its message is spoken verbatim).
    tools = _tools_called(events)
    assert "JOKE" in tools, f"JOKE tool never fired; tool_calls={tools}"

    # The reply must carry actual joke text, not just a bare acknowledgment.
    reply_text = " ".join(_texts_for(events, "assistant_response")).lower()
    assert len(reply_text.strip()) >= 40, (
        f"assistant_response too short to contain a joke: {reply_text!r}"
    )

    # Layer 2: a known fragment from JokeTool.JOKES appears in the reply
    # and/or the transcribed capture (STT on the recording can be noisy).
    transcript = transcribe(captured).lower()
    haystack = f"{reply_text} || {transcript}"
    assert any(frag in haystack for frag in JOKE_FRAGMENTS), (
        f"no known joke fragment found.\n  reply_text={reply_text!r}\n  transcript={transcript!r}"
    )


def test_trivia_over_call(single_play_wav, place_inbound_call, assert_spoke,
                          agent_events, event_names):
    fn = single_play_wav("fun_trivia_tool.wav")
    captured, started_at = place_inbound_call(fn, duration=CALL_DURATION)

    assert_spoke(captured)

    events = agent_events(started_at)
    names = event_names(events)
    assert "user_speech" in names, f"STT never fired; saw {sorted(set(names))}"

    # Layer 1: starting a game must route through the TRIVIA tool (question
    # content is random, so no Layer-2 content assert here).
    tools = _tools_called(events)
    assert "TRIVIA" in tools, f"TRIVIA tool never fired; tool_calls={tools}"

    # The agent said something substantive back (the trivia question).
    assert any(t.strip() for t in _texts_for(events, "assistant_response")), (
        "no assistant_response logged for trivia"
    )
