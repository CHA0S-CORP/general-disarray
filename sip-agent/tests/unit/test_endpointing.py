"""Unit tests for the pure endpointing heuristics (endpointing.py).

Table-driven: looks_complete() (terminal punctuation, short answers,
conjunction/filler/comma endings, digit recitation) and suggest_timeout_ms()
(clamping, no-text speech-length scaling, text overrides).
"""
import pytest

from endpointing import looks_complete, suggest_timeout_ms

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# looks_complete
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    # Terminal punctuation.
    "What's the weather like today?",
    "Set a timer for five minutes.",
    "That's great!",
    "My number is five five five one two three.",
    # Known complete short answers, no punctuation needed.
    "yes",
    "Yes",
    "no",
    "yeah",
    "nope",
    "correct",
    "okay",
    "sure",
    "right",
    "that's all",
    "That's it",
    "thank you",
    "goodbye",
    # Bare number / digit-string answers.
    "42",
    "555 1234",
    "five five five",
    "seven",
    # Ordinary unpunctuated sentences: small Whisper models on telephone
    # audio routinely omit terminal punctuation, and these must still read
    # as complete or speculative mode would hold every such utterance to
    # the max-silence deadline (slower than fixed mode).
    "turn off the kitchen lights",
    "what's the weather in denver",
    "set a timer for ten minutes",
    "i want a large pizza",
])
def test_looks_complete_true(text):
    assert looks_complete(text) is True


@pytest.mark.parametrize("text", [
    # Empty / whitespace.
    "",
    "   ",
    # Conjunction endings.
    "I want the weather and",
    "we could do that or",
    "I called because",
    "it was raining so",
    "I like it but",
    # Filler endings (even Whisper-punctuated).
    "um",
    "Um.",
    "uh",
    "hmm...",
    "let me think er",
    # Dangling prepositions / articles / possessives.
    "send it to",
    "the capital of",
    "I left it in",
    "call the",
    "give me a",
    "what about my",
    "set a timer for",
    "book a table at",
    "and the rest is",
    # Trailing comma.
    "first the weather,",
    "sure,",
    # Trailing ellipsis is stripped, leaving an unfinished thought.
    "I was thinking about...",
    "My number is five five five and…",
    # Digit group mid-recitation: last group shorter than preceding ones.
    "555 12",
    "my number is 555 12",
    "call 800 555 12",
    # Spelled-out digits trailing a longer sentence: mid-recitation too.
    "my number is five five five",
    "the code is four seven",
    # Dangling copula.
    "my number is five five five oh one and the rest is",
    # Very short non-listed fragments stay on the safe side.
    "the weather",
    "pizza",
])
def test_looks_complete_false(text):
    assert looks_complete(text) is False


def test_digit_groups_complete_when_last_group_not_shorter():
    # 4-digit final group after a 3-digit group: a finished phone number.
    assert looks_complete("555 1234") is True
    # Single group is just a bare number answer.
    assert looks_complete("8675309") is True


# ---------------------------------------------------------------------------
# suggest_timeout_ms
# ---------------------------------------------------------------------------

BASE, MIN, MAX = 750, 350, 1500


def test_no_text_short_speech_leans_to_max():
    # Under ~1s of speech: caller may just be winding up -> max hangover.
    assert suggest_timeout_ms(None, 400, BASE, MIN, MAX) == MAX
    assert suggest_timeout_ms("", 1000, BASE, MIN, MAX) == MAX


def test_no_text_long_speech_leans_to_base():
    assert suggest_timeout_ms(None, 4000, BASE, MIN, MAX) == BASE
    assert suggest_timeout_ms(None, 30000, BASE, MIN, MAX) == BASE


def test_no_text_mid_speech_interpolates_monotonically():
    prev = MAX
    for speech_ms in (1000, 1500, 2000, 2500, 3000, 3500, 4000):
        t = suggest_timeout_ms(None, speech_ms, BASE, MIN, MAX)
        assert MIN <= t <= MAX
        assert t <= prev, "timeout must shrink as the utterance grows"
        prev = t
    assert suggest_timeout_ms(None, 2500, BASE, MIN, MAX) < MAX
    assert suggest_timeout_ms(None, 2500, BASE, MIN, MAX) > BASE


def test_complete_text_clamps_to_min():
    assert suggest_timeout_ms("yes", 200, BASE, MIN, MAX) == MIN
    assert suggest_timeout_ms("Set a timer for five minutes.", 5000,
                              BASE, MIN, MAX) == MIN


def test_incomplete_text_clamps_to_max():
    assert suggest_timeout_ms("my number is 555 12", 5000, BASE, MIN, MAX) == MAX
    assert suggest_timeout_ms("I want the weather and", 200, BASE, MIN, MAX) == MAX


def test_result_always_within_bounds():
    # Base below min still clamps into [min, max].
    assert suggest_timeout_ms(None, 10000, 100, MIN, MAX) == MIN
    # Base above max clamps down.
    assert suggest_timeout_ms(None, 10000, 9999, MIN, MAX) == MAX
    # Degenerate max < min: min wins.
    assert suggest_timeout_ms("yes", 0, BASE, 500, 400) == 500
