"""Unit tests for _SpeechRun, the barge-in / cancel-merge debounce.

The gate must mean "the caller has been talking for min_ms" — and mean it at
ANY chunk size. The audio loop's chunks are variable-length (sip_handler hands
it anything from one 20ms frame up to 100ms), and real speech is not
VAD-positive end to end: gaps between syllables and unvoiced consonants read as
silence. A run that resets on the first negative chunk therefore needs the
caller to talk for several times min_ms before it trips — and the finer the
chunks, the worse it gets. That was the barge-in bug.
"""
import pytest

from main import _SpeechRun

pytestmark = pytest.mark.unit

MIN_MS = 400
MAX_GAP_MS = 250


def feed(run: _SpeechRun, chunk_ms: float, pattern: str):
    """Feed a pattern of chunks ('s' = speech, '.' = silence).

    Returns the elapsed ms at which the gate first tripped, or None.
    """
    elapsed = 0.0
    for c in pattern:
        elapsed += chunk_ms
        if run.update(chunk_ms, c == "s"):
            return elapsed
    return None


def test_unbroken_speech_trips_at_the_threshold():
    run = _SpeechRun(MIN_MS, MAX_GAP_MS)
    assert feed(run, 20, "s" * 30) == MIN_MS


@pytest.mark.parametrize("chunk_ms", [20, 40, 100])
def test_gate_is_chunk_size_independent(chunk_ms):
    """400ms of speech is 400ms of speech, whether it arrives as 4 chunks or
    20. Before the gap tolerance, 20ms chunks needed ~4x longer to trip."""
    run = _SpeechRun(MIN_MS, MAX_GAP_MS)
    n = int(1000 / chunk_ms)  # one second of speech
    assert feed(run, chunk_ms, "s" * n) == pytest.approx(MIN_MS, abs=chunk_ms)


def test_short_gaps_inside_speech_do_not_reset_the_run():
    """Inter-syllable gaps are part of speech, not the end of it."""
    run = _SpeechRun(MIN_MS, MAX_GAP_MS)
    # 100ms speech, 60ms gap, repeat — a normal cadence. Total speech reaches
    # 400ms during the fourth burst.
    assert feed(run, 20, "sssss...sssss...sssss...sssss") is not None


def test_a_long_gap_abandons_the_run():
    """A gap past max_gap_ms means the caller stopped: the next burst starts a
    fresh run rather than topping up a stale one."""
    run = _SpeechRun(MIN_MS, MAX_GAP_MS)
    # 300ms of speech, then 300ms of silence (> MAX_GAP_MS) — run abandoned.
    assert feed(run, 20, "s" * 15 + "." * 15) is None
    assert run.speech_ms == 0.0
    # A following 300ms burst must NOT trip the gate by adding to the old 300ms.
    assert feed(run, 20, "s" * 15) is None


def test_isolated_noise_never_trips_the_gate():
    """A click/pop is a couple of chunks; it must not cancel an in-flight turn."""
    run = _SpeechRun(MIN_MS, MAX_GAP_MS)
    assert feed(run, 20, ("ss" + "." * 20) * 10) is None


def test_silence_before_any_speech_costs_nothing():
    run = _SpeechRun(MIN_MS, MAX_GAP_MS)
    assert feed(run, 20, "." * 50) is None
    assert run.gap_ms == 0.0
    # The gate still trips normally afterwards.
    assert feed(run, 20, "s" * 20) == MIN_MS


def test_reset_clears_both_counters():
    run = _SpeechRun(MIN_MS, MAX_GAP_MS)
    feed(run, 20, "sssss.")
    run.reset()
    assert run.speech_ms == 0.0
    assert run.gap_ms == 0.0
