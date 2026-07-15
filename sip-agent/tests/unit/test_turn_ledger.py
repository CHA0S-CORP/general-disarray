"""Unit tests for the barge-in playback ledger reconstruction (spoken_text)."""
import pytest

from call_session import MIN_HEARD_FRACTION, TurnLedger, spoken_text

pytestmark = pytest.mark.unit


def _ledger(*chunks):
    """Build a TurnLedger with tags 1..N over the given chunk texts."""
    return TurnLedger(entries={i + 1: chunk for i, chunk in enumerate(chunks)})


@pytest.mark.parametrize(
    "chunks,completed,current,fraction,expected",
    [
        # All chunks completed: full text back, in enqueue order.
        (("One.", "Two.", "Three."), [1, 2, 3], None, 0.0, "One. Two. Three."),
        # Nothing completed, nothing playing: nothing heard.
        (("One.", "Two."), [], None, 0.0, ""),
        # Current chunk barely started (below threshold): dropped.
        (("One.", "Two."), [1], 2, MIN_HEARD_FRACTION - 0.05, "One."),
        # Current chunk mostly played (above threshold): included.
        (("One.", "Two."), [1], 2, MIN_HEARD_FRACTION + 0.05, "One. Two."),
        # Current chunk exactly at the threshold: included (>=).
        (("One.", "Two."), [1], 2, MIN_HEARD_FRACTION, "One. Two."),
        # Only the current chunk, fully played fraction.
        (("Solo.",), [], 1, 1.0, "Solo."),
        # Current tag not in the ledger (e.g. an untagged chime was playing).
        (("One.", "Two."), [1], 99, 0.9, "One."),
        # Completed tags the ledger doesn't know about are ignored.
        (("One.",), [7, 8], None, 0.0, ""),
        # Completed out of order still reconstructs in enqueue order.
        (("One.", "Two.", "Three."), [3, 1, 2], None, 0.0, "One. Two. Three."),
    ],
)
def test_spoken_text_table(chunks, completed, current, fraction, expected):
    assert spoken_text(_ledger(*chunks), completed, current, fraction) == expected


def test_empty_ledger_returns_empty():
    assert spoken_text(TurnLedger(), [1, 2, 3], 4, 1.0) == ""


def test_none_ledger_returns_empty():
    assert spoken_text(None, [1], 1, 1.0) == ""


def test_min_fraction_override():
    ledger = _ledger("One.", "Two.")
    # A stricter threshold drops a chunk the default would include.
    assert spoken_text(ledger, [1], 2, 0.5, min_fraction=0.9) == "One."
    assert spoken_text(ledger, [1], 2, 0.95, min_fraction=0.9) == "One. Two."


def test_current_none_never_matches_a_tag():
    # An idle player (current_tag None) must not accidentally match anything.
    ledger = _ledger("One.")
    assert spoken_text(ledger, [], None, 1.0) == ""
