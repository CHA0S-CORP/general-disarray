"""Unit tests for the incremental sentence assembler (LLM→TTS streaming).

Two properties matter:
  1. Parity — feeding any text delta-by-delta yields the same chunks as
     split_into_sentences(text) (modulo the final partial flush).
  2. Marker hold — nothing that could be part of a [TOOL:...] marker is
     ever emitted, no matter how the deltas slice it.
"""
import pytest

from sentence_stream import SentenceAssembler, split_into_sentences

pytestmark = pytest.mark.unit


def _pieces(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


def _run(deltas):
    asm = SentenceAssembler()
    emitted = []
    for delta in deltas:
        emitted.extend(asm.feed(delta))
    return asm, emitted


def _stream_chunks(text, size):
    asm, emitted = _run(_pieces(text, size))
    tail = asm.flush()
    return emitted + ([tail] if tail else [])


PARITY_TEXTS = [
    "The weather today is sunny with a high of 75. Winds are light from the "
    "northwest. Have a great day!",
    "Yes. The timer is set for five minutes from now.",
    "Hi. Yo. Ok. This here is a properly long sentence now. And then another "
    "lengthy sentence follows it for good measure.",
    "One single sentence without any terminal punctuation at all",
    "Value is 3.5 which should not split. But this second sentence should "
    "split off on its own!",
    "Short. Tail",
    "Trailing separator ends this properly sized sentence here. ",
    "  Leading whitespace, then a long enough first sentence. Then a tiny "
    "tail. ",
    "Multi?  Spaced!   Sentences. With a longer closer sentence at the end.",
    "",
]


@pytest.mark.parametrize("size", [1, 2, 3, 7, 50, 1000])
@pytest.mark.parametrize("text", PARITY_TEXTS)
def test_delta_fed_parity_with_batch_splitter(text, size):
    assert _stream_chunks(text, size) == split_into_sentences(text)


def test_whitespace_normalized_concat_matches_source():
    text = ("First sentence has plenty of length to emit alone. Second one\n"
            "spans a newline separator.  Third has  odd   spacing inside.")
    for size in (1, 4, 9):
        chunks = _stream_chunks(text, size)
        assert " ".join(" ".join(chunks).split()) == " ".join(text.split())


# --- marker hold ---------------------------------------------------------


def test_marker_held_across_delta_boundary():
    asm, emitted = _run(["[TO", "OL:X:a=1] tail"])
    assert emitted == []            # nothing marker-ish ever emitted
    assert asm.marker_seen
    assert asm.flush() == "[TOOL:X:a=1] tail"


def test_false_prefix_released():
    text = ("[Total nonsense opens this sentence which is long enough. "
            "And a second long-enough sentence follows right after it.")
    for size in (1, 3, 8):
        assert _stream_chunks(text, size) == split_into_sentences(text)
    asm, emitted = _run(_pieces(text, 3))
    assert not asm.marker_seen
    assert any("[Total" in c for c in emitted)


def test_marker_at_position_zero():
    asm, emitted = _run(_pieces("[TOOL:HANGUP] Goodbye then.", 2))
    assert emitted == []
    assert asm.marker_seen
    assert asm.flush() == "[TOOL:HANGUP] Goodbye then."


def test_marker_mid_sentence_holds_the_whole_sentence():
    text = "The answer you wanted is [TOOL:CALC:expression=2+2] obviously."
    asm, emitted = _run(_pieces(text, 5))
    assert emitted == []            # sentence never completed before the marker
    assert asm.marker_seen
    assert asm.flush() == text


def test_sentences_before_marker_still_emit():
    text = ("This first sentence is long enough to be emitted on its own. "
            "[TOOL:JOKE:category=dad]")
    asm, emitted = _run(_pieces(text, 3))
    assert emitted == ["This first sentence is long enough to be emitted on its own."]
    assert asm.marker_seen
    assert asm.flush() == "[TOOL:JOKE:category=dad]"
    assert all("[" not in c for c in emitted)


def test_stream_ending_inside_a_hold():
    asm, emitted = _run(["Some quite long sentence definitely ends here. [TOO"])
    assert emitted == ["Some quite long sentence definitely ends here."]
    assert not asm.marker_seen      # never proven to be a marker
    assert asm.flush() == "[TOO"


def test_after_marker_everything_accumulates_silently():
    asm, emitted = _run(_pieces(
        "Okay. [TOOL:SIMON_SAYS:text=hi] And this trailing sentence is long "
        "enough that it would normally emit.", 4))
    assert emitted == []            # "Okay." was too short to emit pre-marker
    assert asm.marker_seen
    tail = asm.flush()
    assert tail.startswith("Okay. [TOOL:SIMON_SAYS:text=hi]")
    assert tail.endswith("normally emit.")


def test_lowercase_marker_is_not_held():
    text = ("This sentence mentions [tool:fake] in lowercase and is long. "
            "Second sentence is long enough to be its own chunk too.")
    assert _stream_chunks(text, 6) == split_into_sentences(text)
    asm, _ = _run(_pieces(text, 6))
    assert not asm.marker_seen
