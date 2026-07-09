"""Unit tests for the sentence splitter feeding streaming TTS."""
import pytest

from main import split_into_sentences

pytestmark = pytest.mark.unit


def test_splits_on_sentence_boundaries():
    text = "The weather today is sunny with a high of 75. Winds are light from the northwest. Have a great day!"
    assert split_into_sentences(text) == [
        "The weather today is sunny with a high of 75.",
        "Winds are light from the northwest.",
        "Have a great day!",
    ]


def test_short_fragments_merge_forward():
    # "Yes." alone is too short to be worth a TTS round-trip.
    text = "Yes. The timer is set for five minutes from now."
    assert split_into_sentences(text) == [
        "Yes. The timer is set for five minutes from now.",
    ]


def test_decimal_numbers_do_not_split():
    text = "The result of 7 divided by 2 is 3.5 which rounds up to 4."
    assert split_into_sentences(text) == [text]


def test_single_sentence_passthrough():
    assert split_into_sentences("Hello there") == ["Hello there"]


def test_empty_text():
    assert split_into_sentences("   ") == []
