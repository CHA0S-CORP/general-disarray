"""Unit tests for OutboundCallHandler._match_choice (whole-word choice matching).

The matcher must avoid substring false-positives ("yesterday" != "yes",
"I don't know" != "no") while still matching synonyms and multi-word phrases.
"""
import pytest

from api import OutboundCallHandler, ChoiceOption

pytestmark = pytest.mark.unit


@pytest.fixture
def matcher():
    # _match_choice uses no instance state, so skip __init__ entirely.
    return OutboundCallHandler.__new__(OutboundCallHandler)


@pytest.fixture
def yes_no_options():
    return [
        ChoiceOption(value="yes", synonyms=["yeah", "yep", "sure thing"]),
        ChoiceOption(value="no", synonyms=["nope", "nah"]),
    ]


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("yes", "yes"),
        ("yes please", "yes"),
        ("yeah", "yes"),
        ("sure thing, go ahead", "yes"),  # multi-word synonym
        ("no", "no"),
        ("nope not today", "no"),
    ],
)
def test_matches(matcher, yes_no_options, spoken, expected):
    assert matcher._match_choice(spoken, yes_no_options) == expected


@pytest.mark.parametrize(
    "spoken",
    [
        "yesterday i was busy",  # 'yes' is a substring, not a word
        "I don't know",          # 'no' is a substring of 'know'
        "maybe later",           # nothing matches
        "",
    ],
)
def test_no_false_positive(matcher, yes_no_options, spoken):
    assert matcher._match_choice(spoken, yes_no_options) is None
