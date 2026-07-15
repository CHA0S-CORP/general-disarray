"""Unit tests for expand_units_for_speech — measurement abbreviations spoken in
full so TTS says "ounces", not "oh zee". Strings are taken from real drink-tool
output observed on live calls."""
import pytest

from plugins.helpers import expand_units_for_speech

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("raw,expected", [
    # From live calls: "2 oz White Rum ... 1 oz Lime Juice ... 2 oz Prosecco"
    ("2 oz White Rum", "2 ounces White Rum"),
    ("1 oz Lime Juice", "1 ounce Lime Juice"),
    ("0.5 oz fresh lime juice", "0.5 ounces fresh lime juice"),
    ("0.25 oz mint syrup", "0.25 ounces mint syrup"),
    ("1.5 oz Spiced rum", "1.5 ounces Spiced rum"),
    # Proper fractions below one take the singular ("half an ounce").
    ("1/2 oz grenadine", "1/2 ounce grenadine"),
    ("3/4 oz lime", "3/4 ounce lime"),
    # Mixed numbers are more than one -> plural.
    ("1 1/2 oz gin", "1 1/2 ounces gin"),
    # Other units.
    ("2 tsp sugar", "2 teaspoons sugar"),
    ("1 tbsp honey", "1 tablespoon honey"),
    ("30 ml vodka", "30 milliliters vodka"),
    ("4 cl rum", "4 centiliters rum"),
    ("5 mph winds", "5 miles per hour winds"),
    # Singular agreement only when the quantity is exactly 1.
    ("1 tsp salt", "1 teaspoon salt"),
])
def test_expands_measures(raw, expected):
    assert expand_units_for_speech(raw) == expected


def test_full_recipe_line():
    raw = ("You need 2 oz White Rum, 1 oz Sugar Syrup, 1 oz Lime Juice, "
           "2 dashes Angostura Bitters, and 2 oz Prosecco.")
    out = expand_units_for_speech(raw)
    assert "2 ounces White Rum" in out
    assert "1 ounce Sugar Syrup" in out
    assert "2 ounces Prosecco" in out
    assert " oz" not in out
    # "dashes" is already a full word and must be left alone.
    assert "2 dashes Angostura" in out


@pytest.mark.parametrize("text", [
    "",
    "Shake with ice and strain.",           # no measures
    "The html page loaded.",                # 'ml' inside a word must not fire
    "He weighed the options.",              # no bare-letter false positives
    "Order 12 of them.",                    # number with no unit
])
def test_leaves_non_measures_untouched(text):
    assert expand_units_for_speech(text) == text


def test_bare_unit_without_quantity_is_left_alone():
    # No preceding number -> not a measurement, don't touch it.
    assert expand_units_for_speech("an oz of prevention") == "an oz of prevention"
