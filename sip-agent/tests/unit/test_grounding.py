"""Unit tests for the grounding detectors (never-guess enforcement)."""
import pytest

from grounding import (CATEGORY_TOOLS, grounding_category, live_data_category,
                       promised_action, trailing_promise)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("text,category", [
    ("What's the weather in Toledo?", "WEATHER"),
    ("is it going to rain today", "WEATHER"),
    ("Check the forecast for tomorrow.", "WEATHER"),
    ("What time is it?", "DATETIME"),
    ("tell me the time", "DATETIME"),
    ("what day is it today", "DATETIME"),
    ("Any earthquakes today?", "QUAKES"),
    ("what was the most recent quake", "QUAKES"),
    ("check the gpu", "GPU_STATUS"),
    ("How's the graphics card doing?", "GPU_STATUS"),
    ("are there any alerts firing", "ALERTS"),
    ("will the aurora be visible tonight", "KP_INDEX"),
    ("search for the game score", "WEB_SEARCH"),
    ("look up the latest news about the launch", "WEB_SEARCH"),
])
def test_live_data_categories(text, category):
    assert live_data_category(text) == category


@pytest.mark.parametrize("text", [
    "",
    "hello there",
    "tell me a joke",
    "nice weather we're having",           # topic without intent cue
    "that earthquake movie was great",     # topic without intent cue
    "thanks, that was helpful",
    "set a timer for five minutes",        # action, not live data
    "goodbye",
])
def test_casual_chat_never_triggers(text):
    assert live_data_category(text) is None


def test_priority_specific_beats_generic():
    # "weather alerts" must resolve to ALERTS, not WEATHER.
    assert live_data_category("any weather alerts right now?") == "ALERTS"
    # quake question mentioning searching still classifies as QUAKES.
    assert live_data_category("check for earthquakes near me") == "QUAKES"


@pytest.mark.parametrize("reply", [
    "Let me check the USGS data again. One moment.",
    "I'll look that up for you.",
    "Hang on, checking now.",
    "Just a second while I pull that up.",
    "Give me a moment to find that.",
    # Heard on a real call, followed by seventeen seconds of dead air. The
    # verb list stopped at check/look/pull/find/see/verify, so "get" walked
    # straight through.
    "I see you're looking for a potato casserole recipe. "
    "Let me get that for you right away.",
    "I'll go grab that for you.",
    "Let me search for that.",
])
def test_promised_action_true(reply):
    assert promised_action(reply) is True


@pytest.mark.parametrize("reply", [
    "",
    "Here's the forecast: sunny with a high of 75.",
    "It's three thirty PM.",
    "I checked — no alerts are firing.",
    "The largest quake was magnitude four point six.",
    # "get" alone must not make every idiom a promise.
    "Let me get this straight — you want the Tuesday slot?",
])
def test_promised_action_false(reply):
    assert promised_action(reply) is False


@pytest.mark.parametrize("reply", [
    "I see you're looking for a recipe. Let me get that for you right away.",
    "Sure. One moment.",
    "I'll look that up for you.",
])
def test_trailing_promise_true(reply):
    """The reply signs off on a promise: nothing follows, so nothing was
    delivered. This is what gets a tool-calling turn retried."""
    assert trailing_promise(reply) is True


@pytest.mark.parametrize("reply", [
    "",
    # A promise that was KEPT in the same breath must not force a retry — the
    # answer is right there in the final sentence.
    "Let me check that for you. It's seventy-one degrees in Snohomish.",
    "One moment... okay, the largest quake was magnitude four point six.",
    "Here's the forecast: sunny with a high of 75.",
])
def test_trailing_promise_false(reply):
    assert trailing_promise(reply) is False


def test_grounding_category_combines_both():
    assert grounding_category("what's the weather?", "anything") == "WEATHER"
    assert grounding_category("tell me something nice",
                              "Let me check on that. One moment.") == "PROMISED_ACTION"
    assert grounding_category("tell me something nice", "Sure thing.") is None


def test_category_tools_map_covers_all_categories():
    for category in ("ALERTS", "QUAKES", "KP_INDEX", "WEATHER", "DATETIME",
                     "GPU_STATUS", "WEB_SEARCH", "PROMISED_ACTION"):
        assert category in CATEGORY_TOOLS
