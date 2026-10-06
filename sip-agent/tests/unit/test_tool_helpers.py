"""Unit tests for the shared plugin helpers (plugins/helpers.py)."""
from types import SimpleNamespace

import pytest

from plugins.helpers import (fetch_json, home_coordinates, number_to_words,
                             spoken_time_ago)

pytestmark = pytest.mark.unit

NOW_S = 1_800_000_000.0


@pytest.mark.parametrize("n,expected", [
    (0, "zero"), (3, "three"), (13, "thirteen"), (20, "twenty"),
    (42, "forty two"), (99, "ninety nine"), (100, "100"), (1234, "1234"),
])
def test_number_to_words(n, expected):
    assert number_to_words(n) == expected


@pytest.mark.parametrize("age_s,expected", [
    (30, "just now"),
    (600, "about ten minutes ago"),
    (2 * 3600, "about two hours ago"),
    (3 * 86400, "about three days ago"),
])
def test_spoken_time_ago(age_s, expected):
    assert spoken_time_ago((NOW_S - age_s) * 1000, NOW_S) == expected


def test_spoken_time_ago_none():
    assert spoken_time_ago(None, NOW_S) == "recently"


def test_home_coordinates():
    cfg = SimpleNamespace(weather_latitude="47.9", weather_longitude="-121.9")
    assert home_coordinates(cfg) == (47.9, -121.9)
    assert home_coordinates(SimpleNamespace(weather_latitude="",
                                            weather_longitude="")) is None
    assert home_coordinates(None) is None


async def test_fetch_json_raises_on_http_error(monkeypatch):
    import httpx

    def handler(request):
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    orig_client = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_json("https://example.com/x")


async def test_fetch_json_returns_payload(monkeypatch):
    import httpx

    def handler(request):
        assert request.url.params.get("q") == "1"
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    orig_client = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)
    assert await fetch_json("https://example.com/x", params={"q": "1"}) == {"ok": True}


def test_number_to_words_negative():
    # Regression: negatives raised KeyError.
    assert number_to_words(-5) == "minus five"
    assert number_to_words(-42) == "minus forty two"
    assert number_to_words(-250) == "minus 250"


# --- voice-dial policy (toll-fraud screen) -----------------------------------

from plugins.helpers import check_voice_dial_allowed, normalize_dial_target  # noqa: E402


@pytest.mark.parametrize("target", [
    "+44 20 7946 0958",      # international
    "011-44-20-7946-0958",   # US international prefix, with dashes
    "0044 20 7946 0958",     # 00 international prefix
    "1-900-555-0100",        # premium rate
    "(976) 555-0100",        # premium rate, parenthesised
    "+1 900 555 0100",
])
def test_voice_dial_default_deny_pattern(config_factory, target):
    assert check_voice_dial_allowed(target, config_factory()) is not None


@pytest.mark.parametrize("target", ["2001", "555-123-4567", "+1 (555) 123-4567"])
def test_voice_dial_allows_domestic(config_factory, target):
    assert check_voice_dial_allowed(target, config_factory()) is None


def test_voice_dial_allow_pattern_must_fullmatch(config_factory):
    cfg = config_factory(voice_dial_allow_pattern=r"2\d{3}")
    assert check_voice_dial_allowed("2001", cfg) is None
    assert check_voice_dial_allowed("20012", cfg) is not None
    assert check_voice_dial_allowed("5551234567", cfg) is not None


def test_voice_dial_rejects_raw_sip_uri_by_default(config_factory):
    assert check_voice_dial_allowed("sip:2001@evil.example", config_factory()) is not None


def test_voice_dial_screens_sip_uri_user_part_when_allowed(config_factory):
    cfg = config_factory(outbound_allow_sip_uri="true")
    assert check_voice_dial_allowed("sip:2001@pbx", cfg) is None
    assert check_voice_dial_allowed("sip:+442079460958@pbx", cfg) is not None


def test_voice_dial_bad_pattern_fails_closed(config_factory):
    cfg = config_factory(voice_dial_deny_pattern="(")
    assert check_voice_dial_allowed("2001", cfg) is not None


def test_normalize_dial_target():
    assert normalize_dial_target(" (555) 123-4567 ") == "5551234567"
    assert normalize_dial_target("sip:2001@pbx.example.com") == "sip:2001@pbx.example.com"
