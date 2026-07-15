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
