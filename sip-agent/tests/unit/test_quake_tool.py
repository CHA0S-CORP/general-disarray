"""Unit tests for the QUAKES tool (plugins/quake_tool.py): pure helpers plus
execute() with a monkeypatched USGS feed fetch."""
import time
from types import SimpleNamespace

import httpx
import pytest

from plugins import quake_tool
from plugins.quake_tool import (
    EarthquakeTool,
    _ago,
    _clean_place,
    _haversine_km,
    _resolve_feed,
    _resolve_period,
)
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit

NOW_S = 1_800_000_000.0


def _feature(mag, place, time_ms, lon, lat, depth=10.0):
    return {
        "properties": {"mag": mag, "place": place, "time": time_ms},
        "geometry": {"coordinates": [lon, lat, depth]},
    }


def _patch_feed(monkeypatch, features, captured=None):
    async def fake(url, params=None, headers=None):
        if captured is not None:
            captured["url"] = url
        return {"features": features}

    monkeypatch.setattr(quake_tool, "_fetch_json", fake)


# --- Pure helpers ------------------------------------------------------------

def test_haversine_denver_to_boulder():
    # Denver -> Boulder is roughly 39 km
    km = _haversine_km(39.7392, -104.9903, 40.0150, -105.2705)
    assert abs(km - 39.0) < 4.0


def test_clean_place_strips_distance_prefix():
    assert _clean_place("12 km SE of Ridgecrest, CA") == "Ridgecrest, CA"
    assert _clean_place("103km WNW of Anchor Point, Alaska") == "Anchor Point, Alaska"
    assert _clean_place("central Alaska") == "central Alaska"
    assert _clean_place(None) == ""


@pytest.mark.parametrize(
    "age_s,expected",
    [
        (60, "just now"),
        (600, "about ten minutes ago"),
        (2 * 3600, "about two hours ago"),
        (3 * 86400, "about three days ago"),
    ],
)
def test_ago_buckets(age_s, expected):
    assert _ago((NOW_S - age_s) * 1000, NOW_S) == expected


@pytest.mark.parametrize(
    "value,expected",
    [("2.5", "2.5"), ("significant", "significant"), ("9.9", "2.5"), (None, "2.5")],
)
def test_resolve_feed(value, expected):
    assert _resolve_feed(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [("hour", "hour"), ("week", "week"), ("century", "day"), (None, "day")],
)
def test_resolve_period(value, expected):
    assert _resolve_period(value) == expected


# --- execute() ----------------------------------------------------------------

async def test_happy_path_sorted_top_three_spoken(monkeypatch):
    now_ms = int(time.time() * 1000)
    features = [
        _feature(3.1, "10 km N of Barstow, CA", now_ms - 3600_000, -117.0, 34.9),
        _feature(4.6, "12 km SE of Ridgecrest, CA", now_ms - 2 * 3600_000, -117.5, 35.6),
        _feature(2.6, "5 km W of Fresno, CA", now_ms - 600_000, -119.8, 36.7),
        _feature(2.8, "8 km E of Ojai, CA", now_ms - 7200_000, -119.2, 34.4),
    ]
    captured = {}
    _patch_feed(monkeypatch, features, captured)

    tool = EarthquakeTool(assistant=None)
    result = await tool.execute({})

    assert result.status == ToolStatus.SUCCESS
    assert captured["url"].endswith("2.5_day.geojson")
    assert result.data["count"] == 4
    mags = [q["mag"] for q in result.data["quakes"]]
    assert mags == sorted(mags, reverse=True)
    assert result.message.startswith("Four quakes in the last day.")
    assert "four point six" in result.message
    assert "Ridgecrest, California" in result.message
    # Only the top three are spoken; the smallest quake stays in data only
    assert "Fresno" not in result.message
    assert result.data["quakes"][3]["place"] == "Fresno, CA"


async def test_invalid_params_fall_back_to_defaults(monkeypatch):
    captured = {}
    _patch_feed(monkeypatch, [], captured)

    tool = EarthquakeTool(assistant=None)
    result = await tool.execute({"min_magnitude": "9.9", "period": "century"})

    assert result.status == ToolStatus.SUCCESS
    assert captured["url"].endswith("2.5_day.geojson")
    assert result.data["min_magnitude"] == "2.5"
    assert result.data["period"] == "day"


async def test_near_filter_includes_and_excludes(monkeypatch, config_factory):
    cfg = config_factory(weather_latitude="39.7392", weather_longitude="-104.9903")
    assistant = SimpleNamespace(config=cfg, session=None)
    now_ms = int(time.time() * 1000)
    features = [
        _feature(3.2, "3 km W of Boulder, CO", now_ms, -105.2705, 40.0150),
        _feature(5.5, "near Tokyo, Japan", now_ms, 139.7, 35.7),
    ]
    _patch_feed(monkeypatch, features)

    tool = EarthquakeTool(assistant)
    result = await tool.execute({"near": True})

    assert result.status == ToolStatus.SUCCESS
    assert result.data["near"] is True
    assert result.data["count"] == 1
    assert result.data["quakes"][0]["place"] == "Boulder, CO"
    assert "Tokyo" not in result.message


async def test_near_without_coordinates_ignores_filter(monkeypatch, config_factory):
    cfg = config_factory(weather_latitude="", weather_longitude="")
    assistant = SimpleNamespace(config=cfg, session=None)
    now_ms = int(time.time() * 1000)
    features = [
        _feature(3.2, "3 km W of Boulder, CO", now_ms, -105.2705, 40.0150),
        _feature(5.5, "near Tokyo, Japan", now_ms, 139.7, 35.7),
    ]
    _patch_feed(monkeypatch, features)

    tool = EarthquakeTool(assistant)
    result = await tool.execute({"near": True})

    assert result.status == ToolStatus.SUCCESS
    assert result.data["count"] == 2
    assert result.data["near"] is False
    assert "note" in result.data


async def test_empty_features_success_message(monkeypatch):
    _patch_feed(monkeypatch, [])

    tool = EarthquakeTool(assistant=None)
    result = await tool.execute({})

    assert result.status == ToolStatus.SUCCESS
    assert result.message == "No earthquakes above magnitude two point five in the last day."
    assert result.data["count"] == 0


async def test_network_failure_returns_failed(monkeypatch):
    async def fake(url, params=None, headers=None):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(quake_tool, "_fetch_json", fake)

    tool = EarthquakeTool(assistant=None)
    result = await tool.execute({})

    assert result.status == ToolStatus.FAILED
    assert result.message == "Earthquake data is not available right now."
