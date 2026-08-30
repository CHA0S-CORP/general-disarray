"""Unit tests for the MAP tool (plugins/map_tool.py): pure spoken-formatting
helpers plus execute() with a monkeypatched Nominatim/OSRM fetch."""
from types import SimpleNamespace

import pytest

from plugins import map_tool
from plugins.map_tool import (
    MapTool,
    _short_place,
    _spoken_miles,
    _spoken_minutes,
    _METERS_PER_MILE,
)
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


# --- Pure helpers ------------------------------------------------------------

@pytest.mark.parametrize(
    "meters,expected",
    [
        (0.0, "less than a tenth of a mile"),
        (_METERS_PER_MILE * 1, "1 mile"),
        (_METERS_PER_MILE * 4, "4 miles"),
        (_METERS_PER_MILE * 4.3, "4.3 miles"),
        (_METERS_PER_MILE * 25.4, "25 miles"),
    ],
)
def test_spoken_miles(meters, expected):
    assert _spoken_miles(meters) == expected


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (30, "about 1 minute"),
        (600, "about 10 minutes"),
        (3600, "about 1 hour"),
        (3660, "about 1 hour and 1 minute"),
        (5400, "about 1 hour and 30 minutes"),
        (7200, "about 2 hours"),
    ],
)
def test_spoken_minutes(seconds, expected):
    assert _spoken_minutes(seconds) == expected


def test_short_place_trims_civic_hierarchy():
    full = ("Balboa Park, 6th Avenue, San Diego, San Diego County, "
            "California, 92101, United States")
    assert _short_place(full, "fallback") == "Balboa Park, 6th Avenue"
    assert _short_place("", "fallback") == "fallback"
    # Bare house/postal numbers are skipped, not spoken as part of the name.
    assert _short_place("San Diego Zoo, 2920, San Diego", "x") == \
        "San Diego Zoo, San Diego"


# --- execute() ---------------------------------------------------------------

def _assistant(**cfg):
    cfg.setdefault("weather_latitude", "32.7157")
    cfg.setdefault("weather_longitude", "-117.1611")
    return SimpleNamespace(config=SimpleNamespace(**cfg), session=None)


def _patch(monkeypatch, *, geocode=None, route=None, fail_geocode=False):
    """Monkeypatch map_tool.fetch_json to serve Nominatim then OSRM."""
    async def fake(url, params=None, headers=None, **kwargs):
        if "nominatim" in url:
            if fail_geocode:
                return []
            g = geocode or {"lat": "32.7", "lon": "-117.1",
                            "display_name": "Balboa Park, San Diego"}
            return [g]
        # OSRM routing
        r = route or {"distance": _METERS_PER_MILE * 12, "duration": 1200}
        return {"code": "Ok", "routes": [r]}

    monkeypatch.setattr(map_tool, "fetch_json", fake)


@pytest.mark.asyncio
async def test_route_from_home(monkeypatch):
    _patch(monkeypatch)
    tool = MapTool(_assistant())
    result = await tool.execute({"destination": "Balboa Park"})
    assert result.status == ToolStatus.SUCCESS
    assert "from home" in result.message
    assert "12 miles" in result.message
    assert "about 20 minutes" in result.message
    assert result.data["distance_miles"] == 12.0
    assert result.data["duration_minutes"] == 20


@pytest.mark.asyncio
async def test_missing_destination_is_rejected(monkeypatch):
    _patch(monkeypatch)
    tool = MapTool(_assistant())
    result = await tool.execute({"destination": "   "})
    assert result.status == ToolStatus.FAILED


@pytest.mark.asyncio
async def test_unresolvable_place_reports_gracefully(monkeypatch):
    _patch(monkeypatch, fail_geocode=True)
    tool = MapTool(_assistant())
    result = await tool.execute({"destination": "Nowhereville"})
    # Not a hard failure - the model gets a speakable "couldn't find" message.
    assert result.status == ToolStatus.SUCCESS
    assert "couldn't find" in result.message.lower()


@pytest.mark.asyncio
async def test_disabled_without_home_coordinates():
    tool = MapTool(_assistant(weather_latitude="", weather_longitude=""))
    assert tool.enabled is False
