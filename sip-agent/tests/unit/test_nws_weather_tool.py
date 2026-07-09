"""Unit tests for the FORECAST tool (NWS two-step forecast API).

All HTTP is stubbed by monkeypatching the module-level _fetch_json helper with
a dispatcher keyed on URL: the points endpoint returns the gridpoint forecast
URLs, and each forecast URL returns canned periods.
"""
import pytest
from types import SimpleNamespace

import plugins.nws_weather_tool as nws_module
from plugins.nws_weather_tool import (
    NWSForecastTool,
    _period_sentence,
    _select_period,
)
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit

POINTS_URL = "https://api.weather.gov/points/39.7,-104.9"
FORECAST_URL = "https://api.weather.gov/gridpoints/BOU/62,61/forecast"
HOURLY_URL = "https://api.weather.gov/gridpoints/BOU/62,61/forecast/hourly"

FORECAST_PERIODS = [
    {"name": "Today", "temperature": 62, "temperatureUnit": "F",
     "shortForecast": "Sunny", "detailedForecast": "Sunny, with a high near 62.",
     "isDaytime": True, "windSpeed": "5 mph", "windDirection": "NW"},
    {"name": "Tonight", "temperature": 41, "temperatureUnit": "F",
     "shortForecast": "Partly Cloudy", "detailedForecast": "Partly cloudy.",
     "isDaytime": False, "windSpeed": "3 mph", "windDirection": "N"},
    {"name": "Wednesday", "temperature": 70, "temperatureUnit": "F",
     "shortForecast": "Mostly Sunny", "detailedForecast": "Mostly sunny.",
     "isDaytime": True, "windSpeed": "7 mph", "windDirection": "W"},
    {"name": "Wednesday Night", "temperature": 45, "temperatureUnit": "F",
     "shortForecast": "Clear", "detailedForecast": "Clear.",
     "isDaytime": False, "windSpeed": "3 mph", "windDirection": "W"},
    {"name": "Thursday", "temperature": 75, "temperatureUnit": "F",
     "shortForecast": "Chance Showers", "detailedForecast": "Showers likely.",
     "isDaytime": True, "windSpeed": "10 mph", "windDirection": "SW"},
]

HOURLY_PERIODS = [
    {"name": "", "temperature": 54, "temperatureUnit": "F",
     "shortForecast": "Partly Cloudy", "isDaytime": True,
     "windSpeed": "5 mph", "windDirection": "NW"},
]


def make_tool(config_factory):
    cfg = config_factory(weather_latitude="39.7", weather_longitude="-104.9")
    assistant = SimpleNamespace(config=cfg, session=None)
    return NWSForecastTool(assistant)


def make_fake_fetch(calls):
    """Dispatcher fake keyed on URL; appends each requested URL to `calls`."""
    async def fake(url, params=None, headers=None):
        calls.append(url)
        if "api.weather.gov/points/" in url:
            return {"properties": {"forecast": FORECAST_URL,
                                   "forecastHourly": HOURLY_URL}}
        if url == FORECAST_URL:
            return {"properties": {"periods": FORECAST_PERIODS}}
        if url == HOURLY_URL:
            return {"properties": {"periods": HOURLY_PERIODS}}
        raise AssertionError(f"unexpected URL: {url}")
    return fake


# --- execute: happy paths ----------------------------------------------------

async def test_today_forecast(monkeypatch, config_factory):
    calls = []
    monkeypatch.setattr(nws_module, "_fetch_json", make_fake_fetch(calls))
    tool = make_tool(config_factory)

    result = await tool.execute({"when": "today"})

    assert result.status == ToolStatus.SUCCESS
    assert result.message.startswith("Today:")
    assert "62 degrees" in result.message
    assert result.data["when"] == "today"
    assert result.data["periods"][0]["name"] == "Today"
    assert FORECAST_URL in calls


async def test_now_uses_hourly_url(monkeypatch, config_factory):
    calls = []
    monkeypatch.setattr(nws_module, "_fetch_json", make_fake_fetch(calls))
    tool = make_tool(config_factory)

    result = await tool.execute({"when": "now"})

    assert result.status == ToolStatus.SUCCESS
    assert result.message == "Right now it is 54 degrees and partly cloudy."
    assert HOURLY_URL in calls
    assert FORECAST_URL not in calls


async def test_week_summarizes_three_daytime_periods(monkeypatch, config_factory):
    monkeypatch.setattr(nws_module, "_fetch_json", make_fake_fetch([]))
    tool = make_tool(config_factory)

    result = await tool.execute({"when": "week"})

    assert result.status == ToolStatus.SUCCESS
    for name in ("Today:", "Wednesday:", "Thursday:"):
        assert name in result.message
    assert "Tonight:" not in result.message
    assert len(result.data["periods"]) == 3
    assert result.message.count(".") == 3  # one sentence per day


async def test_unknown_period_name_falls_back(monkeypatch, config_factory):
    monkeypatch.setattr(nws_module, "_fetch_json", make_fake_fetch([]))
    tool = make_tool(config_factory)

    # NWS never names a period "Tomorrow" -> falls back to periods[1]
    result = await tool.execute({"when": "tomorrow"})

    assert result.status == ToolStatus.SUCCESS
    assert result.message.startswith("Tonight:")
    assert "41 degrees" in result.message


async def test_points_lookup_cached_across_calls(monkeypatch, config_factory):
    calls = []
    monkeypatch.setattr(nws_module, "_fetch_json", make_fake_fetch(calls))
    tool = make_tool(config_factory)

    await tool.execute({"when": "today"})
    await tool.execute({"when": "tonight"})

    points_calls = [u for u in calls if "api.weather.gov/points/" in u]
    assert len(points_calls) == 1
    assert calls.count(FORECAST_URL) == 2


# --- execute: failure paths --------------------------------------------------

async def test_network_error_returns_failed(monkeypatch, config_factory):
    async def fake(url, params=None, headers=None):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(nws_module, "_fetch_json", fake)
    tool = make_tool(config_factory)

    result = await tool.execute({"when": "today"})

    assert result.status == ToolStatus.FAILED
    assert result.message == "The forecast is not available right now."


async def test_empty_periods_returns_failed(monkeypatch, config_factory):
    async def fake(url, params=None, headers=None):
        if "api.weather.gov/points/" in url:
            return {"properties": {"forecast": FORECAST_URL,
                                   "forecastHourly": HOURLY_URL}}
        return {"properties": {"periods": []}}
    monkeypatch.setattr(nws_module, "_fetch_json", fake)
    tool = make_tool(config_factory)

    result = await tool.execute({"when": "today"})

    assert result.status == ToolStatus.FAILED
    assert result.message == "The forecast is not available right now."


def test_disabled_without_coordinates(config_factory):
    cfg = config_factory(weather_latitude="", weather_longitude="")
    tool = NWSForecastTool(SimpleNamespace(config=cfg, session=None))
    assert tool.enabled is False


# --- pure helpers -------------------------------------------------------------

def test_select_period_matches_name_case_insensitively():
    assert _select_period(FORECAST_PERIODS, "TONIGHT")["name"] == "Tonight"
    assert _select_period(FORECAST_PERIODS, "wednesday")["name"] == "Wednesday"


def test_period_sentence_formats_speech():
    sentence = _period_sentence(FORECAST_PERIODS[1])
    assert sentence == "Tonight: partly cloudy with a low around 41 degrees."
    # Missing temperature still yields a spoken sentence
    assert _period_sentence({"name": "Friday", "shortForecast": "Sunny",
                             "isDaytime": True}) == "Friday: sunny."
