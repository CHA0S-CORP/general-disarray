"""Unit tests for the WEATHER tool (NWS current conditions).

All HTTP is stubbed by monkeypatching the module-level _fetch_json helper with
a dispatcher keyed on URL: points -> stations -> latest observation, plus the
hourly-forecast fallback for sparse observations.
"""
import pytest
from types import SimpleNamespace

import plugins.weather_tool as weather_module
from plugins.weather_tool import WeatherTool, _build_summary
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit

POINTS_URL = "https://api.weather.gov/points/47.9,-121.9"
STATIONS_URL = "https://api.weather.gov/gridpoints/SEW/150,120/stations"
HOURLY_URL = "https://api.weather.gov/gridpoints/SEW/150,120/forecast/hourly"
STATION_ID = "https://api.weather.gov/stations/KPAE"
OBS_URL = f"{STATION_ID}/observations/latest"

POINTS_RESPONSE = {"properties": {"observationStations": STATIONS_URL,
                                  "forecastHourly": HOURLY_URL}}
STATIONS_RESPONSE = {"features": [
    {"id": STATION_ID, "properties": {"name": "Paine Field"}},
]}

FULL_OBS = {"properties": {
    "textDescription": "Partly Cloudy",
    "temperature": {"unitCode": "wmoUnit:degC", "value": 15.0},
    "windChill": {"unitCode": "wmoUnit:degC", "value": None},
    "heatIndex": {"unitCode": "wmoUnit:degC", "value": None},
    "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 85.2},
    "windSpeed": {"unitCode": "wmoUnit:km_h-1", "value": 13.0},
    "windGust": {"unitCode": "wmoUnit:km_h-1", "value": 30.0},
    "windDirection": {"unitCode": "wmoUnit:degree_(angle)", "value": 29.0},
}}

SPARSE_OBS = {"properties": {
    "textDescription": "",
    "temperature": {"unitCode": "wmoUnit:degC", "value": None},
}}

HOURLY_RESPONSE = {"properties": {"periods": [
    {"name": "", "temperature": 59, "temperatureUnit": "F",
     "shortForecast": "Partly Cloudy", "isDaytime": True},
]}}


def make_tool(config_factory):
    cfg = config_factory(weather_latitude="47.9", weather_longitude="-121.9")
    assistant = SimpleNamespace(config=cfg, session=None)
    return WeatherTool(assistant)


def install_fake_fetch(monkeypatch, responses, calls=None):
    """Dispatcher fake keyed on URL substring; records requested URLs."""
    async def fake(url, params=None, headers=None):
        if calls is not None:
            calls.append(url)
        for key, value in responses.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unexpected URL: {url}")
    monkeypatch.setattr(weather_module, "_fetch_json", fake)


async def test_current_conditions_from_observation(config_factory, monkeypatch):
    tool = make_tool(config_factory)
    calls = []
    # Insertion order matters: "/observations/latest" must match before the
    # broader "/stations" substring.
    install_fake_fetch(monkeypatch, {
        "api.weather.gov/points/": POINTS_RESPONSE,
        "/observations/latest": FULL_OBS,
        "/stations": STATIONS_RESPONSE,
    }, calls)

    result = await tool.execute({})

    assert result.status == ToolStatus.SUCCESS
    # 15 C -> 59 F, 13 km/h -> 8 mph, humid, station named
    assert "Paine Field" in result.message
    assert "59 degrees" in result.message
    assert "partly cloudy" in result.message
    assert result.data["temp_f"] == 59
    assert result.data["wind_mph"] == 8
    assert result.data["humidity"] == 85
    assert result.data["source"] == "observation"

    # Second call reuses the resolved station (no repeat points/stations).
    calls.clear()
    result = await tool.execute({})
    assert result.status == ToolStatus.SUCCESS
    assert calls == [OBS_URL]


async def test_sparse_observation_falls_back_to_hourly(config_factory, monkeypatch):
    tool = make_tool(config_factory)
    install_fake_fetch(monkeypatch, {
        "api.weather.gov/points/": POINTS_RESPONSE,
        "/observations/latest": SPARSE_OBS,
        "/forecast/hourly": HOURLY_RESPONSE,
        "/stations": STATIONS_RESPONSE,
    })

    result = await tool.execute({})

    assert result.status == ToolStatus.SUCCESS
    assert "59 degrees" in result.message
    assert result.data["source"] == "forecast"


async def test_api_error_fails_gracefully(config_factory, monkeypatch):
    tool = make_tool(config_factory)
    install_fake_fetch(monkeypatch, {
        "api.weather.gov/points/": RuntimeError("NWS down"),
    })

    result = await tool.execute({})
    assert result.status == ToolStatus.FAILED
    assert "not available" in result.message


def test_disabled_without_coordinates(config_factory):
    cfg = config_factory(weather_latitude="", weather_longitude="")
    tool = WeatherTool(SimpleNamespace(config=cfg, session=None))
    assert tool.enabled is False


async def test_zero_degree_wind_chill_kept(config_factory, monkeypatch):
    """A wind chill of exactly 0 F must not be discarded as falsy."""
    obs = {"properties": {
        "textDescription": "Clear",
        "temperature": {"unitCode": "wmoUnit:degC", "value": -9.4},   # 15 F
        "windChill": {"unitCode": "wmoUnit:degC", "value": -17.78},   # 0 F
        "heatIndex": {"unitCode": "wmoUnit:degC", "value": None},
        "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 50.0},
        "windSpeed": {"unitCode": "wmoUnit:km_h-1", "value": 20.0},
        "windGust": {"unitCode": "wmoUnit:km_h-1", "value": None},
        "windDirection": {"unitCode": "wmoUnit:degree_(angle)", "value": 0.0},
    }}
    tool = make_tool(config_factory)
    install_fake_fetch(monkeypatch, {
        "api.weather.gov/points/": POINTS_RESPONSE,
        "/observations/latest": obs,
        "/stations": STATIONS_RESPONSE,
    })

    result = await tool.execute({})

    assert result.status == ToolStatus.SUCCESS
    assert result.data["feels_like_f"] == 0
    assert "feels like 0" in result.message


def test_summary_feels_like_and_gusts():
    msg = _build_summary("Paine Field", "Clear", temp_f=40, feels_like_f=31,
                         humidity=50.0, wind_mph=22, gust_mph=35,
                         wind_dir_deg=270.0)
    assert "feels like 31" in msg
    assert "gusting to 35" in msg
    assert "west" in msg
    assert "quite windy" in msg
