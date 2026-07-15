"""Unit tests for the KP_INDEX space weather tool (NOAA SWPC planetary K-index)."""
import pytest

import plugins.space_weather_tool as space_weather_tool
from plugins.space_weather_tool import KpIndexTool, _classify, _say_number
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


def _fake_fetch(payload):
    async def fake(url, params=None, headers=None):
        return payload
    return fake


# --- _classify ---------------------------------------------------------------

@pytest.mark.parametrize(
    "kp,expected",
    [
        (2.9, ("quiet", "")),
        (3.0, ("unsettled", "")),
        (4.0, ("active", "")),
        (5.0, ("minor storm", "G1")),
        (6.0, ("moderate storm", "G2")),
        (7.0, ("strong storm", "G3")),
        (8.0, ("severe storm", "G4")),
        (9.5, ("extreme storm", "G5")),
    ],
)
def test_classify_boundaries(kp, expected):
    assert _classify(kp) == expected


# --- _say_number --------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (5.0, "five"),
        (3.67, "three point seven"),
        (0.0, "zero"),
        (9.5, "nine point five"),
    ],
)
def test_say_number(value, expected):
    assert _say_number(value) == expected


# --- execute ------------------------------------------------------------------

async def test_happy_path_uses_last_entry(monkeypatch):
    payload = [
        {"time_tag": "2026-07-09T02:59:00", "kp_index": 1.0, "estimated_kp": 1.0},
        {"time_tag": "2026-07-09T03:00:00", "kp_index": 3.67, "estimated_kp": 3.67},
    ]
    monkeypatch.setattr(space_weather_tool, "_fetch_json", _fake_fetch(payload))
    tool = KpIndexTool(assistant=None)
    result = await tool.execute({})
    assert result.status == ToolStatus.SUCCESS
    assert "three point seven" in result.message
    assert "unsettled" in result.message
    assert result.data["kp"] == 3.67
    assert result.data["label"] == "unsettled"
    assert result.data["storm_level"] == ""
    assert result.data["time_tag"] == "2026-07-09T03:00:00"


async def test_estimated_kp_fallback(monkeypatch):
    payload = [{"time_tag": "2026-07-09T03:00:00", "kp_index": None, "estimated_kp": 4.33}]
    monkeypatch.setattr(space_weather_tool, "_fetch_json", _fake_fetch(payload))
    tool = KpIndexTool(assistant=None)
    result = await tool.execute({})
    assert result.status == ToolStatus.SUCCESS
    assert result.data["kp"] == 4.33
    assert result.data["label"] == "active"


async def test_empty_feed_fails(monkeypatch):
    monkeypatch.setattr(space_weather_tool, "_fetch_json", _fake_fetch([]))
    tool = KpIndexTool(assistant=None)
    result = await tool.execute({})
    assert result.status == ToolStatus.FAILED
    assert result.message == "Space weather data is not available right now."


async def test_garbage_feed_fails(monkeypatch):
    payload = [{"time_tag": "x", "kp_index": "not-a-number"}, "junk"]
    monkeypatch.setattr(space_weather_tool, "_fetch_json", _fake_fetch(payload))
    tool = KpIndexTool(assistant=None)
    result = await tool.execute({})
    assert result.status == ToolStatus.FAILED


async def test_network_error_returns_failed(monkeypatch):
    async def boom(url, params=None, headers=None):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(space_weather_tool, "_fetch_json", boom)
    tool = KpIndexTool(assistant=None)
    result = await tool.execute({})
    assert result.status == ToolStatus.FAILED
    assert result.message == "Space weather data is not available right now."


async def test_storm_appends_aurora_hint_high_latitudes(monkeypatch):
    payload = [{"time_tag": "2026-07-09T03:00:00", "kp_index": 5.33}]
    monkeypatch.setattr(space_weather_tool, "_fetch_json", _fake_fetch(payload))
    tool = KpIndexTool(assistant=None)
    result = await tool.execute({})
    assert result.status == ToolStatus.SUCCESS
    assert "G1 minor storm" in result.message
    assert "high latitudes" in result.message


async def test_strong_storm_mentions_mid_latitudes(monkeypatch):
    payload = [{"time_tag": "2026-07-09T03:00:00", "kp_index": 7.0}]
    monkeypatch.setattr(space_weather_tool, "_fetch_json", _fake_fetch(payload))
    tool = KpIndexTool(assistant=None)
    result = await tool.execute({})
    assert result.status == ToolStatus.SUCCESS
    assert "G3 strong storm" in result.message
    assert "mid-latitudes" in result.message


async def test_quiet_conditions_no_aurora_hint(monkeypatch):
    payload = [{"time_tag": "2026-07-09T03:00:00", "kp_index": 2.0}]
    monkeypatch.setattr(space_weather_tool, "_fetch_json", _fake_fetch(payload))
    tool = KpIndexTool(assistant=None)
    result = await tool.execute({})
    assert result.status == ToolStatus.SUCCESS
    assert "aurora" not in result.message
    assert "quiet" in result.message
