"""Unit tests for the ALERTS tool: parsers, source selection, spoken messages,
and network-failure handling. HTTP is stubbed by monkeypatching _fetch_json.
"""
from types import SimpleNamespace

import pytest

import plugins.alerts_tool as alerts_module
from plugins.alerts_tool import AlertsTool, _parse_alertmanager, _parse_prometheus
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


def make_tool(config_factory, **cfg_overrides):
    cfg = config_factory(**cfg_overrides)
    assistant = SimpleNamespace(config=cfg, session=None)
    return AlertsTool(assistant)


# --- _parse_alertmanager -----------------------------------------------------

def test_parse_alertmanager_happy_path():
    payload = [
        {"labels": {"alertname": "HighGPUTemp", "severity": "warning"},
         "status": {"state": "active"}, "startsAt": "2026-07-08T00:00:00Z"},
        {"labels": {"alertname": "DiskFull", "severity": "critical"},
         "status": {"state": "active"}},
    ]
    assert _parse_alertmanager(payload) == [
        {"name": "HighGPUTemp", "severity": "warning"},
        {"name": "DiskFull", "severity": "critical"},
    ]


def test_parse_alertmanager_missing_severity_is_unknown():
    payload = [{"labels": {"alertname": "NoSeverity"}}]
    assert _parse_alertmanager(payload) == [{"name": "NoSeverity", "severity": "unknown"}]


@pytest.mark.parametrize("garbage", [None, "nope", 42, {"data": []}, [None, "x", {"labels": None}, {}]])
def test_parse_alertmanager_garbage_returns_empty(garbage):
    assert _parse_alertmanager(garbage) == []


# --- _parse_prometheus -------------------------------------------------------

def test_parse_prometheus_filters_non_firing():
    payload = {
        "status": "success",
        "data": {
            "alerts": [
                {"labels": {"alertname": "HighGPUTemp", "severity": "warning"}, "state": "firing"},
                {"labels": {"alertname": "PendingOnly", "severity": "info"}, "state": "pending"},
                {"labels": {"alertname": "NoSeverity"}, "state": "firing"},
            ]
        },
    }
    assert _parse_prometheus(payload) == [
        {"name": "HighGPUTemp", "severity": "warning"},
        {"name": "NoSeverity", "severity": "unknown"},
    ]


@pytest.mark.parametrize("garbage", [None, [], "x", {"data": None}, {"data": {"alerts": "x"}}])
def test_parse_prometheus_garbage_returns_empty(garbage):
    assert _parse_prometheus(garbage) == []


# --- execute: source selection ----------------------------------------------

async def test_alertmanager_used_when_configured(monkeypatch, config_factory):
    tool = make_tool(config_factory, alertmanager_url="http://am:9093/",
                     prometheus_url="http://prom:9090")
    seen = {}

    async def fake_fetch(url, params=None, headers=None):
        seen["url"] = url
        seen["params"] = params
        return [{"labels": {"alertname": "HighGPUTemp", "severity": "warning"}}]

    monkeypatch.setattr(alerts_module, "_fetch_json", fake_fetch)
    result = await tool.execute({})

    assert seen["url"] == "http://am:9093/api/v2/alerts"
    assert seen["params"] == {"active": "true", "silenced": "false"}
    assert result.status == ToolStatus.SUCCESS
    assert result.data["source"] == "alertmanager"
    assert result.message == "One alert is firing: HighGPUTemp, severity warning."


async def test_prometheus_used_when_no_alertmanager(monkeypatch, config_factory):
    tool = make_tool(config_factory, alertmanager_url="",
                     prometheus_url="http://prom:9090/")
    seen = {}

    async def fake_fetch(url, params=None, headers=None):
        seen["url"] = url
        return {"status": "success", "data": {"alerts": [
            {"labels": {"alertname": "DiskFull", "severity": "critical"}, "state": "firing"},
            {"labels": {"alertname": "NotYet", "severity": "info"}, "state": "pending"},
        ]}}

    monkeypatch.setattr(alerts_module, "_fetch_json", fake_fetch)
    result = await tool.execute({})

    assert seen["url"] == "http://prom:9090/api/v1/alerts"
    assert result.status == ToolStatus.SUCCESS
    assert result.data["source"] == "prometheus"
    assert result.data["count"] == 1
    assert result.data["alerts"] == [{"name": "DiskFull", "severity": "critical"}]


# --- execute: spoken messages -------------------------------------------------

async def test_zero_alerts_message(monkeypatch, config_factory):
    tool = make_tool(config_factory, alertmanager_url="http://am:9093")

    async def fake_fetch(url, params=None, headers=None):
        return []

    monkeypatch.setattr(alerts_module, "_fetch_json", fake_fetch)
    result = await tool.execute({})

    assert result.status == ToolStatus.SUCCESS
    assert result.message == "No alerts are firing. Everything looks healthy."
    assert result.data["count"] == 0


async def test_many_alerts_caps_spoken_list_but_reports_total(monkeypatch, config_factory):
    tool = make_tool(config_factory, alertmanager_url="http://am:9093")
    payload = [
        {"labels": {"alertname": f"Alert{i}", "severity": "warning"}}
        for i in range(5)
    ]

    async def fake_fetch(url, params=None, headers=None):
        return payload

    monkeypatch.setattr(alerts_module, "_fetch_json", fake_fetch)
    result = await tool.execute({})

    assert result.status == ToolStatus.SUCCESS
    assert result.message.startswith("Five alerts are firing. The first three are:")
    assert "Alert0" in result.message
    assert "Alert2" in result.message
    assert "Alert3" not in result.message
    # data carries the full alert list, not just the spoken three
    assert result.data["count"] == 5
    assert len(result.data["alerts"]) == 5


# --- execute: network failure --------------------------------------------------

async def test_fetch_error_returns_failed(monkeypatch, config_factory):
    tool = make_tool(config_factory, alertmanager_url="http://am:9093")

    async def fake_fetch(url, params=None, headers=None):
        raise ConnectionError("boom")

    monkeypatch.setattr(alerts_module, "_fetch_json", fake_fetch)
    result = await tool.execute({})

    assert result.status == ToolStatus.FAILED
    assert result.message == "The monitoring stack is not reachable."
