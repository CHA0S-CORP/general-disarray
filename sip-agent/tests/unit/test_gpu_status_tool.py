"""Unit tests for the GPU_STATUS tool (Prometheus/nvitop-exporter reader)."""
import pytest
from types import SimpleNamespace

import httpx

import plugins.gpu_status_tool as gpu_status_tool
from plugins.gpu_status_tool import GpuStatusTool, _first_value
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


def _vector(value):
    """A realistic Prometheus instant-query payload with one sample."""
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [{"metric": {}, "value": [1751900000.0, str(value)]}],
        },
    }


_EMPTY = {"status": "success", "data": {"resultType": "vector", "result": []}}


def _make_tool(config_factory, **cfg_overrides):
    cfg = config_factory(prometheus_url="http://prom:9090", **cfg_overrides)
    assistant = SimpleNamespace(config=cfg, session=None)
    return GpuStatusTool(assistant)


# --- _first_value ------------------------------------------------------------

def test_first_value_normal():
    assert _first_value(_vector("42.5")) == 42.5


def test_first_value_empty_result():
    assert _first_value(_EMPTY) is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"data": {}},
        {"data": {"result": [{"metric": {}}]}},
        {"data": {"result": [{"metric": {}, "value": [1751900000.0, "not-a-number"]}]}},
    ],
)
def test_first_value_malformed(payload):
    assert _first_value(payload) is None


# --- execute -----------------------------------------------------------------

async def test_happy_path_with_variant_fallback(monkeypatch, config_factory):
    # First utilization variant returns an empty vector; the tool must fall
    # through to the next name. Other metrics resolve on their first name --
    # the `_Percentage`/`_C`/`_W` suffixes are what nvitop-exporter really
    # exposes (prometheus_client appends the unit), verified against a live
    # exporter.
    responses = {
        "avg(gpu_utilization_Percentage)": _EMPTY,
        "avg(gpu_utilization)": _vector("42.5"),
        "avg(gpu_memory_percent_Percentage)": _vector("63.2"),
        "avg(gpu_temperature_C)": _vector("71.4"),
        "avg(gpu_power_usage_W)": _vector("94.9"),
    }

    async def fake_fetch(url, params=None, headers=None):
        assert url == "http://prom:9090/api/v1/query"
        return responses.get(params["query"], _EMPTY)

    monkeypatch.setattr(gpu_status_tool, "_fetch_json", fake_fetch)

    tool = _make_tool(config_factory)
    result = await tool.execute({})

    assert result.status == ToolStatus.SUCCESS
    assert result.message == (
        "The GPU is at 42 percent, memory 63 percent, 71 degrees, drawing 95 watts."
    )
    assert result.data == {
        "utilization": 42.5,
        "memory": 63.2,
        "temperature": 71.4,
        "power": 94.9,
    }


async def test_partial_metrics_temperature_only(monkeypatch, config_factory):
    async def fake_fetch(url, params=None, headers=None):
        if "temperature" in params["query"]:
            return _vector("71.4")
        return _EMPTY

    monkeypatch.setattr(gpu_status_tool, "_fetch_json", fake_fetch)

    tool = _make_tool(config_factory)
    result = await tool.execute({})

    assert result.status == ToolStatus.SUCCESS
    assert result.message == "The GPU is 71 degrees."
    assert result.data["temperature"] == 71.4
    assert result.data["utilization"] is None
    assert result.data["memory"] is None
    assert result.data["power"] is None


async def test_no_metrics_resolve_fails(monkeypatch, config_factory):
    async def fake_fetch(url, params=None, headers=None):
        return _EMPTY

    monkeypatch.setattr(gpu_status_tool, "_fetch_json", fake_fetch)

    tool = _make_tool(config_factory)
    result = await tool.execute({})

    assert result.status == ToolStatus.FAILED
    assert result.message == "I could not read the GPU metrics."


async def test_network_error_fails_gracefully(monkeypatch, config_factory):
    async def fake_fetch(url, params=None, headers=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(gpu_status_tool, "_fetch_json", fake_fetch)

    tool = _make_tool(config_factory)
    result = await tool.execute({})  # must not raise

    assert result.status == ToolStatus.FAILED
    assert result.message == "The monitoring stack is not reachable."


def test_self_disables_without_prometheus_url(config_factory):
    cfg = config_factory(prometheus_url="")
    assistant = SimpleNamespace(config=cfg, session=None)
    tool = GpuStatusTool(assistant)
    assert tool.enabled is False


async def test_metric_variants_include_real_exporter_names():
    """Regression: nvitop-exporter exposes unit-suffixed names, e.g.
    gpu_utilization_Percentage / gpu_temperature_C / gpu_power_usage_W.
    Guessed lowercase variants never matched a live Prometheus."""
    assert "gpu_utilization_Percentage" in gpu_status_tool._METRICS["utilization"]
    assert "gpu_memory_percent_Percentage" in gpu_status_tool._METRICS["memory"]
    assert "gpu_temperature_C" in gpu_status_tool._METRICS["temperature"]
    assert "gpu_power_usage_W" in gpu_status_tool._METRICS["power"]
