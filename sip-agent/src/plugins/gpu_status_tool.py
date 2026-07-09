"""
GPU Status Tool Plugin
======================
Reads GPU health (utilization, memory, temperature, power) from the Prometheus
instance that scrapes nvitop-exporter and speaks a one-sentence summary.

Requires configuration:
- PROMETHEUS_URL: base URL of the Prometheus server (part of the
  observability compose stack)

Usage in conversation:
User: "How's the GPU doing?"
LLM: [TOOL:GPU_STATUS]
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

# nvitop-exporter metric names vary with prometheus_client's unit-suffix
# behavior at exposition time, so each metric lists name variants to probe
# in order; the first one that returns a non-empty vector wins.
# prometheus_client appends the unit to the exposed metric name, so the
# nvitop-exporter series are e.g. `gpu_utilization_Percentage` (verified
# against a live exporter). The bare names are kept as fallbacks for other
# exporters / future versions; the first variant that returns data wins.
_METRICS: Dict[str, List[str]] = {
    "utilization": ["gpu_utilization_Percentage", "gpu_utilization",
                    "gpu_sm_utilization_Percentage", "gpu_sm_utilization"],
    "memory": ["gpu_memory_percent_Percentage", "gpu_memory_percent",
               "gpu_memory_utilization_Percentage", "gpu_memory_utilization"],
    "temperature": ["gpu_temperature_C", "gpu_temperature",
                    "gpu_temperature_celsius"],
    "power": ["gpu_power_usage_W", "gpu_power_usage", "gpu_power_usage_watts"],
}


async def _fetch_json(url: str, params: Optional[Dict[str, Any]] = None,
                      headers: Optional[Dict[str, str]] = None) -> Any:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


def _first_value(payload: Any) -> Optional[float]:
    """Extract the first sample value from a Prometheus instant-query vector.

    Returns None for an empty result vector or any malformed payload.
    """
    try:
        result = payload["data"]["result"]
        if not result:
            return None
        return float(result[0]["value"][1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


class GpuStatusTool(BaseTool):
    """Report current GPU utilization, memory, temperature, and power draw."""

    name = "GPU_STATUS"
    description = "Report the current GPU utilization, memory usage, temperature, and power draw"
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {}  # No parameters - always reports the local GPU

    def __init__(self, assistant):
        super().__init__(assistant)
        if self.config and not self.config.prometheus_url:
            self.enabled = False
            logger.info("GPU_STATUS tool disabled - Prometheus URL not configured")

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        base_url = self.config.prometheus_url if self.config else ""
        if not base_url:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="GPU monitoring is not configured",
            )

        query_url = base_url.rstrip("/") + "/api/v1/query"
        values: Dict[str, Optional[float]] = {key: None for key in _METRICS}

        try:
            for key, names in _METRICS.items():
                for name in names:
                    # avg() collapses multi-GPU hosts into one number
                    promql = "avg(" + name + ")"
                    payload = await _fetch_json(query_url, params={"query": promql})
                    value = _first_value(payload)
                    if value is not None:
                        values[key] = value
                        break
        except httpx.HTTPError as e:
            logger.warning(f"GPU status: Prometheus unreachable: {e}")
            return ToolResult(
                status=ToolStatus.FAILED,
                message="The monitoring stack is not reachable.",
            )
        except Exception as e:
            logger.error(f"GPU status error: {e}", exc_info=True)
            return ToolResult(
                status=ToolStatus.FAILED,
                message="I could not read the GPU metrics.",
            )

        if all(v is None for v in values.values()):
            return ToolResult(
                status=ToolStatus.FAILED,
                message="I could not read the GPU metrics.",
            )

        message = self._build_summary(values)
        log_event(logger, logging.INFO, f"GPU status: {message}", event="gpu_status")

        return ToolResult(status=ToolStatus.SUCCESS, message=message, data=values)

    @staticmethod
    def _build_summary(values: Dict[str, Optional[float]]) -> str:
        """Build the spoken sentence from whichever metrics resolved."""
        parts = []
        if values["utilization"] is not None:
            parts.append(f"at {round(values['utilization'])} percent")
        if values["memory"] is not None:
            parts.append(f"memory {round(values['memory'])} percent")
        if values["temperature"] is not None:
            parts.append(f"{round(values['temperature'])} degrees")
        if values["power"] is not None:
            parts.append(f"drawing {round(values['power'])} watts")
        return "The GPU is " + ", ".join(parts) + "."
