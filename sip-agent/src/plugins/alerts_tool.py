"""
Alerts Tool Plugin
==================
Reports currently firing monitoring alerts from the observability stack.

Prefers Alertmanager (`ALERTMANAGER_URL`) when configured; otherwise falls
back to querying Prometheus (`PROMETHEUS_URL`) directly and filtering to
alerts in the "firing" state.

Usage in conversation:
User: "Are any alerts going off?"
LLM: [TOOL:ALERTS]
"""

import logging
from typing import Any, Dict, List, Optional

from plugins.helpers import fetch_json, number_to_words

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

# How many alerts we read out loud; the total count is always spoken.
MAX_SPOKEN_ALERTS = 3

def _count_word(count: int) -> str:
    """Say small counts as words ('Three'), larger ones as digits ('14')."""
    if 0 <= count <= 10:
        return number_to_words(count).capitalize()
    return str(count)


async def _fetch_json(url: str, params: Optional[Dict[str, Any]] = None,
                      headers: Optional[Dict[str, str]] = None) -> Any:
    return await fetch_json(url, params=params, headers=headers)


def _parse_alertmanager(payload: Any) -> List[Dict[str, str]]:
    """Parse an Alertmanager /api/v2/alerts response into [{name, severity}]."""
    if not isinstance(payload, list):
        return []
    alerts: List[Dict[str, str]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        labels = entry.get("labels")
        if not isinstance(labels, dict):
            continue
        name = labels.get("alertname")
        if not isinstance(name, str) or not name:
            continue
        severity = labels.get("severity")
        if not isinstance(severity, str) or not severity:
            severity = "unknown"
        alerts.append({"name": name, "severity": severity})
    return alerts


def _parse_prometheus(payload: Any) -> List[Dict[str, str]]:
    """Parse a Prometheus /api/v1/alerts response, keeping only firing alerts."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    raw_alerts = data.get("alerts")
    if not isinstance(raw_alerts, list):
        return []
    alerts: List[Dict[str, str]] = []
    for entry in raw_alerts:
        if not isinstance(entry, dict):
            continue
        if entry.get("state") != "firing":
            continue
        labels = entry.get("labels")
        if not isinstance(labels, dict):
            continue
        name = labels.get("alertname")
        if not isinstance(name, str) or not name:
            continue
        severity = labels.get("severity")
        if not isinstance(severity, str) or not severity:
            severity = "unknown"
        alerts.append({"name": name, "severity": severity})
    return alerts


def _build_message(alerts: List[Dict[str, str]]) -> str:
    """Build the spoken summary for a list of firing alerts."""
    count = len(alerts)
    if count == 0:
        return "No alerts are firing. Everything looks healthy."
    if count == 1:
        alert = alerts[0]
        return f"One alert is firing: {alert['name']}, severity {alert['severity']}."
    spoken = alerts[:MAX_SPOKEN_ALERTS]
    listed = " ".join(f"{a['name']}, severity {a['severity']}." for a in spoken)
    return (f"{_count_word(count)} alerts are firing. "
            f"The first {_count_word(len(spoken)).lower()} are: {listed}")


class AlertsTool(BaseTool):
    """Report currently firing monitoring alerts."""

    name = "ALERTS"
    description = "Check whether any monitoring alerts are currently firing"
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters: Dict[str, Dict[str, Any]] = {}

    def __init__(self, assistant):
        super().__init__(assistant)
        if self.config and not (self.config.alertmanager_url or self.config.prometheus_url):
            self.enabled = False
            logger.info("ALERTS tool disabled - no Alertmanager or Prometheus URL configured")

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        alertmanager_url = self.config.alertmanager_url if self.config else ""
        prometheus_url = self.config.prometheus_url if self.config else ""

        if not alertmanager_url and not prometheus_url:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="The monitoring stack is not configured.",
            )

        try:
            if alertmanager_url:
                source = "alertmanager"
                payload = await _fetch_json(
                    alertmanager_url.rstrip("/") + "/api/v2/alerts",
                    params={"active": "true", "silenced": "false"},
                )
                alerts = _parse_alertmanager(payload)
            else:
                source = "prometheus"
                payload = await _fetch_json(prometheus_url.rstrip("/") + "/api/v1/alerts")
                alerts = _parse_prometheus(payload)
        except Exception as e:
            logger.warning(f"Alerts fetch failed: {e}")
            return ToolResult(
                status=ToolStatus.FAILED,
                message="The monitoring stack is not reachable.",
            )

        log_event(logger, logging.INFO, f"Alerts check: {len(alerts)} firing via {source}",
                  event="alerts_check", count=len(alerts), source=source)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=_build_message(alerts),
            data={"count": len(alerts), "source": source, "alerts": alerts},
        )
