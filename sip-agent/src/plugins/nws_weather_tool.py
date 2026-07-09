"""
NWS Forecast Tool Plugin
========================
Official US National Weather Service forecast for the configured coordinates
(WEATHER_LATITUDE / WEATHER_LONGITUDE). Distinct from the Tempest WEATHER tool,
which reports live station conditions - this one is the outlook.

Usage in conversation:
User: "What's the forecast for tomorrow?"
LLM: [TOOL:FORECAST:when=tomorrow]

User: "How's the week looking?"
LLM: [TOOL:FORECAST:when=week]
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

# NWS rejects requests without a descriptive User-Agent.
_HEADERS = {
    "User-Agent": "general-disarray (self-hosted voice assistant)",
    "Accept": "application/geo+json",
}

_VALID_WHEN = ("now", "today", "tonight", "tomorrow", "week")

_UNAVAILABLE = "The forecast is not available right now."

# Raw keys carried through into ToolResult.data for each selected period.
_SUBSET_KEYS = (
    "name", "temperature", "temperatureUnit", "shortForecast",
    "isDaytime", "windSpeed", "windDirection",
)


async def _fetch_json(url: str, params: Optional[Dict[str, Any]] = None,
                      headers: Optional[Dict[str, str]] = None) -> Any:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


def _select_period(periods: List[Dict[str, Any]], when: str) -> Dict[str, Any]:
    """Pick the period whose name matches `when` (case-insensitive), with a
    positional fallback: NWS never names a period "Tomorrow", so that falls
    back to the second period; "today"/"tonight" fall back to the first."""
    target = when.lower()
    for period in periods:
        if str(period.get("name") or "").lower() == target:
            return period
    if target == "tomorrow" and len(periods) > 1:
        return periods[1]
    return periods[0]


def _period_sentence(period: Dict[str, Any]) -> str:
    """One spoken sentence for a forecast period, leading with its name."""
    name = str(period.get("name") or "").strip() or "Later"
    temp = period.get("temperature")
    short = str(period.get("shortForecast") or "").strip()
    high_low = "high" if period.get("isDaytime") else "low"

    if short and temp is not None:
        return f"{name}: {short.lower()} with a {high_low} around {temp} degrees."
    if temp is not None:
        return f"{name}: a {high_low} around {temp} degrees."
    if short:
        return f"{name}: {short.lower()}."
    return f"{name}: no forecast details available."


def _now_sentence(period: Dict[str, Any]) -> str:
    """One spoken sentence for the current hourly period."""
    temp = period.get("temperature")
    short = str(period.get("shortForecast") or "").strip().lower()

    if temp is not None and short:
        return f"Right now it is {temp} degrees and {short}."
    if temp is not None:
        return f"Right now it is {temp} degrees."
    if short:
        return f"Right now it is {short}."
    return "Right now the conditions are not available."


def _period_subset(period: Dict[str, Any]) -> Dict[str, Any]:
    return {key: period.get(key) for key in _SUBSET_KEYS}


class NWSForecastTool(BaseTool):
    """Official National Weather Service forecast for the configured location."""

    name = "FORECAST"
    description = ("Get the official National Weather Service forecast for "
                   "right now, today, tonight, tomorrow, or the week ahead")
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {
        "when": {
            "type": "string",
            "description": "Which forecast: 'now', 'today', 'tonight', 'tomorrow', or 'week'",
            "required": False,
            "default": "today",
        }
    }

    def __init__(self, assistant):
        super().__init__(assistant)
        # Resolved gridpoint URLs, cached after the first points lookup.
        self._forecast_url: Optional[str] = None
        self._hourly_url: Optional[str] = None
        if self.config and not (self.config.weather_latitude and self.config.weather_longitude):
            self.enabled = False
            logger.info("FORECAST tool disabled - WEATHER_LATITUDE/WEATHER_LONGITUDE not configured")

    async def _resolve_urls(self) -> None:
        """Resolve the gridpoint forecast URLs for the configured coordinates
        (NWS step one) and cache them on the instance."""
        if self._forecast_url and self._hourly_url:
            return
        lat = self.config.weather_latitude.strip()
        lon = self.config.weather_longitude.strip()
        data = await _fetch_json(f"https://api.weather.gov/points/{lat},{lon}",
                                 headers=_HEADERS)
        props = (data or {}).get("properties") or {}
        forecast_url = props.get("forecast")
        hourly_url = props.get("forecastHourly")
        if not forecast_url or not hourly_url:
            raise ValueError("NWS points response missing forecast URLs")
        self._forecast_url = forecast_url
        self._hourly_url = hourly_url

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        when = str(params.get("when") or "today").strip().lower()
        if when not in _VALID_WHEN:
            when = "today"

        if not self.config or not (self.config.weather_latitude and self.config.weather_longitude):
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)

        try:
            await self._resolve_urls()
            url = self._hourly_url if when == "now" else self._forecast_url
            data = await _fetch_json(url, headers=_HEADERS)
            periods = ((data or {}).get("properties") or {}).get("periods") or []
        except Exception as e:
            logger.error(f"NWS forecast error: {e}")
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)

        if not periods:
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)

        if when == "now":
            chosen = [periods[0]]
            message = _now_sentence(periods[0])
        elif when == "week":
            chosen = [p for p in periods if p.get("isDaytime")][:3]
            if not chosen:
                return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)
            message = " ".join(_period_sentence(p) for p in chosen)
        else:
            chosen = [_select_period(periods, when)]
            message = _period_sentence(chosen[0])

        log_event(logger, logging.INFO, f"Forecast ({when}): {message}",
                  event="forecast_fetch")
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            data={"when": when, "periods": [_period_subset(p) for p in chosen]},
        )
