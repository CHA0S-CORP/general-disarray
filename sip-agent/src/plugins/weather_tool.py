"""
Weather Tool Plugin
===================
Current weather conditions from the National Weather Service (api.weather.gov)
for the configured coordinates:
- WEATHER_LATITUDE / WEATHER_LONGITUDE

Complements the FORECAST tool (same API, outlook periods): WEATHER answers
"what's it like outside right now" from the nearest NWS observation station.
Station observations frequently carry null fields, so when the essentials are
missing the tool falls back to the current hourly-forecast period.

Usage in conversation:
User: "What's the weather like?"
LLM: [TOOL:WEATHER]
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

from plugins.helpers import fetch_json

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

# NWS rejects requests without a descriptive User-Agent.
_HEADERS = {
    "User-Agent": "general-disarray (self-hosted voice assistant)",
    "Accept": "application/geo+json",
}

_UNAVAILABLE = "Current weather conditions are not available right now."


async def _fetch_json(url: str, params: Optional[Dict[str, Any]] = None,
                      headers: Optional[Dict[str, str]] = None) -> Any:
    # follow_redirects: NWS 301-redirects /points/ URLs to its canonical
    # coordinate form.
    return await fetch_json(url, params=params, headers=headers,
                            follow_redirects=True)


def _quantity(props: Dict[str, Any], key: str) -> Tuple[Optional[float], str]:
    """Value + unit code of an NWS quantitative field ({value, unitCode})."""
    field = props.get(key) or {}
    value = field.get("value")
    return (float(value) if value is not None else None,
            str(field.get("unitCode") or ""))


def _to_fahrenheit(value: Optional[float], unit_code: str) -> Optional[int]:
    if value is None:
        return None
    if "degF" in unit_code:
        return round(value)
    return round(value * 9 / 5 + 32)


def _to_mph(value: Optional[float], unit_code: str) -> Optional[int]:
    if value is None:
        return None
    if "m_s" in unit_code:
        return round(value * 2.237)
    return round(value * 0.621371)  # NWS default: km_h


def _wind_direction(degrees: Optional[float]) -> str:
    """Convert degrees to a spoken cardinal direction."""
    if degrees is None:
        return "unknown"
    directions = [
        "north", "north-northeast", "northeast", "east-northeast",
        "east", "east-southeast", "southeast", "south-southeast",
        "south", "south-southwest", "southwest", "west-southwest",
        "west", "west-northwest", "northwest", "north-northwest"
    ]
    idx = int((degrees + 11.25) / 22.5) % 16
    return directions[idx]


def _build_summary(station_name: str, description: str,
                   temp_f: Optional[int], feels_like_f: Optional[int],
                   humidity: Optional[float], wind_mph: Optional[int],
                   gust_mph: Optional[int],
                   wind_dir_deg: Optional[float]) -> str:
    """A natural, conversational current-conditions summary."""
    parts = []

    opening = f"At {station_name}" if station_name else "Right now"
    condition = f"{description.lower()} and " if description else ""

    if temp_f is not None:
        if feels_like_f is not None and abs(temp_f - feels_like_f) > 2:
            parts.append(f"{opening}, it's {condition}{temp_f} degrees, "
                         f"feels like {feels_like_f}")
        else:
            parts.append(f"{opening}, it's {condition}{temp_f} degrees")
    elif description:
        parts.append(f"{opening}, it's {description.lower()}")
    else:
        return ""

    if humidity is not None:
        if humidity >= 85:
            parts.append(f"and very humid at {round(humidity)} percent")
        elif humidity <= 30:
            parts.append("and quite dry")

    if wind_mph is not None:
        if wind_mph == 0:
            parts.append("Wind is calm")
        else:
            wind = (f"Wind from the {_wind_direction(wind_dir_deg)} "
                    f"at {wind_mph} miles per hour")
            if gust_mph and gust_mph > wind_mph + 5:
                wind += f", gusting to {gust_mph}"
            if wind_mph >= 20:
                wind += ". It's quite windy"
            parts.append(wind)

    result = parts[0]
    for part in parts[1:]:
        result += (". " + part) if part[0].isupper() else (", " + part)
    return result + "."


class WeatherTool(BaseTool):
    """Current conditions from the nearest National Weather Service station."""

    name = "WEATHER"
    description = "Get current weather conditions for the local area"
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {}  # No parameters needed - uses configured coordinates

    def __init__(self, assistant):
        super().__init__(assistant)
        # Resolved per-location URLs, cached after the first points lookup.
        self._stations_url: Optional[str] = None
        self._hourly_url: Optional[str] = None
        self._station_obs_url: Optional[str] = None
        self._station_name: str = ""
        if self.config and not (self.config.weather_latitude
                                and self.config.weather_longitude):
            self.enabled = False
            # Read env directly (not config): TEMPEST_* no longer exist in
            # config.py, this only detects a pre-NWS deployment to warn it.
            if os.getenv("TEMPEST_STATION_ID") or os.getenv("TEMPEST_API_TOKEN"):
                logger.warning(
                    "WEATHER tool disabled - Tempest support was replaced by "
                    "the National Weather Service; TEMPEST_STATION_ID/"
                    "TEMPEST_API_TOKEN are ignored. Set WEATHER_LATITUDE/"
                    "WEATHER_LONGITUDE to re-enable the tool")
            else:
                logger.warning("WEATHER tool disabled - WEATHER_LATITUDE/"
                               "WEATHER_LONGITUDE not configured")

    async def _resolve_station(self) -> None:
        """Resolve and cache the nearest observation station for the
        configured coordinates (NWS points -> stations lookup)."""
        if self._station_obs_url:
            return
        lat = self.config.weather_latitude.strip()
        lon = self.config.weather_longitude.strip()
        points = await _fetch_json(
            f"https://api.weather.gov/points/{lat},{lon}", headers=_HEADERS)
        props = (points or {}).get("properties") or {}
        self._stations_url = props.get("observationStations")
        self._hourly_url = props.get("forecastHourly")
        if not self._stations_url:
            raise ValueError("NWS points response missing observationStations")

        stations = await _fetch_json(self._stations_url, headers=_HEADERS)
        features = (stations or {}).get("features") or []
        if not features:
            raise ValueError("NWS returned no observation stations")
        nearest = features[0]
        station_id = nearest.get("id")
        if not station_id:
            raise ValueError("NWS station entry missing id")
        self._station_obs_url = f"{station_id}/observations/latest"
        self._station_name = str(
            ((nearest.get("properties") or {}).get("name")) or "").strip()

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        if not self.config or not (self.config.weather_latitude
                                   and self.config.weather_longitude):
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)

        try:
            await self._resolve_station()
            obs = await _fetch_json(self._station_obs_url, headers=_HEADERS)
            props = (obs or {}).get("properties") or {}
        except Exception as e:
            logger.error(f"NWS observation error: {e}")
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)

        description = str(props.get("textDescription") or "").strip()
        temp_f = _to_fahrenheit(*_quantity(props, "temperature"))
        # Feels-like: NWS reports windChill and heatIndex separately; at most
        # one is non-null at a time. Explicit None checks: 0 degrees is a
        # legitimate feels-like value.
        feels_like_f = _to_fahrenheit(*_quantity(props, "windChill"))
        if feels_like_f is None:
            feels_like_f = _to_fahrenheit(*_quantity(props, "heatIndex"))
        if feels_like_f is None:
            feels_like_f = temp_f
        humidity, _ = _quantity(props, "relativeHumidity")
        wind_mph = _to_mph(*_quantity(props, "windSpeed"))
        gust_mph = _to_mph(*_quantity(props, "windGust"))
        wind_dir, _ = _quantity(props, "windDirection")

        if temp_f is None and not description:
            # Sparse/stale observation (common) - fall back to the current
            # hourly-forecast period for a usable "right now".
            return await self._forecast_fallback()

        summary = _build_summary(self._station_name, description, temp_f,
                                 feels_like_f, humidity, wind_mph, gust_mph,
                                 wind_dir)
        if not summary:
            return await self._forecast_fallback()

        log_event(logger, logging.INFO, f"Weather: {summary}",
                  event="weather_fetch")
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=summary,
            data={
                "temp_f": temp_f,
                "feels_like_f": feels_like_f,
                "humidity": round(humidity) if humidity is not None else None,
                "wind_mph": wind_mph,
                "wind_gust_mph": gust_mph,
                "wind_direction": wind_dir,
                "description": description,
                "station": self._station_name,
                "source": "observation",
            },
        )

    async def _forecast_fallback(self) -> ToolResult:
        """Current hourly-forecast period as a stand-in for a stale/sparse
        station observation."""
        if not self._hourly_url:
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)
        try:
            data = await _fetch_json(self._hourly_url, headers=_HEADERS)
            periods = ((data or {}).get("properties") or {}).get("periods") or []
        except Exception as e:
            logger.error(f"NWS hourly fallback error: {e}")
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)
        if not periods:
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)

        period = periods[0]
        temp = period.get("temperature")
        short = str(period.get("shortForecast") or "").strip().lower()
        if temp is not None and short:
            message = f"Right now it is about {temp} degrees and {short}."
        elif temp is not None:
            message = f"Right now it is about {temp} degrees."
        elif short:
            message = f"Right now it is {short}."
        else:
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)

        log_event(logger, logging.INFO, f"Weather (hourly fallback): {message}",
                  event="weather_fetch")
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            data={"temp_f": temp, "description": short, "source": "forecast"},
        )
