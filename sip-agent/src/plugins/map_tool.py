"""
Map / Directions Tool Plugin
============================
Driving distance and travel time to a place, spoken for the phone. Geocodes
place names/addresses with OpenStreetMap Nominatim and routes with the public
OSRM server - both keyless, matching the no-API-key pattern of the other
information tools.

Origin defaults to the agent's home (WEATHER_LATITUDE / WEATHER_LONGITUDE, the
same coordinates the weather tools use); the caller can also ask for the route
between two named places.

Usage in conversation:
User: "How far is Balboa Park?"
LLM: [TOOL:MAP:destination=Balboa Park, San Diego]

User: "How long to drive from the airport to downtown?"
LLM: [TOOL:MAP:origin=San Diego airport,destination=downtown San Diego]
"""

import logging
from typing import Any, Dict, Optional, Tuple

from plugins.helpers import fetch_json, home_coordinates

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

# Nominatim's usage policy requires a descriptive, identifying User-Agent;
# requests with a generic client string get blocked.
_HEADERS = {"User-Agent": "general-disarray (self-hosted voice assistant)"}

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Public OSRM demo server: driving profile, coordinates as lon,lat pairs.
_OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

_METERS_PER_MILE = 1609.344

_UNAVAILABLE = "I can't look up directions right now."


def _spoken_miles(meters: float) -> str:
    """Distance in miles, spoken: one decimal under ten, whole miles above."""
    miles = meters / _METERS_PER_MILE
    if miles < 0.1:
        return "less than a tenth of a mile"
    if miles < 10:
        value = round(miles, 1)
        # Drop a trailing ".0" so TTS says "four miles", not "four point zero".
        text = str(int(value)) if value == int(value) else str(value)
        unit = "mile" if value == 1 else "miles"
        return f"{text} {unit}"
    value = round(miles)
    return f"{value} miles"


def _spoken_minutes(seconds: float) -> str:
    """Travel time, spoken: minutes, rolling into hours past sixty."""
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        unit = "minute" if minutes == 1 else "minutes"
        return f"about {minutes} {unit}"
    hours, mins = divmod(minutes, 60)
    hour_unit = "hour" if hours == 1 else "hours"
    if mins == 0:
        return f"about {hours} {hour_unit}"
    min_unit = "minute" if mins == 1 else "minutes"
    return f"about {hours} {hour_unit} and {mins} {min_unit}"


def _short_place(display_name: str, fallback: str) -> str:
    """A short, speakable label from a Nominatim display_name.

    Nominatim returns the full civic hierarchy ("Balboa Park, 6th Avenue, San
    Diego, San Diego County, California, 92101, United States"); the leading
    one or two segments name the place without the postal boilerplate. Bare
    numeric segments (house/postal numbers, e.g. "San Diego Zoo, 2920, ...")
    are skipped so the spoken label reads as a name, not a street address.
    """
    parts = [p.strip() for p in (display_name or "").split(",")
             if p.strip() and not p.strip().isdigit()]
    if not parts:
        return fallback
    return ", ".join(parts[:2])


class MapTool(BaseTool):
    """Driving distance and time to a place, via OpenStreetMap."""

    name = "MAP"
    description = ("Get driving distance and travel time to a place or address "
                   "(from home by default, or between two places)")
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {
        "destination": {
            "type": "string",
            "description": "Where to go - a place name or address",
            "required": True,
        },
        "origin": {
            "type": "string",
            "description": ("Starting point - a place name or address. "
                            "Omit to start from home."),
            "required": False,
        },
    }

    def __init__(self, assistant):
        super().__init__(assistant)
        # The origin defaults to the home coordinates; without them the tool
        # can only run when the caller names both endpoints, which is rare on a
        # phone call - self-disable to keep it off the tool list.
        if self.config and home_coordinates(self.config) is None:
            self.enabled = False
            logger.info("MAP tool disabled - WEATHER_LATITUDE/LONGITUDE not set")

    async def _geocode(self, place: str) -> Optional[Tuple[float, float, str]]:
        """Resolve a place string to (lat, lon, short label), or None."""
        try:
            results = await fetch_json(
                _NOMINATIM_URL,
                params={"q": place, "format": "json", "limit": 1},
                headers=_HEADERS,
            )
        except Exception as e:
            logger.error(f"Geocode error for '{place}': {e}")
            return None
        if not results:
            return None
        top = results[0]
        try:
            lat = float(top["lat"])
            lon = float(top["lon"])
        except (KeyError, TypeError, ValueError):
            return None
        return lat, lon, _short_place(str(top.get("display_name") or ""), place)

    async def _route(self, origin: Tuple[float, float],
                     dest: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Driving (distance_m, duration_s) for origin->dest, or None."""
        # OSRM wants lon,lat order.
        coords = (f"{origin[1]},{origin[0]};{dest[1]},{dest[0]}")
        try:
            data = await fetch_json(
                f"{_OSRM_URL}/{coords}",
                params={"overview": "false"},
            )
        except Exception as e:
            logger.error(f"Routing error: {e}")
            return None
        if (data or {}).get("code") != "Ok":
            return None
        routes = data.get("routes") or []
        if not routes:
            return None
        route = routes[0]
        try:
            return float(route["distance"]), float(route["duration"])
        except (KeyError, TypeError, ValueError):
            return None

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        destination = str(params.get("destination") or "").strip()
        if not destination:
            return ToolResult(status=ToolStatus.FAILED,
                              message="Where would you like directions to?")

        origin_text = str(params.get("origin") or "").strip()

        # Resolve the origin: a named place is geocoded; otherwise home.
        if origin_text:
            origin = await self._geocode(origin_text)
            if origin is None:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    message=f"I couldn't find a place called {origin_text}.")
            origin_coords = (origin[0], origin[1])
            origin_label = origin[2]
        else:
            home = home_coordinates(self.config) if self.config else None
            if home is None:
                return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)
            origin_coords = home
            origin_label = "home"

        dest = await self._geocode(destination)
        if dest is None:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message=f"I couldn't find a place called {destination}.")
        dest_coords = (dest[0], dest[1])
        dest_label = dest[2]

        routed = await self._route(origin_coords, dest_coords)
        if routed is None:
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)

        distance_m, duration_s = routed
        miles = _spoken_miles(distance_m)
        minutes = _spoken_minutes(duration_s)

        message = (f"{dest_label} is {miles} from {origin_label}, "
                   f"{minutes} by car.")

        log_event(logging.getLogger(__name__), logging.INFO,
                  f"Map route {origin_label} -> {dest_label}: {message}",
                  event="map_route", origin=origin_label, destination=dest_label)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            data={
                "origin": origin_label,
                "destination": dest_label,
                "distance_miles": round(distance_m / _METERS_PER_MILE, 1),
                "duration_minutes": round(duration_s / 60),
            },
        )
