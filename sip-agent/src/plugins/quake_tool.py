"""
Earthquake Tool Plugin
======================
Reports recent earthquakes from the public USGS GeoJSON summary feeds
(no API key required). Optionally filters to quakes within 500 km of the
configured coordinates (WEATHER_LATITUDE / WEATHER_LONGITUDE).

Usage in conversation:
User: "Have there been any earthquakes today?"
LLM: [TOOL:QUAKES]

User: "Any big quakes near us this week?"
LLM: [TOOL:QUAKES:min_magnitude=4.5,period=week,near=true]
"""

import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

USGS_FEED_URL = ("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"
                 "{feed}_{period}.geojson")
VALID_FEEDS = ("1.0", "2.5", "4.5", "significant")
VALID_PERIODS = ("hour", "day", "week")
NEAR_RADIUS_KM = 500.0
EARTH_RADIUS_KM = 6371.0
MAX_SPOKEN_QUAKES = 3

# USGS "place" strings often lead with a distance, e.g. "12 km SE of Ridgecrest, CA"
_PLACE_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*km\s+[NSEW]{1,3}\s+of\s+",
                              re.IGNORECASE)

# Spoken forms of the feed magnitude thresholds
_THRESHOLD_SPOKEN = {"1.0": "one", "2.5": "two point five", "4.5": "four point five"}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty",
         7: "seventy", 8: "eighty", 9: "ninety"}

# Expand trailing state abbreviations so TTS says "California", not "C A"
_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "the District of Columbia",
}


def _num_word(n: int) -> str:
    """Spell out a small integer for speech ('3' -> 'three'); large numbers stay digits."""
    n = int(n)
    if 0 <= n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("" if ones == 0 else " " + _ONES[ones])
    return str(n)


def _spoken_magnitude(mag: float) -> str:
    """Speak a magnitude naturally: 4.6 -> 'four point six', 5.0 -> 'five'."""
    whole_str, frac_str = f"{abs(float(mag)):.1f}".split(".")
    words = _num_word(int(whole_str))
    if frac_str != "0":
        words += f" point {_ONES[int(frac_str)]}"
    return ("minus " + words) if float(mag) < 0 else words


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _clean_place(place: Optional[str]) -> str:
    """Strip a leading distance prefix: '12 km SE of Ridgecrest, CA' -> 'Ridgecrest, CA'."""
    if not place:
        return ""
    return _PLACE_PREFIX_RE.sub("", place).strip()


def _spoken_place(place: str) -> str:
    """Speech-friendly place: expand a trailing US state abbreviation."""
    if not place:
        return "an unknown location"
    head, sep, tail = place.rpartition(", ")
    if sep and tail.strip() in _STATE_NAMES:
        return f"{head}, {_STATE_NAMES[tail.strip()]}"
    return place


def _ago(epoch_ms: float, now_s: float) -> str:
    """Spoken relative time for an epoch-milliseconds timestamp."""
    seconds = max(0.0, now_s - float(epoch_ms) / 1000.0)
    if seconds < 300:
        return "just now"
    if seconds < 3600:
        minutes = max(1, round(seconds / 60))
        unit = "minute" if minutes == 1 else "minutes"
        return f"about {_num_word(minutes)} {unit} ago"
    if seconds < 86400:
        hours = max(1, round(seconds / 3600))
        unit = "hour" if hours == 1 else "hours"
        return f"about {_num_word(hours)} {unit} ago"
    days = max(1, round(seconds / 86400))
    unit = "day" if days == 1 else "days"
    return f"about {_num_word(days)} {unit} ago"


def _resolve_feed(value: Any) -> str:
    """Map min_magnitude to a valid USGS feed name; unknown values -> '2.5'."""
    feed = str(value or "").strip().lower()
    return feed if feed in VALID_FEEDS else "2.5"


def _resolve_period(value: Any) -> str:
    """Map period to a valid feed period; unknown values -> 'day'."""
    period = str(value or "").strip().lower()
    return period if period in VALID_PERIODS else "day"


async def _fetch_json(url: str, params: Optional[Dict[str, Any]] = None,
                      headers: Optional[Dict[str, str]] = None) -> Any:
    """Fetch and decode JSON (module-level so tests can monkeypatch it)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


class EarthquakeTool(BaseTool):
    """Report recent earthquakes from the USGS feeds."""

    name = "QUAKES"
    description = ("Get recent earthquakes from the US Geological Survey, "
                   "optionally only those near the local area")
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {
        "min_magnitude": {
            "type": "string",
            "description": "Minimum magnitude feed: '1.0', '2.5', '4.5', or 'significant'",
            "required": False,
            "default": "2.5",
        },
        "period": {
            "type": "string",
            "description": "Time window: 'hour', 'day', or 'week'",
            "required": False,
            "default": "day",
        },
        "near": {
            "type": "boolean",
            "description": "Only quakes within 500 kilometers of the configured location",
            "required": False,
            "default": False,
        },
    }

    def _configured_coords(self) -> Optional[Tuple[float, float]]:
        """Configured latitude/longitude, or None when unset/unparseable."""
        if not self.config:
            return None
        try:
            return (float(self.config.weather_latitude),
                    float(self.config.weather_longitude))
        except (TypeError, ValueError):
            return None

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        feed = _resolve_feed(params.get("min_magnitude", "2.5"))
        period = _resolve_period(params.get("period", "day"))
        near = params.get("near", False)
        if isinstance(near, str):
            near = near.strip().lower() in ("true", "yes", "1")

        url = USGS_FEED_URL.format(feed=feed, period=period)
        try:
            payload = await _fetch_json(url)
        except Exception as e:
            logger.warning(f"USGS earthquake feed fetch failed: {e}")
            return ToolResult(status=ToolStatus.FAILED,
                              message="Earthquake data is not available right now.")

        quakes: List[Dict[str, Any]] = []
        for feature in (payload or {}).get("features") or []:
            props = feature.get("properties") or {}
            coords = (feature.get("geometry") or {}).get("coordinates") or []
            mag = props.get("mag")
            if mag is None or len(coords) < 2:
                continue
            quakes.append({
                "mag": float(mag),
                "place": _clean_place(props.get("place")),
                "time_ms": props.get("time"),
                "lat": float(coords[1]),
                "lon": float(coords[0]),
            })

        note = None
        near_applied = False
        if near:
            coords = self._configured_coords()
            if coords is None:
                # Degrade gracefully: report everything and note the skipped filter
                note = "near filter ignored: WEATHER_LATITUDE/WEATHER_LONGITUDE not configured"
                logger.info("QUAKES near filter requested but coordinates not configured")
            else:
                near_applied = True
                quakes = [q for q in quakes
                          if _haversine_km(coords[0], coords[1],
                                           q["lat"], q["lon"]) <= NEAR_RADIUS_KM]

        quakes.sort(key=lambda q: q["mag"], reverse=True)
        now_s = time.time()
        for q in quakes:
            q["ago"] = _ago(q["time_ms"], now_s) if q["time_ms"] is not None else "recently"

        data: Dict[str, Any] = {
            "count": len(quakes),
            "period": period,
            "min_magnitude": feed,
            "near": near_applied,
            "quakes": [{"mag": q["mag"], "place": q["place"], "ago": q["ago"],
                        "lat": q["lat"], "lon": q["lon"]} for q in quakes],
        }
        if note:
            data["note"] = note

        near_suffix = " near you" if near_applied else ""

        if not quakes:
            if feed == "significant":
                message = f"No significant earthquakes in the last {period}{near_suffix}."
            else:
                message = (f"No earthquakes above magnitude {_THRESHOLD_SPOKEN[feed]} "
                           f"in the last {period}{near_suffix}.")
            return ToolResult(status=ToolStatus.SUCCESS, message=message, data=data)

        count = len(quakes)
        noun = "quake" if count == 1 else "quakes"
        descriptions = [
            f"magnitude {_spoken_magnitude(q['mag'])} near {_spoken_place(q['place'])}, {q['ago']}"
            for q in quakes[:MAX_SPOKEN_QUAKES]
        ]

        sentences = [f"{_num_word(count).capitalize()} {noun} in the last {period}{near_suffix}."]
        if count == 1:
            sentences.append(descriptions[0][0].upper() + descriptions[0][1:] + ".")
        else:
            sentences.append(f"The largest was {descriptions[0]}.")
            if len(descriptions) > 1:
                sentences.append("Also " + ", and ".join(descriptions[1:]) + ".")
        message = " ".join(sentences)

        log_event(logger, logging.INFO,
                  f"QUAKES: {count} quakes ({feed}/{period}, near={near_applied})",
                  event="quake_fetch")

        return ToolResult(status=ToolStatus.SUCCESS, message=message, data=data)
