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

from plugins.helpers import fetch_json, home_coordinates, number_to_words, spoken_time_ago
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
# Cap the structured payload: the model only needs the top of the ranking,
# and a 1.0/week feed has thousands of entries — an unbounded list balloons
# the agent context past its turn timeout.
MAX_DATA_QUAKES = 10
VALID_SORTS = ("recent", "biggest", "nearest")

# USGS "place" strings often lead with a distance, e.g. "12 km SE of Ridgecrest, CA"
_PLACE_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*km\s+[NSEW]{1,3}\s+of\s+",
                              re.IGNORECASE)

# Spoken forms of the feed magnitude thresholds
_THRESHOLD_SPOKEN = {"1.0": "one", "2.5": "two point five", "4.5": "four point five"}

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


_num_word = number_to_words


def _spoken_magnitude(mag: float) -> str:
    """Speak a magnitude naturally: 4.6 -> 'four point six', 5.0 -> 'five'."""
    whole_str, frac_str = f"{abs(float(mag)):.1f}".split(".")
    words = _num_word(int(whole_str))
    if frac_str != "0":
        words += f" point {number_to_words(int(frac_str))}"
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


_ago = spoken_time_ago


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
    return await fetch_json(url, params=params, headers=headers)


class EarthquakeTool(BaseTool):
    """Report recent earthquakes from the USGS feeds."""

    name = "QUAKES"
    description = ("Get recent earthquakes from the US Geological Survey, "
                   "sorted and classified — the result always names the most "
                   "recent and the largest explicitly; optionally only those "
                   "near the local area")
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
        "sort": {
            "type": "string",
            "description": "Ranking to report: 'recent' (newest first, default), "
                           "'biggest' (largest first), or 'nearest' (closest "
                           "first; needs the configured location)",
            "required": False,
            "default": "recent",
        },
    }

    def _configured_coords(self) -> Optional[Tuple[float, float]]:
        """Configured latitude/longitude, or None when unset/unparseable."""
        return home_coordinates(self.config)

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        feed = _resolve_feed(params.get("min_magnitude", "2.5"))
        period = _resolve_period(params.get("period", "day"))
        sort = str(params.get("sort") or "recent").strip().lower()
        if sort not in VALID_SORTS:
            sort = "recent"
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
        home = self._configured_coords()
        if near:
            if home is None:
                # Degrade gracefully: report everything and note the skipped filter
                note = "near filter ignored: WEATHER_LATITUDE/WEATHER_LONGITUDE not configured"
                logger.info("QUAKES near filter requested but coordinates not configured")
            else:
                near_applied = True
                quakes = [q for q in quakes
                          if _haversine_km(home[0], home[1],
                                           q["lat"], q["lon"]) <= NEAR_RADIUS_KM]

        now_s = time.time()
        for q in quakes:
            q["ago"] = _ago(q["time_ms"], now_s) if q["time_ms"] is not None else "recently"
            if home is not None:
                q["distance_km"] = round(_haversine_km(home[0], home[1],
                                                       q["lat"], q["lon"]))

        # Rank per the requested sort; 'nearest' needs coordinates.
        if sort == "nearest" and home is None:
            sort = "recent"
            note = note or "nearest sort ignored: location not configured"
        if sort == "biggest":
            quakes.sort(key=lambda q: q["mag"], reverse=True)
        elif sort == "nearest":
            quakes.sort(key=lambda q: q.get("distance_km", 1e9))
        else:  # recent
            quakes.sort(key=lambda q: q["time_ms"] or 0, reverse=True)

        def _entry(q: Dict[str, Any]) -> Dict[str, Any]:
            entry = {"mag": q["mag"], "place": q["place"], "ago": q["ago"],
                     "lat": q["lat"], "lon": q["lon"]}
            if "distance_km" in q:
                entry["distance_km"] = q["distance_km"]
            return entry

        # Explicit classification so the model never has to re-rank the list:
        # most_recent and largest are always named, and `quakes` is capped —
        # the full feed can be thousands of entries and would balloon the
        # LLM context past the turn timeout.
        most_recent = max(quakes, key=lambda q: q["time_ms"] or 0, default=None)
        largest = max(quakes, key=lambda q: q["mag"], default=None)

        data: Dict[str, Any] = {
            "count": len(quakes),
            "period": period,
            "min_magnitude": feed,
            "near": near_applied,
            "sort": sort,
            "most_recent": _entry(most_recent) if most_recent else None,
            "largest": _entry(largest) if largest else None,
            "quakes": [_entry(q) for q in quakes[:MAX_DATA_QUAKES]],
        }
        if len(quakes) > MAX_DATA_QUAKES:
            data["note_truncated"] = (f"list capped at {MAX_DATA_QUAKES} of "
                                      f"{len(quakes)} (sorted by {sort})")
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

        def _describe(q: Dict[str, Any]) -> str:
            text = (f"magnitude {_spoken_magnitude(q['mag'])} near "
                    f"{_spoken_place(q['place'])}, {q['ago']}")
            if sort == "nearest" and "distance_km" in q:
                text += f", about {q['distance_km']} kilometers away"
            return text

        sentences = [f"{_num_word(count).capitalize()} {noun} in the last {period}{near_suffix}."]
        if count == 1:
            desc = _describe(quakes[0])
            sentences.append(desc[0].upper() + desc[1:] + ".")
        else:
            lead = {"recent": "The most recent was",
                    "biggest": "The largest was",
                    "nearest": "The closest was"}[sort]
            sentences.append(f"{lead} {_describe(quakes[0])}.")
            # One cross-ranking fact so "most recent" and "largest" are both
            # always available to the caller in a single tool round.
            if sort != "biggest" and largest is not None and largest is not quakes[0]:
                sentences.append(f"The largest was {_describe(largest)}.")
            elif sort == "biggest" and most_recent is not None and most_recent is not quakes[0]:
                sentences.append(f"The most recent was {_describe(most_recent)}.")
            elif len(quakes) > 1:
                sentences.append("Next: " + _describe(quakes[1]) + ".")
        message = " ".join(sentences)

        log_event(logger, logging.INFO,
                  f"QUAKES: {count} quakes ({feed}/{period}, sort={sort}, "
                  f"near={near_applied})",
                  event="quake_fetch")

        return ToolResult(status=ToolStatus.SUCCESS, message=message, data=data)
