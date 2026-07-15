"""
Shared plugin helpers
=====================
Utilities duplicated across the data-fetching tools (weather, forecast,
quakes, alerts, web search): a common JSON fetch, small-number speech
spelling, spoken relative time, and the home-coordinates gate.

Not a tool module — PluginLoader skips it (no BaseTool subclass).
"""

import logging
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 10.0

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty",
         7: "seventy", 8: "eighty", 9: "ninety"}


async def fetch_json(url: str, params: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None,
                     follow_redirects: bool = False,
                     timeout: float = DEFAULT_TIMEOUT_S) -> Any:
    """GET a JSON document (raises on HTTP errors, like the per-plugin
    wrappers it replaces)."""
    async with httpx.AsyncClient(timeout=timeout,
                                 follow_redirects=follow_redirects) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


def number_to_words(n: int) -> str:
    """Spell a small integer for speech ('3' -> 'three'); >=100 stays digits."""
    n = int(n)
    if 0 <= n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("" if ones == 0 else " " + _ONES[ones])
    return str(n)


def spoken_time_ago(epoch_ms: Optional[float], now_s: float) -> str:
    """Spoken relative age for an epoch-milliseconds timestamp."""
    if epoch_ms is None:
        return "recently"
    seconds = max(0.0, now_s - float(epoch_ms) / 1000.0)
    if seconds < 300:
        return "just now"
    if seconds < 3600:
        minutes = max(1, round(seconds / 60))
        unit = "minute" if minutes == 1 else "minutes"
        return f"about {number_to_words(minutes)} {unit} ago"
    if seconds < 86400:
        hours = max(1, round(seconds / 3600))
        unit = "hour" if hours == 1 else "hours"
        return f"about {number_to_words(hours)} {unit} ago"
    days = max(1, round(seconds / 86400))
    unit = "day" if days == 1 else "days"
    return f"about {number_to_words(days)} {unit} ago"


def home_coordinates(config) -> Optional[Tuple[float, float]]:
    """The configured WEATHER_LATITUDE/WEATHER_LONGITUDE pair, or None when
    unset/unparseable (the shared gate for location-aware tools)."""
    if not config:
        return None
    try:
        return (float(config.weather_latitude), float(config.weather_longitude))
    except (TypeError, ValueError):
        return None
