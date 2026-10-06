"""
Shared plugin helpers
=====================
Utilities duplicated across the data-fetching tools (weather, forecast,
quakes, alerts, web search): a common JSON fetch, small-number speech
spelling, spoken relative time, and the home-coordinates gate.

Not a tool module — PluginLoader skips it (no BaseTool subclass).
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 10.0

# Unit abbreviations a TTS voice mangles or reads as letters ("oz" -> "oh zee").
# Each maps to (singular, plural); a preceding quantity of exactly "1" takes the
# singular, everything else (2, 0.5, 1/2, 1 1/2, ...) the plural. Only
# unambiguous measures are listed — bare "l"/"g" are omitted on purpose, since
# they collide with ordinary words.
_UNIT_EXPANSIONS = {
    "oz": ("ounce", "ounces"),
    "ozs": ("ounce", "ounces"),
    "tsp": ("teaspoon", "teaspoons"),
    "tsps": ("teaspoon", "teaspoons"),
    "tbsp": ("tablespoon", "tablespoons"),
    "tbsps": ("tablespoon", "tablespoons"),
    "ml": ("milliliter", "milliliters"),
    "cl": ("centiliter", "centiliters"),
    "dl": ("deciliter", "deciliters"),
    "qt": ("quart", "quarts"),
    "pt": ("pint", "pints"),
    "lb": ("pound", "pounds"),
    "lbs": ("pound", "pounds"),
    "mph": ("mile per hour", "miles per hour"),
    "kph": ("kilometer per hour", "kilometers per hour"),
    "kmh": ("kilometer per hour", "kilometers per hour"),
}

# quantity (int, decimal, fraction, or mixed "1 1/2") + unit token. The unit
# must end on a word boundary so "ml" doesn't fire inside "html".
_UNIT_RE = re.compile(
    r"(?<![A-Za-z])(\d+(?:\.\d+)?(?:\s+\d+/\d+)?|\d+/\d+)\s*"
    r"(" + "|".join(sorted(_UNIT_EXPANSIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def expand_units_for_speech(text: str) -> str:
    """Expand measurement abbreviations to full words for TTS.

    "2 oz white rum" -> "2 ounces white rum"; "1 oz lime" -> "1 ounce lime";
    "5 mph winds" -> "5 miles per hour winds". Only expands a token when it
    directly follows a quantity, so prose words are never touched.
    """
    if not text:
        return text

    def _sub(m: re.Match) -> str:
        qty, unit = m.group(1), m.group(2).lower()
        singular, plural = _UNIT_EXPANSIONS[unit]
        return f"{qty} {singular if _is_singular_quantity(qty) else plural}"

    return _UNIT_RE.sub(_sub, text)


def _is_singular_quantity(qty: str) -> bool:
    """Whether a measurement quantity takes a singular unit.

    Exactly "1", or a proper fraction below one ("1/2 ounce", "3/4 teaspoon" —
    read like "half an ounce"). Everything else — 2, 0.5, 1 1/2 — is plural.
    """
    qty = qty.strip()
    if qty == "1":
        return True
    m = re.fullmatch(r"(\d+)/(\d+)", qty)
    if m and int(m.group(2)) and int(m.group(1)) < int(m.group(2)):
        return True
    return False

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
    """Spell a small integer for speech ('3' -> 'three', -5 -> 'minus five');
    magnitudes >= 100 stay digits."""
    n = int(n)
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
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


# Characters a spoken/typed phone number carries for readability only.
_DIAL_PUNCTUATION_RE = re.compile(r"[\s\-().]")


def normalize_dial_target(target: str) -> str:
    """Strip readability punctuation (spaces, dashes, dots, parentheses) from a
    bare number ("(555) 123-4567" -> "5551234567"). SIP URIs are returned
    stripped of surrounding whitespace only (their dots are significant)."""
    target = str(target or "").strip()
    if target.lower().startswith(("sip:", "sips:")) or "@" in target:
        return target
    return _DIAL_PUNCTUATION_RE.sub("", target)


def _dial_number_part(target: str) -> str:
    """The number to screen: the user part of a SIP URI, else the target,
    normalized (punctuation removed) so "+44 20..." can't dodge the deny
    pattern by formatting."""
    m = re.match(r"(?i)sips?:([^@;>]+)", target)
    user = m.group(1) if m else target.split("@", 1)[0]
    return _DIAL_PUNCTUATION_RE.sub("", user)


def check_voice_dial_allowed(target: str, config) -> Optional[str]:
    """Dial-target policy for VOICE-initiated dialing (CALLBACK to a number
    other than the caller's own, TRANSFER). Returns an error string, or None
    when the target may be dialed.

    Layers on top of the REST policy (``api.check_extension_allowed``: raw
    SIP URIs / pattern) because these targets come from an untrusted caller
    talking to the LLM — toll-fraud protection:

    - VOICE_DIAL_ALLOW_PATTERN, when set, must fullmatch the number;
    - VOICE_DIAL_DENY_PATTERN (default: international and premium-rate
      prefixes) must not match it.

    The number is normalized (spaces/dashes/dots/parens removed; the user
    part of a SIP URI) before the patterns run. A malformed pattern fails
    CLOSED.
    """
    # Imported lazily: api.py is heavy and imports plugins indirectly.
    from api import check_extension_allowed

    target = normalize_dial_target(target)
    error = check_extension_allowed(target, config)
    if error:
        return error
    number = _dial_number_part(target)
    if not number:
        return "extension is required"

    allow = getattr(config, "voice_dial_allow_pattern", "") or ""
    deny = getattr(config, "voice_dial_deny_pattern", "") or ""
    try:
        if allow and not re.fullmatch(allow, number):
            return "number is not in the allowed voice-dial range"
        if deny and re.match(deny, number):
            return "number is blocked by the voice-dial policy"
    except re.error as e:
        logger.error(f"Invalid VOICE_DIAL_*_PATTERN ({e}) - refusing to dial")
        return "voice-dial policy is misconfigured"
    return None
