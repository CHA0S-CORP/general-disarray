"""
Space Weather Tool Plugin
=========================
Reports the current planetary K-index (geomagnetic activity) from NOAA's
Space Weather Prediction Center, classified into plain-language conditions
with an aurora hint during geomagnetic storms.

Usage in conversation:
User: "Any chance of seeing the northern lights tonight?"
LLM: [TOOL:KP_INDEX]
"""

import logging
from typing import Any, Dict, Optional, Tuple

import httpx

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

SWPC_KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"

_UNAVAILABLE_MSG = "Space weather data is not available right now."

_DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


async def _fetch_json(url: str, params: Optional[Dict[str, Any]] = None,
                      headers: Optional[Dict[str, str]] = None) -> Any:
    """Module-level HTTP helper so tests can monkeypatch it."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


def _classify(kp: float) -> Tuple[str, str]:
    """Map a Kp value to (condition label, NOAA G storm level)."""
    if kp < 3:
        return ("quiet", "")
    if kp < 4:
        return ("unsettled", "")
    if kp < 5:
        return ("active", "")
    if kp >= 9:
        return ("extreme storm", "G5")
    storm = {
        5: ("minor storm", "G1"),
        6: ("moderate storm", "G2"),
        7: ("strong storm", "G3"),
        8: ("severe storm", "G4"),
    }
    return storm[int(kp)]


def _say_number(value: float) -> str:
    """Spell a Kp value naturally to one decimal: 5.0 -> 'five', 3.67 -> 'three point seven'."""
    rounded = round(value, 1)
    whole = int(rounded)
    tenth = int(round((rounded - whole) * 10))
    whole_word = _DIGIT_WORDS[whole] if 0 <= whole < len(_DIGIT_WORDS) else str(whole)
    if tenth == 0:
        return whole_word
    return f"{whole_word} point {_DIGIT_WORDS[tenth]}"


def _latest_kp(entries: Any) -> Optional[Tuple[float, str]]:
    """Return (kp, time_tag) from the newest entry with a usable Kp value."""
    if not isinstance(entries, list):
        return None
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        raw = entry.get("kp_index")
        if raw is None:
            raw = entry.get("estimated_kp")
        if raw is None:
            continue
        try:
            kp = float(raw)
        except (TypeError, ValueError):
            continue
        return kp, str(entry.get("time_tag", ""))
    return None


class KpIndexTool(BaseTool):
    """Current planetary K-index (geomagnetic activity) from NOAA SWPC."""

    name = "KP_INDEX"
    description = "Get the current planetary K-index geomagnetic activity level and aurora outlook"
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {}  # No parameters - always reports the latest reading

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        try:
            entries = await _fetch_json(SWPC_KP_URL)
        except Exception as e:
            logger.error(f"Kp index fetch error: {e}")
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE_MSG)

        latest = _latest_kp(entries)
        if latest is None:
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE_MSG)

        kp, time_tag = latest
        label, storm_level = _classify(kp)

        message = f"The Kp index is {_say_number(kp)} - geomagnetic conditions are {label}."
        if kp >= 5:
            latitudes = "mid-latitudes" if kp >= 7 else "high latitudes"
            message += f" That is a {storm_level} {label}; aurora may be visible at {latitudes}."

        log_event(logger, logging.INFO, f"Kp index: {kp} ({label})", event="kp_index_fetch")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            data={
                "kp": kp,
                "label": label,
                "storm_level": storm_level,
                "time_tag": time_tag,
            },
        )
