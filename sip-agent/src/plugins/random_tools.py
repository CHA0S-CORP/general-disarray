"""
Random Tools Plugin
===================
Pure-chance tools: DICE rolls dice, COIN flips coins. No config or network
needed; results are spoken back to the caller verbatim.

Usage in conversation:
User: "Roll a twenty-sided die"
LLM: [TOOL:DICE:sides=20]

User: "Flip three coins"
LLM: [TOOL:COIN:count=3]
"""

import logging
import random
from typing import Any, Dict, List

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

_NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]

_TENS_WORDS = {
    2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
    6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety",
}


def _spell(n: int) -> str:
    """Spell a number as words for natural speech (dice totals reach 20000).

    Digits must never reach the TTS engine, so this covers the whole range a
    roll can produce rather than just the first twenty.
    """
    if n < 0:
        return f"negative {_spell(-n)}"
    if n <= 20:
        return _NUMBER_WORDS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        word = _TENS_WORDS[tens]
        return word if ones == 0 else f"{word}-{_NUMBER_WORDS[ones]}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        word = f"{_NUMBER_WORDS[hundreds]} hundred"
        return word if rest == 0 else f"{word} {_spell(rest)}"
    thousands, rest = divmod(n, 1000)
    word = f"{_spell(thousands)} thousand"
    return word if rest == 0 else f"{word} {_spell(rest)}"


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    """Coerce value to an int and clamp it to [low, high]; fall back to default."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _spoken_list(items: List[str]) -> str:
    """Join words the way they are said aloud: 'a', 'a and b', 'a, b, and c'."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


class DiceTool(BaseTool):
    """Roll one or more dice."""

    name = "DICE"
    description = "Roll dice and report the results, for example a six-sided die or three twenty-sided dice"
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {
        "sides": {
            "type": "integer",
            "description": "Number of sides on each die (2 to 1000)",
            "required": False,
            "default": 6,
        },
        "count": {
            "type": "integer",
            "description": "How many dice to roll (1 to 20)",
            "required": False,
            "default": 1,
        },
    }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        sides = _clamp_int(params.get("sides", 6), 2, 1000, 6)
        count = _clamp_int(params.get("count", 1), 1, 20, 1)

        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls)

        if count == 1:
            message = f"I rolled a {_spell(rolls[0])}."
        else:
            rolls_spoken = _spoken_list([_spell(r) for r in rolls])
            message = (
                f"Rolling {_spell(count)} dice: {rolls_spoken}. "
                f"That is {_spell(total)} total."
            )

        log_event(
            logger, logging.INFO, f"Dice: {count}d{sides} -> {total}",
            event="dice_rolled", sides=sides, count=count, total=total,
        )

        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            data={"rolls": rolls, "sides": sides, "total": total},
        )


class CoinTool(BaseTool):
    """Flip one or more coins."""

    name = "COIN"
    description = "Flip one or more coins and report heads or tails"
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {
        "count": {
            "type": "integer",
            "description": "How many coins to flip (1 to 20)",
            "required": False,
            "default": 1,
        },
    }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        count = _clamp_int(params.get("count", 1), 1, 20, 1)

        flips = [random.choice(["heads", "tails"]) for _ in range(count)]
        heads = flips.count("heads")
        tails = flips.count("tails")

        if count == 1:
            message = f"{flips[0].capitalize()}."
        elif heads == 0 or tails == 0:
            # "Two heads, zero tails" is not how anyone says it.
            side = "heads" if heads else "tails"
            message = f"All {side}. {_spell(count).capitalize()} in a row."
        else:
            flips_spoken = _spoken_list(flips).capitalize()
            heads_part = f"{_spell(heads)} {'head' if heads == 1 else 'heads'}"
            tails_part = f"{_spell(tails)} {'tail' if tails == 1 else 'tails'}"
            message = f"{flips_spoken}. {heads_part.capitalize()}, {tails_part}."

        log_event(
            logger, logging.INFO, f"Coin: {count} flips -> {heads} heads",
            event="coin_flipped", count=count, heads=heads, tails=tails,
        )

        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            data={"flips": flips, "heads": heads, "tails": tails},
        )
