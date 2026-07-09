"""Unit tests for the pure random tools: DICE and COIN.

Both tools work with assistant=None; randomness is controlled by seeding or
monkeypatching the random module.
"""
import random

import pytest

import plugins.random_tools as random_tools
from plugins.random_tools import CoinTool, DiceTool, _clamp_int, _spell
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


# --- helpers -----------------------------------------------------------------

@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "zero"), (1, "one"), (11, "eleven"), (20, "twenty"),
        (21, "twenty-one"), (30, "thirty"), (99, "ninety-nine"),
        (100, "one hundred"), (118, "one hundred eighteen"),
        (1000, "one thousand"), (20000, "twenty thousand"),
        (-1, "negative one"),
    ],
)
def test_spell(n, expected):
    assert _spell(n) == expected


def test_spell_never_emits_digits():
    """Digits must never reach the TTS engine; dice totals reach 20 * 1000."""
    for n in range(0, 20001, 137):
        assert not any(ch.isdigit() for ch in _spell(n)), n


@pytest.mark.parametrize(
    "value,expected",
    [(5, 5), (1, 2), (9999, 1000), ("12", 12), ("garbage", 6), (None, 6)],
)
def test_clamp_int(value, expected):
    assert _clamp_int(value, 2, 1000, 6) == expected


# --- DICE ---------------------------------------------------------------------

async def test_dice_single_roll(monkeypatch):
    monkeypatch.setattr(random_tools.random, "randint", lambda a, b: 4)
    tool = DiceTool(assistant=None)
    result = await tool.execute({})
    assert result.status == ToolStatus.SUCCESS
    assert result.message == "I rolled a four."
    assert result.data == {"rolls": [4], "sides": 6, "total": 4}


async def test_dice_multiple_rolls_message(monkeypatch):
    rolls = iter([4, 1, 6])
    monkeypatch.setattr(random_tools.random, "randint", lambda a, b: next(rolls))
    tool = DiceTool(assistant=None)
    result = await tool.execute({"count": 3})
    assert result.status == ToolStatus.SUCCESS
    assert result.message == "Rolling three dice: four, one, and six. That is eleven total."
    assert result.data == {"rolls": [4, 1, 6], "sides": 6, "total": 11}


async def test_dice_clamps_bounds():
    tool = DiceTool(assistant=None)

    result = await tool.execute({"sides": 1, "count": 0})
    assert result.data["sides"] == 2
    assert len(result.data["rolls"]) == 1

    result = await tool.execute({"sides": 9999, "count": 50})
    assert result.data["sides"] == 1000
    assert len(result.data["rolls"]) == 20
    assert all(1 <= r <= 1000 for r in result.data["rolls"])


async def test_dice_rolls_within_range():
    random.seed(42)
    tool = DiceTool(assistant=None)
    result = await tool.execute({"sides": 20, "count": 10})
    assert result.status == ToolStatus.SUCCESS
    assert all(1 <= r <= 20 for r in result.data["rolls"])
    assert result.data["total"] == sum(result.data["rolls"])


# --- COIN ----------------------------------------------------------------------

async def test_coin_single_flip(monkeypatch):
    monkeypatch.setattr(random_tools.random, "choice", lambda seq: "heads")
    tool = CoinTool(assistant=None)
    result = await tool.execute({})
    assert result.status == ToolStatus.SUCCESS
    assert result.message == "Heads."
    assert result.data == {"flips": ["heads"], "heads": 1, "tails": 0}


async def test_coin_multiple_flips_message(monkeypatch):
    flips = iter(["heads", "tails", "heads"])
    monkeypatch.setattr(random_tools.random, "choice", lambda seq: next(flips))
    tool = CoinTool(assistant=None)
    result = await tool.execute({"count": 3})
    assert result.status == ToolStatus.SUCCESS
    assert result.message == "Heads, tails, and heads. Two heads, one tail."
    assert result.data == {"flips": ["heads", "tails", "heads"], "heads": 2, "tails": 1}


async def test_coin_all_same_side_reads_naturally(monkeypatch):
    """'Two heads, zero tails' is not how anyone says it."""
    monkeypatch.setattr(random_tools.random, "choice", lambda seq: "heads")
    tool = CoinTool(assistant=None)
    result = await tool.execute({"count": 3})
    assert result.message == "All heads. Three in a row."
    assert "zero" not in result.message


async def test_coin_clamps_count():
    tool = CoinTool(assistant=None)

    result = await tool.execute({"count": 0})
    assert len(result.data["flips"]) == 1

    result = await tool.execute({"count": 50})
    assert len(result.data["flips"]) == 20
    assert result.data["heads"] + result.data["tails"] == 20
