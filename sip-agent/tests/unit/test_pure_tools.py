"""Unit tests for the pure built-in tools (no assistant/network needed):
JOKE, DATETIME, SIMON_SAYS. STATUS needs the assistant and is covered in the
component tier.
"""
import pytest

from plugins.joke_tool import JokeTool
from plugins.datetime_tool import DateTimeTool
from plugins.simon_says_tool import SimonSaysTool
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


# --- JOKE ------------------------------------------------------------------

async def test_joke_default_category():
    tool = JokeTool(assistant=None)
    result = await tool.execute({})
    assert result.status == ToolStatus.SUCCESS
    assert result.message
    assert result.data["category"] == "general"


async def test_joke_specific_category():
    tool = JokeTool(assistant=None)
    result = await tool.execute({"category": "tech"})
    assert result.status == ToolStatus.SUCCESS
    assert result.message in JokeTool.JOKES["tech"]


async def test_joke_unknown_category_falls_back():
    tool = JokeTool(assistant=None)
    result = await tool.execute({"category": "nonexistent"})
    assert result.status == ToolStatus.SUCCESS
    assert result.data["category"] == "general"


# --- DATETIME --------------------------------------------------------------

@pytest.mark.parametrize(
    "fmt,needle",
    [
        ("time", "current time"),
        ("date", "Today is"),
        ("datetime", "It's"),
        ("full", "It's"),
    ],
)
async def test_datetime_formats(fmt, needle):
    tool = DateTimeTool(assistant=None)
    result = await tool.execute({"format": fmt, "timezone": "UTC"})
    assert result.status == ToolStatus.SUCCESS
    assert needle in result.message
    assert result.data["timezone"] == "UTC"


async def test_datetime_invalid_timezone_falls_back():
    tool = DateTimeTool(assistant=None)
    result = await tool.execute({"timezone": "Mars/Phobos"})
    assert result.status == ToolStatus.SUCCESS
    assert result.data["timezone"] == "US/Pacific"


# --- SIMON_SAYS ------------------------------------------------------------

async def test_simon_says_echoes_verbatim():
    tool = SimonSaysTool(assistant=None)
    phrase = "the eagle has landed"
    result = await tool.execute({"text": phrase})
    assert result.status == ToolStatus.SUCCESS
    assert result.message == phrase
    assert result.data["echoed"] == phrase


async def test_simon_says_empty_fails():
    tool = SimonSaysTool(assistant=None)
    result = await tool.execute({"text": ""})
    assert result.status == ToolStatus.FAILED
