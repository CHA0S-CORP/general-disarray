"""Unit tests for the STORY tool (LLM-backed storyteller, stubbed engine)."""
import pytest

from types import SimpleNamespace

from plugins.story_tool import StoryTool, _STORY_SYSTEM_PROMPT
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit

STORY_TEXT = ("Once upon a time a brave little turtle crossed the wide river. "
              "She paddled hard, made a friend, and got home before dark.")


class StubEngine:
    """Stands in for LLMEngine; records summarize_text calls."""

    def __init__(self, response=STORY_TEXT, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def summarize_text(self, system_prompt, text, timeout_s):
        self.calls.append((system_prompt, text, timeout_s))
        if self.error is not None:
            raise self.error
        return self.response


def make_assistant(engine):
    return SimpleNamespace(config=None, llm_engine=engine)


async def test_story_happy_path():
    engine = StubEngine()
    tool = StoryTool(make_assistant(engine))
    result = await tool.execute({"topic": "a brave turtle", "audience": "kids",
                                 "length": "short"})
    assert result.status == ToolStatus.SUCCESS
    assert result.message == STORY_TEXT
    assert result.data == {"topic": "a brave turtle", "audience": "kids",
                           "length": "short", "chars": len(STORY_TEXT)}

    system_prompt, request, timeout_s = engine.calls[0]
    assert system_prompt is _STORY_SYSTEM_PROMPT  # system prompt passed first
    assert timeout_s == 30.0
    assert "a brave turtle" in request
    assert "kids" in request


async def test_story_defaults_and_word_target_in_request():
    engine = StubEngine()
    tool = StoryTool(make_assistant(engine))
    result = await tool.execute({})
    assert result.status == ToolStatus.SUCCESS
    _, request, _ = engine.calls[0]
    assert "short" in request
    assert "120" in request
    assert "kids" in request
    assert result.data["audience"] == "kids"
    assert result.data["length"] == "short"


async def test_story_medium_adult():
    engine = StubEngine()
    tool = StoryTool(make_assistant(engine))
    result = await tool.execute({"audience": "adult", "length": "medium"})
    assert result.status == ToolStatus.SUCCESS
    _, request, _ = engine.calls[0]
    assert "250" in request
    assert "adult" in request


async def test_story_invalid_audience_and_length_fall_back():
    engine = StubEngine()
    tool = StoryTool(make_assistant(engine))
    result = await tool.execute({"audience": "robots", "length": "epic"})
    assert result.status == ToolStatus.SUCCESS
    assert result.data["audience"] == "kids"
    assert result.data["length"] == "short"


async def test_story_engine_returns_none_fails():
    engine = StubEngine(response=None)
    tool = StoryTool(make_assistant(engine))
    result = await tool.execute({"topic": "dragons"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "I could not think of a story just now."


async def test_story_engine_raises_returns_failed():
    engine = StubEngine(error=RuntimeError("llm exploded"))
    tool = StoryTool(make_assistant(engine))
    result = await tool.execute({})
    assert result.status == ToolStatus.FAILED
    assert result.message == "I could not think of a story just now."


async def test_story_missing_engine_fails():
    tool = StoryTool(SimpleNamespace(config=None, llm_engine=None))
    result = await tool.execute({})
    assert result.status == ToolStatus.FAILED
    assert result.message == "I cannot tell stories right now."


async def test_story_no_assistant_fails():
    tool = StoryTool(assistant=None)
    result = await tool.execute({})
    assert result.status == ToolStatus.FAILED
    assert result.message == "I cannot tell stories right now."
