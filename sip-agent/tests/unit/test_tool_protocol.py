"""Unit tests for the text-based tool-call protocol parsing in llm_engine.

The LLM emits `[TOOL:NAME:k=v,...]` markers; `_process_tool_calls` regex-parses
them, dispatches to the tool manager, and strips the markers from the spoken
text. Here we use a recording fake tool manager so the test is pure parsing.
"""
import pytest

from llm_engine import LLMEngine
from tool_plugins import ToolResult, ToolStatus

pytestmark = pytest.mark.unit


class RecordingToolManager:
    """Captures every ToolCall and returns a canned success result."""

    def __init__(self):
        self.calls = []

    async def execute_tool(self, tool_call):
        self.calls.append(tool_call)
        return ToolResult(status=ToolStatus.SUCCESS, message=f"ran {tool_call.name}")


@pytest.fixture
def engine_and_tm():
    tm = RecordingToolManager()
    return LLMEngine(config=None, tool_manager=tm), tm


async def test_parses_tool_with_params(engine_and_tm):
    engine, tm = engine_and_tm
    clean, results = await engine._process_tool_calls(
        "Sure thing. [TOOL:CALC:expression=2+2]"
    )
    assert len(tm.calls) == 1
    call = tm.calls[0]
    assert call.name == "CALC"
    assert call.params == {"expression": "2+2"}
    # Marker stripped from spoken text.
    assert "[TOOL" not in clean
    assert clean == "Sure thing."
    assert results[0]["tool"] == "CALC"


async def test_parses_tool_without_params(engine_and_tm):
    engine, tm = engine_and_tm
    clean, results = await engine._process_tool_calls("Goodbye! [TOOL:HANGUP]")
    assert len(tm.calls) == 1
    assert tm.calls[0].name == "HANGUP"
    assert tm.calls[0].params == {}
    assert "[TOOL" not in clean
    assert clean == "Goodbye!"


async def test_comma_inside_value_is_preserved(engine_and_tm):
    engine, tm = engine_and_tm
    await engine._process_tool_calls(
        "[TOOL:SET_TIMER:duration=300,message=Hello, world it is done]"
    )
    params = tm.calls[0].params
    assert params["duration"] == 300
    # The comma in the message value must not truncate it.
    assert params["message"] == "Hello, world it is done"


async def test_multiple_tool_calls(engine_and_tm):
    engine, tm = engine_and_tm
    clean, results = await engine._process_tool_calls(
        "[TOOL:CALC:expression=1+1] and [TOOL:JOKE]"
    )
    names = sorted(c.name for c in tm.calls)
    assert names == ["CALC", "JOKE"]
    assert "[TOOL" not in clean
    assert len(results) == 2


async def test_plain_text_has_no_tool_calls(engine_and_tm):
    engine, tm = engine_and_tm
    clean, results = await engine._process_tool_calls("Just a normal reply.")
    assert tm.calls == []
    assert results == []
    assert clean == "Just a normal reply."
