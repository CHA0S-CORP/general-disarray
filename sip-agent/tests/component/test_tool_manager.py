"""Component tests for ToolManager: execution, the CALLBACK caller-number
special-casing, and the scheduled-task bookkeeping.
"""
from types import SimpleNamespace

import pytest

from tool_manager import ToolStatus

pytestmark = pytest.mark.component


def _call(name, **params):
    # execute_tool only needs .name and .params.
    return SimpleNamespace(name=name, params=params, raw="")


async def test_execute_calc(assistant):
    result = await assistant.tool_manager.execute_tool(_call("CALC", expression="2+2"))
    assert result.status == ToolStatus.SUCCESS
    assert "4" in result.message


async def test_unknown_tool(assistant):
    result = await assistant.tool_manager.execute_tool(_call("NOPE"))
    assert result.status == ToolStatus.FAILED
    assert "unknown tool" in result.message.lower()


async def test_callback_defaults_to_caller_number(assistant):
    assistant.current_call = SimpleNamespace(remote_uri="sip:+15551234567@host")
    result = await assistant.tool_manager.execute_tool(_call("CALLBACK", delay=120))
    assert result.status == ToolStatus.SUCCESS
    # The manager intercepts CALLBACK and routes to assistant.schedule_callback.
    assert len(assistant.scheduled_callbacks) == 1
    delay, _message, destination = assistant.scheduled_callbacks[0]
    assert delay == 120
    assert destination == "sip:+15551234567@host"


async def test_callback_without_number_fails(assistant):
    assistant.current_call = None
    result = await assistant.tool_manager.execute_tool(_call("CALLBACK"))
    assert result.status == ToolStatus.FAILED
    assert assistant.scheduled_callbacks == []


async def test_schedule_and_cancel_tasks(assistant):
    tm = assistant.tool_manager
    task_id = await tm.schedule_task("timer", 3600, "ping")
    pending = tm.get_pending_tasks()
    assert any(t.id == task_id for t in pending)

    cancelled = await tm.cancel_tasks("all")
    assert cancelled >= 1
    assert tm.get_pending_tasks() == []
