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


async def test_scheduled_calls_persist_across_restart(assistant, comp_config):
    from datetime import datetime, timedelta
    from tool_manager import ToolManager

    tm = assistant.tool_manager
    kept = await tm.schedule_task("scheduled_call", 3600, "future call",
                                  target_uri="1001", metadata={"extension": "1001"})
    # Timers are call-bound and must NOT survive a restart.
    await tm.schedule_task("timer", 3600, "call-bound timer")
    assert (comp_config.data_dir / "scheduled_tasks.json").exists()

    # Simulate a stale one-shot missed by more than the grace period.
    stale = await tm.schedule_task("scheduled_call", 3600, "stale call",
                                   target_uri="1002", metadata={"extension": "1002"})
    tm.scheduled_tasks[stale].execute_at = datetime.now() - timedelta(hours=2)
    tm._persist_tasks()

    # "Restart": a fresh manager reloads from the same data dir.
    tm2 = ToolManager(assistant)
    tm2._load_persisted_tasks()
    assert kept in tm2.scheduled_tasks           # future one-shot restored
    assert stale not in tm2.scheduled_tasks      # too-late one-shot dropped
    assert all(t.task_type != "timer" for t in tm2.scheduled_tasks.values())


async def test_plugin_autodiscovery_from_data_dir(assistant, comp_config):
    """A tool file dropped into data/plugins is discovered and registered."""
    from textwrap import dedent
    from tool_manager import ToolManager

    plugin_dir = comp_config.data_dir / "plugins"
    plugin_dir.mkdir(exist_ok=True)
    (plugin_dir / "ping_tool.py").write_text(dedent("""
        from tool_plugins import BaseTool, ToolResult, ToolStatus

        class PingTool(BaseTool):
            name = "PING_TEST"
            description = "test-only ping tool"
            parameters = {}

            async def execute(self, params):
                return ToolResult(status=ToolStatus.SUCCESS, message="pong")
    """))

    tm = ToolManager(assistant)
    assert tm.has_tool("PING_TEST")
    # Builtins are still explicitly registered, not shadowed by discovery.
    assert tm.has_tool("CALC")

    result = await tm.get_tool("PING_TEST").execute({})
    assert result.message == "pong"


async def test_cancel_task_removes_from_persistence(assistant, comp_config):
    import json as _json
    tm = assistant.tool_manager
    task_id = await tm.schedule_task("callback", 3600, "call me",
                                     target_uri="sip:1001@host")
    assert tm.cancel_task(task_id) is True
    assert tm.cancel_task(task_id) is False
    persisted = _json.loads((comp_config.data_dir / "scheduled_tasks.json").read_text())
    assert all(entry["id"] != task_id for entry in persisted)
