"""Unit tests for the HANGUP tool's delayed-hangup scoping.

The tool waits a few seconds so the spoken goodbye can finish playing. During
that window the call it was scheduled for can end and a *different* caller can
become the live call — the delayed task must never drop that unrelated call.
"""
import asyncio
from types import SimpleNamespace

import pytest

import plugins.hangup_tool as hangup_tool
from plugins.hangup_tool import HangupTool
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


class _Assistant:
    """Minimal assistant whose `current_call` can be swapped mid-test."""

    def __init__(self, call):
        self.config = None
        self.current_call = call
        self.hung_up = []
        self.sip_handler = SimpleNamespace(hangup_call=self._hangup)

    async def _hangup(self, call_info):
        self.hung_up.append(call_info)


@pytest.fixture(autouse=True)
def _no_real_delay(monkeypatch):
    """Keep the delayed task's structure but make the sleep instant."""
    monkeypatch.setattr(hangup_tool, "HANGUP_DELAY_SECONDS", 0)


async def _drain():
    """Let the fire-and-forget delayed_hangup task run to completion."""
    while HangupTool._pending_tasks:
        await asyncio.gather(*list(HangupTool._pending_tasks))


async def test_hangs_up_the_call_it_was_scheduled_for():
    call_a = SimpleNamespace(call_id="A")
    assistant = _Assistant(call_a)
    tool = HangupTool(assistant)

    result = await tool.execute({})
    assert result.status == ToolStatus.SUCCESS
    await _drain()

    assert assistant.hung_up == [call_a]


async def test_does_not_hang_up_a_different_caller():
    """Regression: A's delayed hangup must not drop B's live call.

    Caller A says goodbye (HANGUP scheduled), A's leg drops, and caller B
    dials in before the delay elapses. Re-reading `current_call` at fire time
    would disconnect B mid-greeting.
    """
    call_a = SimpleNamespace(call_id="A")
    call_b = SimpleNamespace(call_id="B")
    assistant = _Assistant(call_a)
    tool = HangupTool(assistant)

    await tool.execute({})
    # A ends, B becomes the live call while the delayed task is pending.
    assistant.current_call = call_b
    await _drain()

    assert assistant.hung_up == [], "hung up an unrelated caller's call"


async def test_no_hangup_when_call_already_gone():
    call_a = SimpleNamespace(call_id="A")
    assistant = _Assistant(call_a)
    tool = HangupTool(assistant)

    await tool.execute({})
    assistant.current_call = None  # A hung up on their own
    await _drain()

    assert assistant.hung_up == []


async def test_no_active_call_fails():
    assistant = _Assistant(None)
    tool = HangupTool(assistant)
    result = await tool.execute({})
    assert result.status == ToolStatus.FAILED
    assert not HangupTool._pending_tasks
