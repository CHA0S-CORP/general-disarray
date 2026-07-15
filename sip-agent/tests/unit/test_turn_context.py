"""Unit tests: TurnContext isolation.

Per-turn tool bookkeeping (speak_result messages pending the relay check,
executed-tool counts) lives in a TurnContext owned by each
generate_response/stream_response invocation — never on the engine instance.
Two interleaved turns on ONE engine must not cross-contaminate: with the old
instance fields, the second turn's reset would silently drop the first
turn's pending speak_result message.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from llm_engine import LLMEngine, TurnContext, _StreamedToolCall

pytestmark = pytest.mark.unit


class StubTool:
    """Informational tool: its result message must reach the caller."""
    speak_result = True
    enabled = True
    description = "echoes its id"
    parameters = {}


class StubToolManager:
    def __init__(self):
        self.tools = {"ECHO": StubTool()}
        self.executed = []

    def get_tool(self, name):
        return self.tools.get(name)

    def get_tools_prompt(self):
        return ""

    async def execute_tool(self, tool_call):
        self.executed.append((tool_call.name, dict(tool_call.params)))
        return SimpleNamespace(status="success",
                               message=f"echoresult {tool_call.params['id']}")


def _text_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None),
            finish_reason="stop")],
        usage=None)


def _tool_response(call_id):
    tc = _StreamedToolCall(f"tc-{call_id}", "ECHO",
                           json.dumps({"id": call_id}))
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=[tc]),
            finish_reason="tool_calls")],
        usage=None)


class InterleavingCompletions:
    """Scripted native tool-calling backend.

    Round 0 always returns an ECHO tool call carrying the turn's id; the
    compose round returns text that does NOT relay the tool result (so
    _fold_unspoken_results must prepend it). Turn A's compose round parks
    until turn B's round 0 has begun, forcing the two turns to interleave
    exactly where shared instance state used to be corrupted.
    """

    def __init__(self):
        self.b_round0_started = asyncio.Event()

    async def create(self, messages=None, tools=None, **kwargs):
        turn = next(m["content"] for m in reversed(messages)
                    if m["role"] == "user")
        if messages[-1]["role"] == "tool":
            # Compose round: interleave A's fold after B's reset point.
            if turn == "turn a":
                await self.b_round0_started.wait()
            return _text_response("Understood.")
        # Round 0: ask for the tool.
        if turn == "turn b":
            self.b_round0_started.set()
        return _tool_response("a" if turn == "turn a" else "b")


async def test_interleaved_turns_keep_their_own_spoken_results(config_factory):
    cfg = config_factory(llm_tool_calling="native",
                         grounding_retry_enabled="false")
    tool_manager = StubToolManager()
    engine = LLMEngine(cfg, tool_manager)
    completions = InterleavingCompletions()
    engine.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions))

    task_a = asyncio.create_task(
        engine.generate_response([{"role": "user", "content": "turn a"}]))
    # Let A run its tool round (its speak_result message is now pending),
    # then start B, whose turn begins while A is still composing.
    while not any(p.get("id") == "a" for _, p in tool_manager.executed):
        await asyncio.sleep(0.001)
    task_b = asyncio.create_task(
        engine.generate_response([{"role": "user", "content": "turn b"}]))

    response_b = await asyncio.wait_for(task_b, timeout=5)
    response_a = await asyncio.wait_for(task_a, timeout=5)

    # Each turn folded exactly ITS OWN unrelayed tool message: shared
    # instance bookkeeping would have dropped A's ("echoresult a") when B
    # started, and could leak one turn's result into the other's reply.
    assert response_a == "echoresult a Understood."
    assert response_b == "echoresult b Understood."


async def test_turn_context_defaults():
    ctx = TurnContext()
    assert ctx.spoken_results == []
    assert ctx.tool_calls == 0
    # Contexts are independent (no shared default list).
    ctx.spoken_results.append(("ECHO", "hi"))
    assert TurnContext().spoken_results == []
