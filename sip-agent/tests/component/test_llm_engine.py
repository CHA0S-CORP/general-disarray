"""Component tests for the LLM engine against the mock vLLM server.

Exercises the full text-based tool-call round trip: the OpenAI client hits the
mock over HTTP, the response's [TOOL:...] markers are parsed, dispatched to the
*real* tool manager, executed, stripped from the spoken text, and informational
tool output (SIMON_SAYS/CALC) is appended to the reply.
"""
import pytest
import pytest_asyncio

from llm_engine import LLMEngine
from mock_vllm import ECHO_PHRASE

pytestmark = pytest.mark.component


@pytest_asyncio.fixture
async def engine(assistant):
    # assistant.config already points llm_base_url at the mock vLLM.
    eng = LLMEngine(assistant.config, assistant.tool_manager)
    await eng.start()
    yield eng
    await eng.stop()


async def test_plain_reply_passes_through(engine):
    reply = await engine.generate_response([{"role": "user", "content": "hello there"}])
    assert reply == "Sure, I can help with that."
    assert "[TOOL" not in reply


async def test_tool_call_executes_and_strips_marker(engine):
    # Mock returns "[TOOL:SIMON_SAYS:text=the eagle has landed]".
    reply = await engine.generate_response(
        [{"role": "user", "content": "please simon says something"}]
    )
    assert "[TOOL" not in reply           # marker stripped from spoken text
    assert ECHO_PHRASE in reply           # SIMON_SAYS result appended


async def test_calc_tool_call_round_trip(engine):
    reply = await engine.generate_response(
        [{"role": "user", "content": "do some calc for me"}]
    )
    assert "[TOOL" not in reply
    assert "4" in reply  # CALC expression 2+2 -> result appended to reply
