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


# --- native (OpenAI tools API) mode ------------------------------------------

class _StubMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.finish_reason = "stop"


class _StubToolCall:
    def __init__(self, call_id, name, arguments):
        from types import SimpleNamespace
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)

    def model_dump(self):
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name,
                             "arguments": self.function.arguments}}


class _StubClient:
    """Scripted chat.completions client: first asks for CALC, then answers."""

    def __init__(self):
        from types import SimpleNamespace
        self.requests = []

        async def create(**kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                msg = _StubMessage(tool_calls=[
                    _StubToolCall("tc-1", "CALC", '{"expression": "2+2"}')])
            else:
                # Final answer references the tool result fed back to it.
                tool_msgs = [m for m in kwargs["messages"] if m.get("role") == "tool"]
                msg = _StubMessage(content=f"The answer is in: {tool_msgs[0]['content']}")
            choice = SimpleNamespace(message=msg, finish_reason="stop")
            return SimpleNamespace(choices=[choice], usage=None)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


async def test_native_tool_calling_round_trip(assistant, config_factory):
    cfg = config_factory(llm_tool_calling="native")
    eng = LLMEngine(cfg, assistant.tool_manager)
    stub = _StubClient()
    eng.client = stub

    reply = await eng.generate_response([{"role": "user", "content": "what is 2+2"}])

    # Round 1 carried the tool schemas; the CALC result was executed for real
    # and fed back; round 2 produced the spoken reply.
    assert len(stub.requests) == 2
    assert any(t["function"]["name"] == "CALC" for t in stub.requests[0]["tools"])
    assert "4" in reply
    # Native mode must not ALSO inject the [TOOL:...] marker instructions.
    system_prompt = stub.requests[0]["messages"][0]["content"]
    assert "[TOOL:" not in system_prompt


async def test_native_loop_respects_tool_round_budget(assistant, config_factory):
    """LLM_MAX_TOOL_ROUNDS bounds the native loop (was hardcoded to 3)."""
    from types import SimpleNamespace

    cfg = config_factory(llm_tool_calling="native", llm_max_tool_rounds="1")
    eng = LLMEngine(cfg, assistant.tool_manager)

    requests = []

    async def always_tools(**kwargs):
        requests.append(kwargs)
        msg = _StubMessage(tool_calls=[
            _StubToolCall(f"tc-{len(requests)}", "CALC", '{"expression": "2+2"}')])
        choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
        return SimpleNamespace(choices=[choice], usage=None)

    eng.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=always_tools)))

    reply = await eng.generate_response([{"role": "user", "content": "loop"}])
    assert len(requests) == 1  # budget of one round, not the old default
    assert "too many steps" in reply.lower()


# --- call context + sampling params ------------------------------------------

async def test_call_context_reaches_system_prompt(engine):
    import mock_vllm
    await engine.generate_response(
        [{"role": "user", "content": "hello there"}],
        {"remote_uri": "sip:1001@pbx.lan", "duration": 42.0},
    )
    system = mock_vllm.REQUESTS[-1]["messages"][0]["content"]
    assert "Caller: 1001" in system          # user part extracted from the URI
    assert "42 seconds" in system
    assert "Caller: unknown" not in system


async def test_frequency_penalty_omitted_by_default(engine):
    import mock_vllm
    await engine.generate_response([{"role": "user", "content": "hello there"}])
    assert "frequency_penalty" not in mock_vllm.REQUESTS[-1]


async def test_frequency_penalty_sent_when_configured(assistant, config_factory,
                                                      speaches_url, vllm_url):
    import mock_vllm
    from llm_engine import LLMEngine
    cfg = config_factory(
        speaches_api_url=speaches_url,
        llm_base_url=f"{vllm_url}/v1",
        llm_model="mock-model",
        llm_frequency_penalty="0.5",
    )
    eng = LLMEngine(cfg, assistant.tool_manager)
    await eng.start()
    try:
        await eng.generate_response([{"role": "user", "content": "hello there"}])
    finally:
        await eng.stop()
    assert mock_vllm.REQUESTS[-1]["frequency_penalty"] == 0.5


async def test_assistant_wires_caller_context_end_to_end(config_factory,
                                                         speaches_url, vllm_url):
    """Kills the caller_id/remote_uri key-mismatch regression class."""
    import mock_vllm
    from types import SimpleNamespace
    from main import SIPAIAssistant

    cfg = config_factory(
        speaches_api_url=speaches_url,
        llm_base_url=f"{vllm_url}/v1",
        llm_model="mock-model",
    )
    a = SIPAIAssistant(cfg)
    await a.llm_engine.start()
    try:
        call = SimpleNamespace(is_active=True, remote_uri="sip:1001@host", media_ready=False)
        session = a._begin_session(call, "inbound", "sip:1001@host")
        await a._generate_response(session, "hello there")
        system = mock_vllm.REQUESTS[-1]["messages"][0]["content"]
        assert "Caller: 1001" in system
        await a._teardown_session()
    finally:
        await a.llm_engine.stop()


# --- reformat_for_speech -----------------------------------------------------

async def test_reformat_rewrites_via_llm(engine):
    from mock_vllm import SPOKEN_REWRITE
    raw = "deploy failed at 2026-07-08T17:03Z p99=340ms"
    assert await engine.reformat_for_speech(raw, timeout_s=10.0) == SPOKEN_REWRITE


async def test_reformat_falls_back_on_client_error(engine):
    from types import SimpleNamespace

    async def boom(**kwargs):
        raise RuntimeError("backend down")

    engine.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
    raw = "deploy failed at noon"
    try:
        assert await engine.reformat_for_speech(raw, timeout_s=10.0) == raw
    finally:
        engine.client = None  # let the fixture's stop() run cleanly


async def test_reformat_falls_back_on_empty_content(engine):
    from types import SimpleNamespace

    async def empty(**kwargs):
        msg = SimpleNamespace(content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    engine.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=empty)))
    raw = "deploy failed at noon"
    try:
        assert await engine.reformat_for_speech(raw, timeout_s=10.0) == raw
    finally:
        engine.client = None  # let the fixture's stop() run cleanly


async def test_reformat_uses_expanded_token_budget(engine):
    from types import SimpleNamespace
    seen = {}

    async def capture(**kwargs):
        seen.update(kwargs)
        msg = SimpleNamespace(content="A spoken version.")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    engine.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=capture)))
    try:
        result = await engine.reformat_for_speech("raw text", timeout_s=10.0)
    finally:
        engine.client = None  # let the fixture's stop() run cleanly
    assert result == "A spoken version."
    # Reasoning models need far more than the conversation budget, and must
    # run at the model's default sampling (see reformat_for_speech).
    assert seen["max_tokens"] >= 2048
    assert "temperature" not in seen
    assert "top_p" not in seen


async def test_reformat_falls_back_without_client(engine):
    engine.client = None
    raw = "deploy failed at noon"
    assert await engine.reformat_for_speech(raw, timeout_s=10.0) == raw


async def test_reformat_falls_back_on_timeout(engine):
    raw = "deploy failed at noon"
    assert await engine.reformat_for_speech(raw, timeout_s=0.000001) == raw


async def test_reformat_passes_empty_through(engine):
    assert await engine.reformat_for_speech("", timeout_s=10.0) == ""
