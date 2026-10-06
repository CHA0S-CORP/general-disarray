"""Component tests for the LangGraph agentic engine (LLM_BACKEND=langgraph).

The agent runs against the mock vLLM server over real HTTP via ChatOpenAI.
The mock scripts native tool_calls rounds (see mock_vllm) so the full ReAct
round trip — model asks for a tool, the REAL tool manager executes it, the
result is fed back, the model answers — is exercised deterministically.
"""
import pytest
import pytest_asyncio

from mock_vllm import ECHO_PHRASE

pytestmark = pytest.mark.component

langchain_engine = pytest.importorskip(
    "langchain_engine", reason="langchain deps not installed")
from langchain_engine import LangChainEngine  # noqa: E402


def _make_engine(assistant, config_factory, vllm_url, speaches_url, **overrides):
    cfg = config_factory(
        speaches_api_url=speaches_url,
        llm_base_url=f"{vllm_url}/v1",
        llm_model="mock-model",
        llm_backend="langgraph",
        **overrides,
    )
    return LangChainEngine(cfg, assistant.tool_manager)


@pytest_asyncio.fixture
async def native_engine(assistant, config_factory, vllm_url, speaches_url):
    eng = _make_engine(assistant, config_factory, vllm_url, speaches_url,
                       llm_tool_calling="native")
    await eng.start()
    yield eng
    await eng.stop()


@pytest_asyncio.fixture
async def text_engine(assistant, config_factory, vllm_url, speaches_url):
    eng = _make_engine(assistant, config_factory, vllm_url, speaches_url,
                       llm_tool_calling="text")
    await eng.start()
    yield eng
    await eng.stop()


async def test_plain_reply_passes_through(native_engine):
    reply = await native_engine.generate_response(
        [{"role": "user", "content": "hello there"}])
    assert reply == "Sure, I can help with that."


async def test_native_tool_round_trip(native_engine):
    """Model asks for SIMON_SAYS, the real tool runs, result feeds back."""
    import mock_vllm
    before = len(mock_vllm.REQUESTS)
    reply = await native_engine.generate_response(
        [{"role": "user", "content": "please simon says something"}])

    assert ECHO_PHRASE in reply
    assert "[TOOL" not in reply
    # The final model call must carry the executed tool result back as a
    # role=tool message (don't assert exact request counts: the HTTP client
    # may retry against the threaded mock server, duplicating entries).
    rounds = mock_vllm.REQUESTS[before:]
    assert rounds and all(r.get("tools") for r in rounds)
    tool_msgs = [m for m in rounds[-1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and ECHO_PHRASE in tool_msgs[-1]["content"]


async def test_tool_round_limit_yields_apology(assistant, config_factory,
                                               vllm_url, speaches_url):
    """A model that demands tools forever hits the recursion budget."""
    eng = _make_engine(assistant, config_factory, vllm_url, speaches_url,
                       llm_tool_calling="native", llm_max_tool_rounds="1")
    await eng.start()
    try:
        reply = await eng.generate_response(
            [{"role": "user", "content": "loop forever please simon"}])
    finally:
        await eng.stop()
    assert "too many steps" in reply.lower()


async def test_text_mode_marker_fallback(text_engine):
    """Unbound agent + [TOOL:...] markers: text-mode tools keep working."""
    import mock_vllm
    before = len(mock_vllm.REQUESTS)
    reply = await text_engine.generate_response(
        [{"role": "user", "content": "please simon says something"}])
    assert ECHO_PHRASE in reply
    assert "[TOOL" not in reply
    # Text mode must not bind tools on the request.
    assert not mock_vllm.REQUESTS[before].get("tools")


async def test_system_prompt_context_blocks(assistant, config_factory,
                                            vllm_url, speaches_url):
    """Caller memory + rolling summary reach the agent's system prompt."""
    import mock_vllm
    native_engine = _make_engine(assistant, config_factory, vllm_url,
                                 speaches_url, llm_tool_calling="native",
                                 llm_split_system_prompt="true")
    await native_engine.start()
    try:
        await _context_turn(native_engine)
    finally:
        await native_engine.stop()
    msgs = mock_vllm.REQUESTS[-1]["messages"]
    # Static prefix first, per-turn context second (prefix-cache layout).
    assert [m["role"] for m in msgs[:3]] == ["system", "system", "user"]
    assert msgs[0]["content"] == native_engine.config.system_prompt
    system = msgs[1]["content"]
    assert "Caller: 1001" in system
    assert "Name is Bob" in system
    assert "Bob asked about the weather earlier." in system


async def test_system_prompt_unsplit_by_default(native_engine):
    """Default (split off): one system message carrying everything."""
    import mock_vllm
    await _context_turn(native_engine)
    msgs = mock_vllm.REQUESTS[-1]["messages"]
    assert [m["role"] for m in msgs[:2]] == ["system", "user"]
    assert "Name is Bob" in msgs[0]["content"]


async def _context_turn(engine):
    await engine.generate_response(
        [{"role": "user", "content": "hello there"}],
        {
            "remote_uri": "sip:1001@pbx.lan",
            "duration": 42.0,
            "caller_memory": "- Name is Bob",
            "conversation_summary": "Bob asked about the weather earlier.",
        },
    )


async def test_thinking_switch_reaches_body(assistant, config_factory,
                                            speaches_url, vllm_url):
    """LLM_ENABLE_THINKING is forwarded through ChatOpenAI's extra_body."""
    import mock_vllm
    from langchain_engine import LangChainEngine
    cfg = config_factory(
        speaches_api_url=speaches_url,
        llm_base_url=f"{vllm_url}/v1",
        llm_model="mock-model",
        llm_backend="langgraph",
        llm_tool_calling="native",
        llm_enable_thinking="false",
    )
    eng = LangChainEngine(cfg, assistant.tool_manager)
    await eng.start()
    try:
        await eng.generate_response([{"role": "user", "content": "hello there"}])
    finally:
        await eng.stop()
    assert mock_vllm.REQUESTS[-1]["chat_template_kwargs"] == {"enable_thinking": False}


async def test_backend_error_falls_back_to_phrase(assistant, config_factory,
                                                  speaches_url):
    """Dead backend -> spoken error phrase, never an exception."""
    cfg = config_factory(
        speaches_api_url=speaches_url,
        llm_base_url="http://127.0.0.1:9/v1",  # nothing listens here
        llm_model="mock-model",
        llm_backend="langgraph",
        llm_tool_calling="native",
    )
    eng = LangChainEngine(cfg, assistant.tool_manager)
    await eng.start()
    try:
        reply = await eng.generate_response(
            [{"role": "user", "content": "hello there"}])
    finally:
        await eng.stop()
    assert reply in cfg.phrases.errors


# --- factory gating -----------------------------------------------------------

def test_factory_creates_langchain_engine(assistant, config_factory):
    from llm_engine import create_llm_engine
    cfg = config_factory(llm_backend="langgraph")
    eng = create_llm_engine(cfg, assistant.tool_manager)
    assert isinstance(eng, LangChainEngine)


def test_factory_falls_back_when_deps_missing(assistant, config_factory,
                                              monkeypatch):
    import llm_engine as llm_engine_mod
    monkeypatch.setattr(langchain_engine, "LANGCHAIN_AVAILABLE", False)
    cfg = config_factory(llm_backend="langgraph")
    eng = llm_engine_mod.create_llm_engine(cfg, assistant.tool_manager)
    assert type(eng) is llm_engine_mod.LLMEngine

async def test_speak_result_content_reaches_caller(native_engine):
    """A speak_result tool's message (the joke) must be spoken even when the
    model's final answer only comments on it (the JOKE-tool bug)."""
    import mock_vllm
    before = len(mock_vllm.REQUESTS)
    reply = await native_engine.generate_response(
        [{"role": "user", "content": "tell me a joke"}])

    rounds = mock_vllm.REQUESTS[before:]
    tool_msgs = [m for m in rounds[-1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs, "JOKE tool round never happened"
    joke = tool_msgs[-1]["content"]

    # The actual joke is prepended before the model's commentary.
    assert joke in reply
    assert "Hope that made you smile" in reply
    assert reply.index(joke) < reply.index("Hope that made you smile")


# --- grounding retry (never-guess enforcement) --------------------------------

async def test_grounding_retry_forces_tool_on_fabrication(native_engine):
    """A live-data question answered with no tool call triggers exactly one
    forced retry, and the real tool output reaches the caller."""
    import re
    import mock_vllm
    before = len(mock_vllm.REQUESTS)
    reply = await native_engine.generate_response(
        [{"role": "user", "content": "what time is it"}])

    rounds = mock_vllm.REQUESTS[before:]
    forced = [r for r in rounds if r.get("tool_choice") == "required"]
    assert len(forced) == 1, "expected exactly one forced retry request"
    # The real DATETIME output was folded into the reply.
    assert re.search(r"\d{1,2}:\d{2} (AM|PM)", reply), reply
    assert "three o'clock" not in reply.lower() or re.search(r"\d{1,2}:\d{2}", reply)


async def test_grounding_retry_skips_casual_chat(native_engine):
    import mock_vllm
    before = len(mock_vllm.REQUESTS)
    reply = await native_engine.generate_response(
        [{"role": "user", "content": "hello there"}])
    assert reply == "Sure, I can help with that."
    assert not any(r.get("tool_choice") for r in mock_vllm.REQUESTS[before:])


async def test_grounding_retry_disabled_by_config(assistant, config_factory,
                                                  vllm_url, speaches_url):
    import mock_vllm
    eng = _make_engine(assistant, config_factory, vllm_url, speaches_url,
                       llm_tool_calling="native",
                       GROUNDING_RETRY_ENABLED="false")
    await eng.start()
    try:
        before = len(mock_vllm.REQUESTS)
        reply = await eng.generate_response(
            [{"role": "user", "content": "what time is it"}])
    finally:
        await eng.stop()
    assert not any(r.get("tool_choice") for r in mock_vllm.REQUESTS[before:])
    assert "three o'clock" in reply.lower()


async def test_grounding_retry_on_promised_action(native_engine):
    """'Let me check, one moment' with no tool call also triggers the retry."""
    import mock_vllm
    before = len(mock_vllm.REQUESTS)
    reply = await native_engine.generate_response(
        [{"role": "user", "content": "how's the gpu doing"}])
    forced = [r for r in mock_vllm.REQUESTS[before:]
              if r.get("tool_choice") == "required"]
    assert len(forced) == 1
    # DATETIME (the mock's forced tool) output reached the reply either via
    # compose or the fold safety net.
    import re
    assert re.search(r"\d{1,2}:\d{2} (AM|PM)", reply), reply


async def test_grounding_retry_nudge_fallback_on_400(native_engine):
    """A backend rejecting tool_choice=required falls back to a nudge re-run."""
    import mock_vllm
    mock_vllm.REJECT_TOOL_CHOICE = True
    try:
        before = len(mock_vllm.REQUESTS)
        reply = await native_engine.generate_response(
            [{"role": "user", "content": "what time is it"}])
    finally:
        mock_vllm.REJECT_TOOL_CHOICE = False

    rounds = mock_vllm.REQUESTS[before:]
    # A nudge re-run carried the grounding instruction in the user turn
    # (never as a mid-conversation system message).
    nudged = [r for r in rounds if any(
        mock_vllm.NUDGE_MARKER in str(m.get("content") or "")
        for m in r.get("messages", []) if m.get("role") == "user")]
    assert not any(
        mock_vllm.NUDGE_MARKER in str(m.get("content") or "")
        for r in rounds for m in r.get("messages", [])
        if m.get("role") == "system")
    assert nudged, "nudge fallback request never sent"
    import re
    assert re.search(r"\d{1,2}:\d{2} (AM|PM)", reply), reply


# --- agent wall clock pauses during keypad entry ---------------------------------

def test_invoke_with_budget_pauses_while_dtmf_collecting():
    """The VERIFY tool's DTMF wait is the caller's time: with the clock paused a
    turn longer than LLM_AGENT_TIMEOUT_S still completes; unpaused it times out."""
    import asyncio
    from langchain_engine import LangChainEngine

    class _Self:
        paused = True

        def _clock_paused(self):
            return self.paused

    async def slow():
        await asyncio.sleep(0.6)
        return "done"

    async def go(paused):
        fake = _Self()
        fake.paused = paused
        return await LangChainEngine._invoke_with_budget(fake, slow(), 0.3)

    assert asyncio.run(go(True)) == "done"
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go(False))


# --- grounding retry after tools already ran (stubbed graph) -------------------

class _StubTM:
    def __init__(self):
        from types import SimpleNamespace
        self.tools = {n: SimpleNamespace(name=n, enabled=True, parameters={},
                                         speak_result=False, json_schema=None,
                                         description=n)
                      for n in ("WEATHER", "SET_TIMER")}
        self.executed = []

    def get_tool(self, name):
        return self.tools.get(name.upper())

    def get_tools_prompt(self):
        return ""

    async def execute_tool(self, tool_call):
        from types import SimpleNamespace
        self.executed.append(tool_call.name)
        return SimpleNamespace(status="success", message="ok")


class _StubAgent:
    def __init__(self, tool_name):
        self.tool_name = tool_name

    async def ainvoke(self, state, config=None):
        from langchain_core.messages import AIMessage, ToolMessage
        return {"messages": list(state["messages"]) + [
            AIMessage(content="", tool_calls=[
                {"name": self.tool_name, "args": {}, "id": "t1"}]),
            ToolMessage(content="ok", tool_call_id="t1"),
            AIMessage(content="Done. Let me check on that for you."),
        ]}


def _stub_engine(config_factory, tool_name):
    cfg = config_factory(llm_backend="langgraph", llm_tool_calling="native")
    eng = LangChainEngine(cfg, _StubTM())
    eng.client = object()
    eng._agent = _StubAgent(tool_name)
    eng._lc_tools = ["bound"]
    calls = []

    async def fake_retry(messages, category, ctx):
        calls.append((messages, category))
        return "Retried answer."

    eng._grounding_retry = fake_retry
    return eng, calls


async def test_trailing_promise_after_side_effect_tool_skips_retry(config_factory):
    """A forced retry after SET_TIMER already ran could set it twice."""
    eng, calls = _stub_engine(config_factory, "SET_TIMER")
    reply = await eng.generate_response([{"role": "user", "content": "timer"}])
    assert calls == []
    assert reply == "Done. Let me check on that for you."


async def test_trailing_promise_retry_sees_agent_tool_results(config_factory):
    """After read-only tools, the retry runs on the full graph state (the
    agent's tool calls + results), minus the dangling promise message."""
    from langchain_core.messages import AIMessage, ToolMessage
    eng, calls = _stub_engine(config_factory, "WEATHER")
    reply = await eng.generate_response([{"role": "user", "content": "hi"}])
    assert reply == "Retried answer."
    (messages, category), = calls
    assert category == "PROMISED_ACTION"
    assert isinstance(messages[-1], ToolMessage)
    assert any(isinstance(m, AIMessage) and m.tool_calls for m in messages)


async def test_unrelated_400_does_not_disable_tool_choice(config_factory):
    """Only a 400 naming tool_choice marks it unsupported."""
    from langchain_core.messages import HumanMessage

    class _Bound:
        async def ainvoke(self, messages):
            raise Exception("Error code: 400 - maximum context length exceeded")

    class _Chat:
        def bind_tools(self, tools, tool_choice=None):
            return _Bound()

    cfg = config_factory(llm_backend="langgraph", llm_tool_calling="native")
    eng = LangChainEngine(cfg, _StubTM())
    eng._chat = _Chat()
    eng._lc_tools = ["bound"]
    from llm_engine import TurnContext
    text = await eng._grounding_retry(
        [HumanMessage(content="what time is it")], "DATETIME", TurnContext())
    assert text is None
    assert eng._tool_choice_supported is True


def test_lc_nudge_rides_in_last_user_turn():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    import grounding
    msgs = [SystemMessage(content="sys"), HumanMessage(content="what time"),
            AIMessage(content="hmm")]
    out = LangChainEngine._with_lc_nudge(msgs)
    assert [type(m) for m in out] == [type(m) for m in msgs]
    assert out[1].content.startswith("what time")
    assert grounding.NUDGE in out[1].content
    assert msgs[1].content == "what time"  # input untouched


def test_lc_tools_use_json_schema_when_present(config_factory):
    """MCP-style tools carrying a full JSON schema are bound with it
    verbatim (nested objects survive), like _build_native_tools."""
    from types import SimpleNamespace
    from langchain_core.utils.function_calling import convert_to_openai_tool
    schema = {"type": "object",
              "properties": {"query": {"type": "string"},
                             "filters": {"type": "object", "properties": {
                                 "tags": {"type": "array",
                                          "items": {"type": "string"}}}}},
              "required": ["query"]}
    tm = _StubTM()
    tm.tools = {"MCP_SEARCH": SimpleNamespace(
        name="MCP_SEARCH", enabled=True, parameters={}, json_schema=schema,
        description="search")}
    cfg = config_factory(llm_backend="langgraph", llm_tool_calling="native")
    eng = LangChainEngine(cfg, tm)
    (tool,) = eng._build_lc_tools()
    params = convert_to_openai_tool(tool)["function"]["parameters"]
    assert params["properties"]["filters"]["properties"]["tags"]["type"] == "array"
    assert params["required"] == ["query"]
