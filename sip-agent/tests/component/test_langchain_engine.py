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


async def test_system_prompt_context_blocks(native_engine):
    """Caller memory + rolling summary reach the agent's system prompt."""
    import mock_vllm
    await native_engine.generate_response(
        [{"role": "user", "content": "hello there"}],
        {
            "remote_uri": "sip:1001@pbx.lan",
            "duration": 42.0,
            "caller_memory": "- Name is Bob",
            "conversation_summary": "Bob asked about the weather earlier.",
        },
    )
    system = mock_vllm.REQUESTS[-1]["messages"][0]["content"]
    assert "Caller: 1001" in system
    assert "Name is Bob" in system
    assert "Bob asked about the weather earlier." in system


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
    # A nudge re-run carried the grounding instruction as a system message.
    nudged = [r for r in rounds if any(
        mock_vllm.NUDGE_MARKER in str(m.get("content") or "")
        for m in r.get("messages", []) if m.get("role") == "system")]
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
