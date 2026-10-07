"""Unit regressions for LLMEngine: marker parsing/scrubbing, schema-aware
param coercion, the text-mode grounding retry, the native grounding force
(round budget, tool_choice caching, nudge placement) and the thinking switch
on utility completions. Stub tool manager + stub OpenAI client — no server.
"""
import logging
from types import SimpleNamespace

import pytest

import grounding
from llm_engine import LLMEngine, TurnContext, tool_choice_rejected
from sentence_stream import SentenceAssembler

pytestmark = pytest.mark.unit


class StubTool:
    def __init__(self, name, parameters=None, speak_result=False,
                 json_schema=None):
        self.name = name
        self.description = f"{name} tool"
        self.enabled = True
        self.parameters = parameters or {}
        self.speak_result = speak_result
        self.json_schema = json_schema


class StubToolManager:
    def __init__(self, *tools):
        self.tools = {t.name: t for t in tools}
        self.executed = []

    def get_tool(self, name):
        return self.tools.get(name.upper())

    def get_tools_prompt(self):
        return "TOOLS: [TOOL:NAME]"

    async def execute_tool(self, tool_call):
        self.executed.append((tool_call.name, dict(tool_call.params)))
        return SimpleNamespace(status="success",
                               message=f"{tool_call.name} result")


def _default_tools():
    return StubToolManager(
        StubTool("WEATHER"),
        StubTool("DATETIME"),
        StubTool("SET_TIMER", {"duration": {"type": "integer"},
                               "message": {"type": "string"}}),
        StubTool("SIMON_SAYS", {"text": {"type": "string"}}),
        StubTool("CALC", {"expression": {"type": "string"}}),
        StubTool("CONTAINER_CTL", {"confirm": {"type": "boolean"},
                                   "ratio": {"type": "number"}}),
        StubTool("MCP_THING", json_schema={
            "type": "object",
            "properties": {"count": {"type": ["integer", "null"]},
                           "label": {"type": "string"}}}),
    )


def _text(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None),
            finish_reason=finish_reason)],
        usage=None)


class ScriptedClient:
    """chat.completions.create returning scripted items in order (an
    Exception item is raised)."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.script.pop(0) if self.script else _text("fallthrough")
        if isinstance(item, Exception):
            raise item
        return item


def _engine(config_factory, tool_manager=None, client=None, **cfg):
    eng = LLMEngine(config_factory(**cfg), tool_manager or _default_tools())
    eng.client = client
    return eng


# --- marker parsing / scrubbing ----------------------------------------------

@pytest.mark.parametrize("reply,expected_tools,expected_text", [
    # Truncated at max_tokens: nothing runs, nothing is spoken.
    ("Okay. [TOOL:SET_TIMER:duration=3", [], "Okay."),
    # Whitespace / case variants are unambiguous -> executed.
    ("[TOOL: WEATHER] Here you go.", ["WEATHER"], "Here you go."),
    ("Sure. [tool:weather]", ["WEATHER"], "Sure."),
    # Invalid name: scrubbed, not executed.
    ("Checking. [TOOL:GET-WEATHER]", [], "Checking."),
    # Unterminated marker must not swallow the valid one after it.
    ("a [TOOL:SET_TIMER:duration=3 [TOOL:WEATHER] b", ["WEATHER"], "a b"),
    # Ordinary brackets are untouched.
    ("Sure [Total] thing", [], "Sure [Total] thing"),
])
async def test_markers_parse_and_never_leak(config_factory, reply,
                                            expected_tools, expected_text):
    tm = _default_tools()
    eng = _engine(config_factory, tm)
    text, _ = await eng._process_tool_calls(reply)
    assert [name for name, _ in tm.executed] == expected_tools
    assert text == expected_text
    assert "[TOOL" not in text.upper()


async def test_unparsed_marker_is_logged(config_factory, caplog):
    eng = _engine(config_factory)
    with caplog.at_level(logging.WARNING, logger="llm_engine"):
        await eng._process_tool_calls("Okay. [TOOL:SET_TIMER:duration=3")
    events = [r for r in caplog.records
              if getattr(r, "event_type", None) == "marker_unparsed"]
    assert events and "SET_TIMER" in events[0].event_data["fragment"]


async def test_finalize_stream_scrubs_truncated_marker(config_factory):
    tm = _default_tools()
    eng = _engine(config_factory, tm, grounding_retry_enabled="false")
    assembler = SentenceAssembler()
    emitted = []
    for s in assembler.feed("This first sentence is long enough to emit. "
                            "Then [TOOL:SET_TIMER:duration=3"):
        emitted.append(s)
    events = [e async for e in eng._finalize_stream(
        assembler, emitted, "length", "set a timer", TurnContext())]
    final = events[-1]["text"]
    assert "[TOOL" not in final
    assert all("[TOOL" not in e["text"] for e in events)
    assert tm.executed == []


def test_streamed_sentence_variant_markers_scrubbed(config_factory):
    """The assembler only holds the exact '[TOOL:' prefix; lowercase /
    spaced variants that slip into a sentence are scrubbed before TTS."""
    eng = _engine(config_factory)
    out = eng._scrub_sentences(["Sure thing. [tool:weather] okay.",
                                "[TOOL: DATETIME]", "Plain sentence."])
    assert out == ["Sure thing. okay.", "Plain sentence."]


# --- schema-aware coercion ----------------------------------------------------

@pytest.mark.parametrize("marker,tool,params", [
    ("[TOOL:SIMON_SAYS:text=yes]", "SIMON_SAYS", {"text": "yes"}),
    ("[TOOL:CALC:expression=42]", "CALC", {"expression": "42"}),
    ("[TOOL:SET_TIMER:duration=300,message=no]", "SET_TIMER",
     {"duration": 300, "message": "no"}),
    ("[TOOL:CONTAINER_CTL:confirm=yes,ratio=2.5]", "CONTAINER_CTL",
     {"confirm": True, "ratio": 2.5}),
    # Full JSON schema (MCP) with a nullable integer.
    ("[TOOL:MCP_THING:count=7,label=8]", "MCP_THING",
     {"count": 7, "label": "8"}),
    # Undeclared params keep the legacy heuristic.
    ("[TOOL:WEATHER:days=3]", "WEATHER", {"days": 3}),
])
async def test_marker_params_follow_declared_schema(config_factory, marker,
                                                    tool, params):
    tm = _default_tools()
    eng = _engine(config_factory, tm)
    await eng._process_tool_calls(marker)
    assert tm.executed == [(tool, params)]


# --- text-mode grounding retry ------------------------------------------------

async def test_text_retry_judges_tool_use_by_delta(config_factory):
    """The original reply already ran a (read-only) tool; a retry that runs
    NONE must not be adopted just because ctx.tool_calls was already > 0."""
    client = ScriptedClient(
        _text("[TOOL:DATETIME] Let me check on that for you."),
        _text("I really couldn't say."))
    eng = _engine(config_factory, client=client)
    reply = await eng.generate_response(
        [{"role": "user", "content": "hello"}])
    assert len(client.requests) == 2  # the trailing promise did retry
    assert reply == "Let me check on that for you."


async def test_text_retry_skipped_after_side_effecting_tool(config_factory):
    tm = _default_tools()
    client = ScriptedClient(
        _text("[TOOL:SET_TIMER:duration=60] Let me check on that."),
        _text("[TOOL:SET_TIMER:duration=60] Timer set."))
    eng = _engine(config_factory, tm, client=client)
    reply = await eng.generate_response(
        [{"role": "user", "content": "set a timer"}])
    assert len(client.requests) == 1  # no re-run
    assert [n for n, _ in tm.executed] == ["SET_TIMER"]  # ran exactly once
    assert reply == "Let me check on that."


async def test_text_retry_error_keeps_original(config_factory):
    client = ScriptedClient(
        _text("It is three o'clock."), RuntimeError("backend down"))
    eng = _engine(config_factory, client=client)
    reply = await eng.generate_response(
        [{"role": "user", "content": "what time is it"}])
    assert len(client.requests) == 2
    assert reply == "It is three o'clock."


async def test_text_retry_nudge_in_last_user_message(config_factory):
    client = ScriptedClient(_text("It is three o'clock."),
                            _text("[TOOL:DATETIME]"))
    eng = _engine(config_factory, client=client)
    history = [{"role": "user", "content": "what time is it"}]
    await eng.generate_response(history)
    retry_msgs = client.requests[1]["messages"]
    assert retry_msgs[-1]["role"] == "user"
    assert retry_msgs[-1]["content"].startswith("what time is it")
    assert grounding.NUDGE in retry_msgs[-1]["content"]
    assert not any(grounding.NUDGE in (m.get("content") or "")
                   for m in retry_msgs if m["role"] == "system")
    assert history == [{"role": "user", "content": "what time is it"}]


# --- native grounding force ---------------------------------------------------

async def test_native_no_forced_tool_with_zero_round_budget(config_factory):
    client = ScriptedClient(_text("It is three o'clock."))
    eng = _engine(config_factory, client=client, llm_tool_calling="native",
                  llm_max_tool_rounds="0")
    reply = await eng.generate_response(
        [{"role": "user", "content": "what time is it"}])
    assert reply == "It is three o'clock."
    assert len(client.requests) == 1
    assert not any(r.get("tool_choice") for r in client.requests)
    assert client.requests[0].get("tools") is None


class _BadRequest(Exception):
    pass


async def test_native_tool_choice_rejection_is_cached(config_factory):
    rejected = _BadRequest(
        "Error code: 400 - tool_choice 'required' is not supported")
    client = ScriptedClient(
        _text("It is three o'clock."), rejected, _text("Nudged answer."),
        _text("It is four o'clock."), _text("Nudged again."))
    eng = _engine(config_factory, client=client, llm_tool_calling="native")
    for _ in range(2):
        await eng.generate_response(
            [{"role": "user", "content": "what time is it"}])
    forced = [r for r in client.requests if r.get("tool_choice") == "required"]
    assert len(forced) == 1  # second turn skipped the doomed request
    assert eng._tool_choice_supported is False
    # The nudge re-run carried NUDGE in the user turn, not a system message.
    nudged = client.requests[2]["messages"]
    assert grounding.NUDGE in nudged[-1]["content"]
    assert nudged[-1]["role"] == "user"
    assert not any(m["role"] == "system" and grounding.NUDGE in m["content"]
                   for m in nudged)


async def test_native_unrelated_400_not_cached(config_factory):
    client = ScriptedClient(
        _text("It is three o'clock."),
        _BadRequest("Error code: 400 - maximum context length exceeded"))
    eng = _engine(config_factory, client=client, llm_tool_calling="native")
    reply = await eng.generate_response(
        [{"role": "user", "content": "what time is it"}])
    assert reply == "It is three o'clock."  # original kept
    assert eng._tool_choice_supported is True


def test_tool_choice_rejected_detector():
    assert tool_choice_rejected(_BadRequest("400 tool_choice unsupported"))
    assert tool_choice_rejected(Exception(
        'Error code: 400 - "auto" tool choice requires --enable-auto-tool-choice'))
    assert not tool_choice_rejected(_BadRequest("400 context too long"))
    assert not tool_choice_rejected(Exception("500 tool_choice exploded"))


# --- thinking switch on utility completions -----------------------------------

async def test_utility_completions_carry_thinking_switch(config_factory):
    client = ScriptedClient(_text("Rewritten."), _text("Summary."))
    eng = _engine(config_factory, client=client, llm_enable_thinking="false")
    assert await eng.reformat_for_speech("raw text", 5) == "Rewritten."
    assert await eng.summarize_text("sys", "text", 5) == "Summary."
    for req in client.requests:
        assert req["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}}


async def test_utility_completions_omit_thinking_switch_by_default(config_factory):
    client = ScriptedClient(_text("Rewritten."))
    eng = _engine(config_factory, client=client)
    await eng.reformat_for_speech("raw text", 5)
    assert "extra_body" not in client.requests[0]


# --- system prompt split gating -----------------------------------------------

def test_split_requires_native_tools(config_factory):
    text_eng = _engine(config_factory, client=object(),
                       llm_split_system_prompt="true")
    assert len(text_eng._build_system_messages({"remote_uri": "sip:1@x"})) == 1
    native_eng = _engine(config_factory, client=object(),
                         llm_split_system_prompt="true",
                         llm_tool_calling="native")
    assert len(native_eng._build_system_messages({"remote_uri": "sip:1@x"})) == 2
