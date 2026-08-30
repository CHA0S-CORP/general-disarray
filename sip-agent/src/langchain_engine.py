"""
LangChain / LangGraph Engine
============================
Agentic LLM engine (LLM_BACKEND=langgraph): a LangGraph ReAct loop over the
same OpenAI-compatible backend, letting the model chain several tool calls in
one turn (bounded by LLM_MAX_TOOL_ROUNDS and LLM_AGENT_TIMEOUT_S).

Subclasses LLMEngine so everything else — system-prompt assembly (caller
memory, rolling summary, knowledge context), the [TOOL:...] marker safety
net, reformat_for_speech/summarize_text utilities, fallback phrases — is
shared. With LLM_TOOL_CALLING=text the agent runs with no bound tools and
tools keep working through the marker parser (degradation path for models
without native tool-call support).

All optional imports are guarded: constructing this engine without the
langchain deps raises ImportError, which create_llm_engine catches to fall
back to the classic engine. Runtime failures during a call fall back to a
spoken error phrase — never a crash, never a mock response.
"""

import asyncio
import contextvars
import logging
import time
from typing import Any, Dict, List, Optional

try:
    from langchain_core.messages import (AIMessage, HumanMessage,
                                         SystemMessage, ToolMessage)
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from langgraph.errors import GraphRecursionError
    from langgraph.prebuilt import create_react_agent
    from pydantic import Field, create_model
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

import grounding
from config import Config
from llm_engine import LLMEngine, ToolCall, TurnContext
from logging_utils import log_event
from telemetry import create_span, Metrics

logger = logging.getLogger(__name__)

# The turn's TurnContext, visible to the LangChain tool closures. The bound
# StructuredTools are built once at start() and executed deep inside
# langgraph's ainvoke, so the per-turn context can't be threaded through
# their signatures; a contextvar propagates it along the ainvoke task tree
# instead (each concurrent turn sees only its own context).
_TURN_CTX: "contextvars.ContextVar[Optional[TurnContext]]" = (
    contextvars.ContextVar("langchain_turn_ctx", default=None))

_PARAM_TYPES = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


class LangChainEngine(LLMEngine):
    """LangGraph ReAct agent over the OpenAI-compatible backend."""

    def __init__(self, config: Config, tool_manager):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LLM_BACKEND=langgraph requires langchain-core, "
                "langchain-openai and langgraph")
        super().__init__(config, tool_manager)
        self._agent = None
        self._chat = None
        self._lc_tools: List[Any] = []
        # Whether the backend accepts tool_choice="required"; probed once,
        # cached so a vLLM that rejects it costs a single failed request.
        self._tool_choice_supported = True

    def _clock_paused(self) -> bool:
        """True while a tool is waiting on the caller (keypad entry): that wait
        is the caller's time, not the LLM's, so it isn't charged to the budget."""
        try:
            session = getattr(self.tool_manager.assistant, "session", None)
            return bool(getattr(session, "dtmf_collecting", False))
        except Exception:
            return False

    async def _invoke_with_budget(self, coro, timeout: float):
        """``asyncio.wait_for`` whose clock pauses while ``_clock_paused()``.

        Otherwise the VERIFY tool's DTMF wait (prompt + up to
        VERIFY_DTMF_TIMEOUT_S) alone could exhaust LLM_AGENT_TIMEOUT_S and a
        correct code would still end in the spoken error phrase.
        """
        task = asyncio.ensure_future(coro)
        loop = asyncio.get_event_loop()
        remaining = float(timeout)
        try:
            while True:
                tick = loop.time()
                done, _ = await asyncio.wait({task}, timeout=min(remaining, 0.25))
                if done:
                    return task.result()
                if not self._clock_paused():
                    remaining -= loop.time() - tick
                if remaining <= 0:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                    raise asyncio.TimeoutError()
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def start(self):
        # Keeps self.client (AsyncOpenAI) alive for the shared utility paths
        # (reformat_for_speech, summarize_text) and connectivity logging.
        await super().start()

        chat_kwargs: Dict[str, Any] = {
            "base_url": self.config.llm_base_url,
            "api_key": self.config.llm_api_key,
            "model": self.config.llm_model,
            "max_tokens": self.config.llm_max_tokens,
            "temperature": self.config.llm_temperature,
            "top_p": self.config.llm_top_p,
            "timeout": 60.0,
        }
        if self.config.llm_frequency_penalty:
            chat_kwargs["frequency_penalty"] = self.config.llm_frequency_penalty
        chat = ChatOpenAI(**chat_kwargs)

        # Tools are bound only in native mode; in text mode the agent is a
        # plain conversational graph and [TOOL:...] markers do the work.
        tools = (self._build_lc_tools()
                 if self.config.llm_tool_calling.lower() == "native" else [])
        self._agent = create_react_agent(chat, tools)
        # Kept for the grounding retry (one forced tool round outside the
        # graph — see _grounding_retry).
        self._chat = chat
        self._lc_tools = tools
        logger.info(
            f"LangGraph agent ready ({len(tools)} bound tools, "
            f"max {self.config.llm_max_tool_rounds} tool rounds)")

    def _build_lc_tools(self) -> List[Any]:
        """Wrap every enabled ToolManager tool as an async StructuredTool.

        Execution dispatches through tool_manager.execute_tool so behavior
        (CALLBACK caller defaulting, metrics, validation) matches the other
        engines exactly.
        """
        lc_tools = []
        for name, tool in self.tool_manager.tools.items():
            if not getattr(tool, "enabled", True):
                continue
            fields: Dict[str, Any] = {}
            for pname, spec in (getattr(tool, "parameters", {}) or {}).items():
                ptype = _PARAM_TYPES.get(spec.get("type", "string"), str)
                desc = spec.get("description", "")
                if spec.get("required"):
                    fields[pname] = (ptype, Field(description=desc))
                else:
                    fields[pname] = (Optional[ptype],
                                     Field(default=None, description=desc))
            args_schema = create_model(f"{name}_args", **fields)

            async def _run(_tool_name: str = name, **kwargs) -> str:
                params = {k: v for k, v in kwargs.items() if v is not None}
                try:
                    result = await self.tool_manager.execute_tool(
                        ToolCall(name=_tool_name, params=params, raw=""))
                    ctx = _TURN_CTX.get()
                    if ctx is not None:
                        self._collect_spoken_result(ctx, _tool_name, result)
                    return getattr(result, "message", "") or "Done."
                except Exception as e:
                    logger.error(f"Agent tool execution error ({_tool_name}): {e}")
                    return f"Tool {_tool_name} failed."

            lc_tools.append(StructuredTool.from_function(
                coroutine=_run,
                name=name,
                description=getattr(tool, "description", "") or name,
                args_schema=args_schema,
            ))
        return lc_tools

    @staticmethod
    def _to_lc_messages(history: List[Dict[str, str]]) -> List[Any]:
        messages = []
        for msg in history:
            content = msg.get("content") or ""
            if msg.get("role") == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages

    @staticmethod
    def _extract_text(message: Any) -> str:
        """Final-message content can be a string or a list of content blocks."""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return " ".join(p for p in parts if p)
        return ""

    def _streaming_supported(self) -> bool:
        # The LangGraph agent loop is out of scope for token streaming;
        # stream_response falls back to the default generate_response replay
        # path (sentence events from the completed agent turn).
        return False

    async def generate_response(
        self,
        conversation_history: List[Dict[str, str]],
        call_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """One agentic turn: system prompt + history through the ReAct graph."""
        # No client (e.g. openai lib missing) -> the classic path's mock mode
        # is still the least-surprising behavior for dev setups.
        if self._agent is None or self.client is None:
            return await super().generate_response(
                conversation_history, call_context)

        # Per-turn tool bookkeeping; also published to the LC tool closures
        # via the contextvar for the duration of this turn (reset on exit so
        # a stale context can never leak into a later turn).
        ctx = TurnContext()
        ctx_token = _TURN_CTX.set(ctx)
        try:
            return await self._agent_generate(
                conversation_history, call_context, ctx)
        finally:
            _TURN_CTX.reset(ctx_token)

    async def _agent_generate(
        self,
        conversation_history: List[Dict[str, str]],
        call_context: Optional[Dict[str, Any]],
        ctx: TurnContext,
    ) -> str:
        """The agentic turn body (see generate_response, which owns ctx)."""
        messages: List[Any] = [
            SystemMessage(content=self._build_system_prompt(call_context))]
        # Same windowing rule as the classic engine (see _history_window):
        # don't re-truncate history the rolling summary has already sliced.
        messages.extend(self._to_lc_messages(
            self._history_window(conversation_history, call_context)))

        # Each tool round costs one agent step (model call) + one tool step.
        recursion_limit = 2 * self.config.llm_max_tool_rounds + 1

        with create_span("llm.generate_agent", {
            "llm.model": self.config.llm_model,
            "llm.recursion_limit": recursion_limit,
        }) as span:
            start_time = time.time()
            try:
                result = await self._invoke_with_budget(
                    self._agent.ainvoke(
                        {"messages": messages},
                        config={"recursion_limit": recursion_limit},
                    ),
                    self.config.llm_agent_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Agent turn exceeded {self.config.llm_agent_timeout_s}s")
                Metrics.record_llm_error(self.config.llm_model, "agent_timeout")
                return self._fallback_error()
            except GraphRecursionError:
                logger.warning("Agent hit the tool-round limit without answering")
                Metrics.record_llm_error(self.config.llm_model, "agent_recursion")
                return ("I'm sorry, that request took too many steps. "
                        "Could you try again?")
            except Exception as e:
                logger.error(f"Agent generation error: {e}")
                span.record_exception(e)
                Metrics.record_llm_error(self.config.llm_model, type(e).__name__)
                return self._fallback_error()

            latency_ms = (time.time() - start_time) * 1000
            Metrics.record_llm_latency(latency_ms, self.config.llm_model)
            span.set_attribute("llm.latency_ms", latency_ms)

            out_messages = result.get("messages", []) if isinstance(result, dict) else []
            tool_rounds = sum(
                1 for m in out_messages
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
            span.set_attribute("llm.tool_rounds", tool_rounds)
            if tool_rounds:
                log_event(logger, logging.INFO,
                          f"Agent turn used {tool_rounds} tool rounds",
                          event="agent_turn", tool_rounds=tool_rounds,
                          latency_ms=round(latency_ms))

            final = out_messages[-1] if out_messages else None
            content = self._extract_text(final) if final is not None else ""

            # "Never guess": a live-data question answered with zero tool
            # calls (or a reply that merely promises to check) gets ONE
            # forced-tool retry. Casual chat never triggers (grounding.py).
            #
            # A turn that DID call a tool still needs this when it signs off on
            # a promise: calling a tool is not the same as answering. The wrong
            # tool can be called, or the right one can come back empty, and the
            # model then defers — "Let me get that for you right away" — which
            # at end of turn is just dead air. Gating the retry on
            # tool_rounds == 0 let exactly that through.
            if self._lc_tools and self.config.grounding_retry_enabled:
                last_user = next(
                    (m.content for m in reversed(messages)
                     if isinstance(m, HumanMessage)), "")
                if tool_rounds == 0:
                    category = grounding.grounding_category(
                        str(last_user), content)
                elif grounding.trailing_promise(content):
                    category = "PROMISED_ACTION"
                else:
                    category = None
                if category and self._category_tools_available(category):
                    retried = await self._grounding_retry(messages, category, ctx)
                    if retried:
                        content = retried

            # gpt-oss and friends can legitimately return empty content. The
            # grounding retry may still have fetched real data (recorded on
            # ctx.spoken_results) — speak that instead of the error phrase.
            if not content or not content.strip():
                logger.warning("Agent returned empty content")
                Metrics.record_llm_error(self.config.llm_model, "empty_response")
                return (self._fold_unspoken_results("", ctx)
                        or self._fallback_error())

        # Marker safety net + informational-tool append, shared with the
        # classic engine (also covers text mode, where this IS the tool path).
        response_text = await self._apply_marker_tools(content.strip(), ctx)
        # Guarantee speak_result tool output (jokes, weather, search results)
        # actually reaches the caller even when the model only comments on it.
        return self._fold_unspoken_results(response_text, ctx)

    async def _grounding_retry(self, messages: List[Any], category: str,
                               ctx: TurnContext) -> Optional[str]:
        """One forced-tool round outside the graph, then a plain compose call.

        bind_tools(tool_choice="required") makes the model pick a tool; the
        calls are dispatched through tool_manager (populating
        ctx.spoken_results, so _fold_unspoken_results speaks the real data
        even if the compose step fails). A backend that rejects
        tool_choice="required" (HTTP 400) is remembered and the retry falls
        back to re-running the agent with an explicit grounding instruction.
        """
        start_time = time.time()
        outcome = "error"
        result_text: Optional[str] = None
        try:
            result_text, outcome = await asyncio.wait_for(
                self._grounding_retry_inner(messages, ctx),
                timeout=self.config.grounding_retry_timeout_s)
        except asyncio.TimeoutError:
            outcome = "timeout"
        except Exception as e:
            logger.error(f"Grounding retry error: {e}")
            outcome = "error"
        log_event(logger, logging.INFO,
                  f"Grounding retry ({category}): {outcome}",
                  event="grounding_retry", category=category, outcome=outcome,
                  latency_ms=round((time.time() - start_time) * 1000))
        return result_text

    async def _grounding_retry_inner(self, messages: List[Any],
                                     ctx: TurnContext):
        """Returns (result_text | None, outcome)."""
        if self._tool_choice_supported:
            try:
                forced = await self._chat.bind_tools(
                    self._lc_tools, tool_choice="required").ainvoke(messages)
            except Exception as e:
                # vLLM without guided-decoding tool_choice answers 400; any
                # BadRequest here means "not supported" — remember and fall
                # back to the nudge path below.
                if "400" not in str(e) and "BadRequest" not in type(e).__name__:
                    raise
                logger.warning(f"tool_choice=required rejected by backend: {e}")
                self._tool_choice_supported = False
            else:
                tool_calls = getattr(forced, "tool_calls", None) or []
                if not tool_calls:
                    return None, "no_tool_call"
                tool_messages = []
                for tc in tool_calls:
                    params = {k: v for k, v in (tc.get("args") or {}).items()
                              if v is not None}
                    result = await self.tool_manager.execute_tool(
                        ToolCall(name=tc["name"], params=params, raw=""))
                    self._collect_spoken_result(ctx, tc["name"], result)
                    tool_messages.append(ToolMessage(
                        content=getattr(result, "message", "") or "Done.",
                        tool_call_id=tc.get("id") or "forced_0"))
                # Compose with no tools bound — cannot recurse. Even an empty
                # compose is fine: _fold_unspoken_results speaks the data.
                composed = await self._chat.ainvoke(
                    messages + [forced, *tool_messages])
                return (self._extract_text(composed) or "").strip() or None, "tool_used"

        # Fallback: strong instruction re-run through the normal agent.
        nudge = SystemMessage(content=grounding.NUDGE)
        result = await self._agent.ainvoke(
            {"messages": messages + [nudge]},
            config={"recursion_limit": 5})
        out = result.get("messages", []) if isinstance(result, dict) else []
        text = self._extract_text(out[-1]) if out else ""
        return (text or "").strip() or None, "forced_unsupported_nudge"
