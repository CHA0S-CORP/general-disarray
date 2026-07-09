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
import logging
import time
from typing import Any, Dict, List, Optional

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from langgraph.errors import GraphRecursionError
    from langgraph.prebuilt import create_react_agent
    from pydantic import Field, create_model
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from config import Config
from llm_engine import LLMEngine, ToolCall
from logging_utils import log_event
from telemetry import create_span, Metrics

logger = logging.getLogger(__name__)

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

        messages: List[Any] = [
            SystemMessage(content=self._build_system_prompt(call_context))]
        messages.extend(self._to_lc_messages(
            conversation_history[-self.config.max_conversation_turns * 2:]))

        # Each tool round costs one agent step (model call) + one tool step.
        recursion_limit = 2 * self.config.llm_max_tool_rounds + 1

        with create_span("llm.generate_agent", {
            "llm.model": self.config.llm_model,
            "llm.recursion_limit": recursion_limit,
        }) as span:
            start_time = time.time()
            try:
                result = await asyncio.wait_for(
                    self._agent.ainvoke(
                        {"messages": messages},
                        config={"recursion_limit": recursion_limit},
                    ),
                    timeout=self.config.llm_agent_timeout_s,
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
            # gpt-oss and friends can legitimately return empty content.
            if not content or not content.strip():
                logger.warning("Agent returned empty content")
                Metrics.record_llm_error(self.config.llm_model, "empty_response")
                return self._fallback_error()

        # Marker safety net + informational-tool append, shared with the
        # classic engine (also covers text mode, where this IS the tool path).
        return await self._apply_marker_tools(content.strip())
