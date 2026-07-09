"""
LLM Engine
==========
Handles LLM inference with tool calling support.
Supports multiple backends: vLLM, Ollama, LM Studio.
"""

import asyncio
import json
import random
import re
import time
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

try:
    from openai import AsyncOpenAI
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from config import Config
from logging_utils import log_event
from telemetry import create_span, Metrics

if TYPE_CHECKING:
    from tool_manager import ToolManager




logger = logging.getLogger(__name__)


_REFORMAT_SYSTEM_PROMPT = """Rewrite the user's message so it can be read aloud naturally by a text-to-speech engine on a phone call.
Preserve EVERY piece of information: numbers, names, identifiers, dates, times, quantities, statuses. Do not summarize, do not drop details, do not add commentary or greetings.
- Say dates, times, and numbers the way a person would: "July eighth at five oh three PM", "ninety-nine point two percent".
- Speak URLs as just their site name ("the Grafana dashboard", "example dot com"); spell out short IDs and codes letter by letter or digit by digit.
- Expand abbreviations and symbols ("ms" -> "milliseconds", "%" -> "percent", "&" -> "and").
- Plain spoken prose only: no markdown, bullets, emoji, or headings.
Output ONLY the rewritten message."""


def _format_caller(remote_uri: str) -> str:
    """'sip:1001@pbx' / '"Bob" <sip:1001@pbx>' -> '1001'; falls back to raw."""
    m = re.search(r'sips?:([^@;>\s]+)', remote_uri or "")
    return m.group(1) if m else (remote_uri or "unknown")


@dataclass
class ToolCall:
    """Parsed tool call from LLM response."""
    name: str
    params: Dict[str, Any]
    raw: str


class LLMEngine:
    """LLM inference engine with tool support."""
    
    def __init__(self, config: Config, tool_manager: 'ToolManager'):
        self.config = config
        self.tool_manager = tool_manager
        self.client: Optional[AsyncOpenAI] = None
        
    async def start(self):
        """Initialize the LLM client."""
        if not OPENAI_CLIENT_AVAILABLE:
            logger.warning("OpenAI client not available, using mock LLM")
            return
            
        # Create OpenAI-compatible client for local LLM
        self.client = AsyncOpenAI(
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            timeout=60.0
        )
        
        # Test connection
        try:
            models = await self.client.models.list()
            logger.info(f"Connected to LLM backend. Available models: {[m.id for m in models.data]}")
        except Exception as e:
            logger.warning(f"Could not connect to LLM backend: {e}")
            logger.info("Will retry on first request")
            
    async def stop(self):
        """Cleanup."""
        if self.client:
            await self.client.close()
            
    async def generate_greeting(self) -> str:
        """Generate a greeting for incoming calls."""
        # Use configured greetings
        greetings = self.config.phrases.greetings
        
        import random
        return random.choice(greetings)
        
    async def generate_response(
        self,
        conversation_history: List[Dict[str, str]],
        call_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a response to the conversation."""
        
        # Build messages with system prompt
        messages = [
            {"role": "system", "content": self._build_system_prompt(call_context)}
        ]

        messages.extend(self._history_window(conversation_history, call_context))

        # Generate response
        if self._native_tools_active():
            response_text = await self._generate_native(messages)
        else:
            response_text = await self._generate(messages)

        # Parse and execute any text-marker tool calls. This also runs in
        # native mode as a safety net for models that ignore the tools param
        # and emit [TOOL:...] markers anyway.
        return await self._apply_marker_tools(response_text)

    async def _apply_marker_tools(self, response_text: str) -> str:
        """Execute [TOOL:...] markers in a reply and fold in spoken results.

        Shared marker postprocess used by every generation path (classic,
        native safety net, langgraph agent).
        """
        response_text, tool_results = await self._process_tool_calls(response_text)

        # Append results from informational tools (like WEATHER)
        # These tools return data that should be spoken to the user
        for result in tool_results:
            tool_name = result.get("tool", "")
            tool_result = result.get("result")

            # For informational tools (speak_result=True), append the message
            tool = self.tool_manager.get_tool(tool_name)
            if getattr(tool, "speak_result", False) and tool_result:
                if hasattr(tool_result, 'message') and tool_result.message:
                    # Add the result to the response
                    if response_text:
                        response_text = f"{response_text} {tool_result.message}"
                    else:
                        response_text = tool_result.message

        return response_text


    def _history_window(
        self,
        conversation_history: List[Dict[str, str]],
        call_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """The verbatim turns to send alongside the system prompt.

        When the rolling summary is active, main.py has already sliced off the
        turns it folded into `conversation_summary` and passes the rest — so
        applying the tail bound again here would drop the newly-overflowed
        turn from BOTH the window and the (not yet updated) summary. The
        summarizer runs a turn behind by design, so the window it hands us is
        legitimately a little longer than max_conversation_turns * 2.

        A generous safety cap still applies: summarization is fail-open, and a
        backend that keeps failing must not grow the prompt without bound.
        """
        window = self.config.max_conversation_turns * 2
        if not self.config.summary_enabled:
            return conversation_history[-window:]

        cap = window * 2
        if len(conversation_history) > cap:
            logger.warning(
                f"Summary is lagging ({len(conversation_history)} un-summarized "
                f"messages); clipping history to the last {cap}")
            return conversation_history[-cap:]
        return list(conversation_history)

    def _build_system_prompt(self, call_context: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt with dynamic context and tools."""
        prompt = self.config.system_prompt
        
        # Add time context
        now = datetime.now()
        prompt += f"\n\nCurrent time: {now.strftime('%I:%M %p on %A, %B %d, %Y')}"
        
        # Add call context
        if call_context:
            prompt += f"\n\nCall information:"
            prompt += f"\n- Caller: {_format_caller(call_context.get('remote_uri', 'unknown'))}"
            prompt += f"\n- Call length so far: {call_context.get('duration', 0):.0f} seconds"

            # Cross-call caller memory (loaded at call start, engine-agnostic).
            memory = call_context.get("caller_memory")
            if memory:
                prompt += (
                    "\n\nWhat you remember about this caller from previous calls"
                    " (use naturally, don't recite):\n" + memory)

            # Rolling summary of earlier turns that no longer fit the window.
            summary = call_context.get("conversation_summary")
            if summary:
                prompt += "\n\nConversation so far (earlier in this call):\n" + summary

            # Optional auto-injected knowledge-base excerpts for this turn.
            knowledge = call_context.get("knowledge_context")
            if knowledge:
                prompt += (
                    "\n\nRelevant excerpts from your knowledge base:\n" + knowledge)


        # Add dynamic tools section from ToolManager. In native mode the tool
        # schemas travel in the request's `tools` param instead — including the
        # [TOOL:...] marker instructions there would just confuse the model.
        if not self._native_tools_active():
            tools_prompt = self.tool_manager.get_tools_prompt()
            if tools_prompt:
                prompt += f"\n\n{tools_prompt}"

        return prompt

    def _sampling_kwargs(self) -> Dict[str, Any]:
        """Sampling params shared by both generation paths.

        frequency_penalty is included only when nonzero so backends that
        reject the param are unaffected at the default setting.
        """
        kwargs: Dict[str, Any] = {
            "model": self.config.llm_model,
            "max_tokens": self.config.llm_max_tokens,
            "temperature": self.config.llm_temperature,
            "top_p": self.config.llm_top_p,
        }
        if self.config.llm_frequency_penalty:
            kwargs["frequency_penalty"] = self.config.llm_frequency_penalty
        return kwargs

    def _fallback_error(self) -> str:
        """A configurable spoken fallback for generic LLM failures."""
        return random.choice(self.config.phrases.errors)

    async def reformat_for_speech(self, text: str, timeout_s: float) -> str:
        """Rewrite `text` into natural spoken form, preserving all facts.

        Fail-open: any failure (no client, timeout, error, suspicious result)
        returns the original text — a call must never be blocked or corrupted
        by the reformatter.
        """
        if not text or not text.strip():
            return text
        # Requires the OpenAI-compatible client (vLLM/LM Studio). Backends
        # without it (Ollama override, mock mode) skip the rewrite entirely.
        if self.client is None:
            return text

        # Reformat-specific request shape, NOT the conversation sampling:
        # - Reasoning models (e.g. gpt-oss) spend hundreds of tokens thinking
        #   before the rewrite, so the conversation budget (LLM_MAX_TOKENS) is
        #   far too small — an exhausted budget means empty content.
        # - They also misbehave under constrained sampling (temp/top_p tuned
        #   for conversation makes gpt-oss deliberate for 30+ seconds and then
        #   echo the input verbatim); the model's server-side defaults rewrite
        #   quickly and well, so temperature/top_p are deliberately omitted.
        kwargs: Dict[str, Any] = {
            "model": self.config.llm_model,
            "max_tokens": max(self.config.llm_max_tokens, 2048),
        }
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": _REFORMAT_SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    **kwargs,
                ),
                timeout=timeout_s,
            )
            result = response.choices[0].message.content
        except (asyncio.TimeoutError, Exception) as e:
            log_event(logger, logging.WARNING,
                      f"Message reformat failed ({type(e).__name__}); using original",
                      event="message_reformat", outcome="error",
                      chars_in=len(text))
            return text

        if (not result or not result.strip()
                or "[TOOL:" in result):
            log_event(logger, logging.WARNING,
                      "Message reformat produced no usable rewrite; using original",
                      event="message_reformat", outcome="rejected",
                      chars_in=len(text))
            return text

        result = result.strip()
        log_event(logger, logging.INFO,
                  f"Message reformatted for speech ({len(text)} -> {len(result)} chars)",
                  event="message_reformat", outcome="ok",
                  chars_in=len(text), chars_out=len(result))
        return result

    async def summarize_text(self, system_prompt: str, text: str,
                             timeout_s: float) -> Optional[str]:
        """Run a one-shot utility completion (summaries, fact extraction).

        Returns the model's text, or None on any failure — callers treat None
        as "keep what you had" (fail-open). Same request shape rationale as
        reformat_for_speech: generous token budget for reasoning models,
        server-default sampling.
        """
        if not text or not text.strip() or self.client is None:
            return None
        kwargs: Dict[str, Any] = {
            "model": self.config.llm_model,
            "max_tokens": max(self.config.llm_max_tokens, 2048),
        }
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    **kwargs,
                ),
                timeout=timeout_s,
            )
            result = response.choices[0].message.content
        except (asyncio.TimeoutError, Exception) as e:
            log_event(logger, logging.WARNING,
                      f"Utility summarization failed ({type(e).__name__})",
                      event="summarize_text", outcome="error",
                      chars_in=len(text))
            return None
        if not result or not result.strip() or "[TOOL:" in result:
            return None
        return result.strip()

    def _native_tools_active(self) -> bool:
        """True when native OpenAI function calling should be used."""
        return (self.config.llm_tool_calling.lower() == "native"
                and self.client is not None)

    def _build_native_tools(self) -> List[Dict[str, Any]]:
        """Convert registered tools into OpenAI function-calling schemas."""
        tools = []
        for name, tool in self.tool_manager.tools.items():
            if not getattr(tool, "enabled", True):
                continue
            properties: Dict[str, Any] = {}
            required: List[str] = []
            for pname, spec in (getattr(tool, "parameters", {}) or {}).items():
                prop: Dict[str, Any] = {"type": spec.get("type", "string")}
                if spec.get("description"):
                    prop["description"] = spec["description"]
                properties[pname] = prop
                if spec.get("required"):
                    required.append(pname)
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": getattr(tool, "description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return tools

    async def _generate_native(self, messages: List[Dict[str, Any]]) -> str:
        """Native function-calling loop (LLM_TOOL_CALLING=native).

        Sends tool schemas via the OpenAI `tools` param, executes any returned
        tool_calls through the ToolManager, feeds results back as `tool`
        messages, and repeats until the model answers in plain text.
        """
        tools = self._build_native_tools()
        messages = list(messages)
        # Bound on tool-call round trips per turn so a model that keeps asking
        # for tools can't loop forever on a live phone call. The budget buys
        # max_rounds *tool* rounds, plus one final completion to turn the last
        # tool result into a spoken answer — that final call withholds the
        # tools param so the model has to answer rather than ask again.
        max_rounds = self.config.llm_max_tool_rounds

        with create_span("llm.generate_native", {
            "llm.model": self.config.llm_model,
            "llm.tools_count": len(tools),
        }) as span:
            try:
                for round_no in range(max_rounds + 1):
                    budget_spent = round_no == max_rounds
                    start_time = time.time()
                    response = await self.client.chat.completions.create(
                        messages=messages,
                        tools=None if budget_spent else (tools or None),
                        **self._sampling_kwargs(),
                    )
                    Metrics.record_llm_latency(
                        (time.time() - start_time) * 1000, self.config.llm_model)

                    msg = response.choices[0].message
                    tool_calls = getattr(msg, "tool_calls", None)
                    if not tool_calls:
                        span.set_attribute("llm.tool_rounds", round_no)
                        content = msg.content
                        if content is None or not content.strip():
                            logger.warning("LLM returned empty content in native mode")
                            return self._fallback_error()
                        return content.strip()

                    messages.append({
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [tc.model_dump() for tc in tool_calls],
                    })
                    for tc in tool_calls:
                        try:
                            params = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            params = {}
                        try:
                            result = await self.tool_manager.execute_tool(ToolCall(
                                name=tc.function.name,
                                params=params,
                                raw=tc.function.arguments or "",
                            ))
                            result_text = getattr(result, "message", "") or ""
                        except Exception as e:
                            logger.error(f"Native tool execution error: {e}")
                            result_text = f"Tool {tc.function.name} failed."
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_text,
                        })

                span.set_attribute("llm.tool_rounds", max_rounds)
                logger.warning("Native tool-calling hit the round limit without a final answer")
                return "I'm sorry, that request took too many steps. Could you try again?"

            except Exception as e:
                logger.error(f"LLM native generation error: {e}")
                span.record_exception(e)
                Metrics.record_llm_error(self.config.llm_model, type(e).__name__)
                return self._fallback_error()
        
    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM to generate a response."""
        if not self.client:
            # Mock response
            return self._mock_response(messages)
        
        with create_span("llm.generate", {
            "llm.model": self.config.llm_model,
            "llm.messages_count": len(messages),
            "llm.max_tokens": self.config.llm_max_tokens
        }) as span:
            start_time = time.time()
            first_token_time = None
            try:
                response = await self.client.chat.completions.create(
                    messages=messages,
                    **self._sampling_kwargs(),
                )
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                # --- CRITICAL FIX ---
                # gpt-oss-20b / vLLM can return None for content if it gets confused 
                # or tries to use native tools. We must fallback to empty string.
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                
                span.set_attribute("llm.latency_ms", latency_ms)
                span.set_attribute("llm.finish_reason", finish_reason or "unknown")
                
                # Record usage metrics if available
                if hasattr(response, 'usage') and response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    
                    span.set_attribute("llm.prompt_tokens", prompt_tokens)
                    span.set_attribute("llm.completion_tokens", completion_tokens)
                    span.set_attribute("llm.total_tokens", total_tokens)
                    
                    # Record token metrics
                    Metrics.record_llm_tokens_input(prompt_tokens, self.config.llm_model)
                    Metrics.record_llm_tokens_output(completion_tokens, self.config.llm_model)
                    Metrics.record_llm_context_tokens(total_tokens, self.config.llm_model)
                    
                    # Calculate tokens per second for output
                    if completion_tokens > 0 and latency_ms > 0:
                        tps = completion_tokens / (latency_ms / 1000)
                        Metrics.record_llm_tokens_per_second(tps, self.config.llm_model)
                        span.set_attribute("llm.tokens_per_second", tps)
                
                Metrics.record_llm_latency(latency_ms, self.config.llm_model)
                
                # --- FIX: Handle Empty Content / Length Finish ---
                if content is None or not content.strip():
                    logger.warning(f"LLM returned empty content. Reason: {finish_reason}")
                    span.set_attribute("llm.empty_response", True)
                    Metrics.record_llm_error(self.config.llm_model, "empty_response")
                    
                    # If it ran out of tokens while thinking, we can't recover easily 
                    # without more tokens, so we give a polite error.
                    if finish_reason == 'length':
                        return "I'm sorry, I was thinking too hard and ran out of time. Could you ask that again?"

                    return self._fallback_error()

                span.set_attribute("llm.response_length", len(content))
                return content
                
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                span.record_exception(e)
                Metrics.record_llm_error(self.config.llm_model, type(e).__name__)
                return self._fallback_error()
            
    def _mock_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate mock response when LLM unavailable."""
        last_user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"].lower()
                break
                
        # Simple keyword matching for testing
        if "timer" in last_user_msg or "remind" in last_user_msg:
            return "I'll set that timer for you. [TOOL:SET_TIMER:duration=300,message=Timer complete]"
        elif "call" in last_user_msg and "back" in last_user_msg:
            return "I'll call you back. [TOOL:CALLBACK:delay=60,message=Callback as requested]"
        elif "bye" in last_user_msg or "goodbye" in last_user_msg:
            return "Goodbye! Have a great day! [TOOL:HANGUP]"
        elif "help" in last_user_msg:
            return "I can help you with timers, reminders, and callbacks. What would you like me to do?"
        else:
            return "I understand. Is there anything specific I can help you with?"
            
    async def _process_tool_calls(self, response: str) -> Tuple[str, List[Dict]]:
        """Parse and execute tool calls from response."""
        tool_results = []
        
        # Find tool calls in format: [TOOL:name:param1=val1,param2=val2] or [TOOL:name]
        pattern_with_params = r'\[TOOL:(\w+):([^\]]+)\]'
        pattern_no_params = r'\[TOOL:(\w+)\]'
        
        # Process tools with parameters
        matches = list(re.finditer(pattern_with_params, response))
        for match in matches:
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # Parse parameters
            # Split only on commas that immediately precede a `key=` token, so
            # commas inside a value (e.g. timer/callback messages) are preserved
            # rather than truncating the value and dropping trailing fragments.
            params = {}
            for param in re.split(r',(?=\s*\w+=)', params_str):
                if '=' in param:
                    key, value = param.split('=', 1)
                    value = self._parse_param_value(value)
                    params[key.strip()] = value
                    
            tool_call = ToolCall(
                name=tool_name,
                params=params,
                raw=match.group(0)
            )
            
            try:
                result = await self.tool_manager.execute_tool(tool_call)
                tool_results.append({
                    "tool": tool_name,
                    "params": params,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                tool_results.append({
                    "tool": tool_name,
                    "params": params,
                    "error": str(e)
                })
        
        # Process tools without parameters (e.g., HANGUP)
        # Remove already-matched sections first to avoid double-matching
        temp_response = re.sub(pattern_with_params, '', response)
        matches_no_params = list(re.finditer(pattern_no_params, temp_response))
        
        for match in matches_no_params:
            tool_name = match.group(1)
            
            tool_call = ToolCall(
                name=tool_name,
                params={},
                raw=match.group(0)
            )
            
            try:
                result = await self.tool_manager.execute_tool(tool_call)
                tool_results.append({
                    "tool": tool_name,
                    "params": {},
                    "result": result
                })
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                tool_results.append({
                    "tool": tool_name,
                    "params": {},
                    "error": str(e)
                })
                
        # Remove all tool calls from response text
        clean_response = re.sub(pattern_with_params, '', response)
        clean_response = re.sub(pattern_no_params, '', clean_response).strip()
        
        return clean_response, tool_results
        
    def _parse_param_value(self, value: str) -> Any:
        """Parse parameter value to appropriate type."""
        value = value.strip()

        # Try boolean
        if value.lower() in ('true', 'yes'):
            return True
        if value.lower() in ('false', 'no'):
            return False

        # Only coerce numbers that can't be identifiers. Phone numbers (leading
        # "+") and area/zip codes (leading "0" padding) must stay strings so the
        # original value is preserved and the right number is dialed.
        # Canonical int: optional "-", then "0" or a non-zero-leading run of digits.
        if re.fullmatch(r'-?(?:0|[1-9]\d*)', value):
            return int(value)
        # Float: non-zero-padded integer part with optional/leading/trailing dot
        # (".5", "5.", "1.25") and an optional exponent ("1e2", "1.5E-3").
        if re.fullmatch(r'-?(?:(?:0|[1-9]\d*)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value):
            return float(value)

        # Return as string
        return value


class OllamaEngine(LLMEngine):
    """
    Alternative engine using Ollama directly.
    Useful if not running vLLM.
    """
    
    def __init__(self, config: Config, tool_manager: 'ToolManager'):
        super().__init__(config, tool_manager)
        self.ollama_url = config.llm_base_url.replace('/v1', '')
        
    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate using Ollama API."""
        if not HTTPX_AVAILABLE:
            return self._mock_response(messages)
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.config.llm_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": self.config.llm_max_tokens,
                            "temperature": self.config.llm_temperature,
                            "top_p": self.config.llm_top_p
                        }
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
                
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return "I'm sorry, I'm having trouble processing that."


class LMStudioEngine(LLMEngine):
    """
    Alternative engine for LM Studio.
    LM Studio provides OpenAI-compatible API on port 1234 by default.
    """
    
    def __init__(self, config: Config, tool_manager: 'ToolManager'):
        config.llm_base_url = config.llm_base_url or "http://localhost:1234/v1"
        super().__init__(config, tool_manager)


# Factory function
def create_llm_engine(config: Config, tool_manager: 'ToolManager') -> LLMEngine:
    """Create appropriate LLM engine based on config."""
    backend = config.llm_backend.lower()

    if backend == "ollama":
        return OllamaEngine(config, tool_manager)
    elif backend == "lmstudio":
        return LMStudioEngine(config, tool_manager)
    elif backend == "langgraph":
        # Agentic engine (multi-step tool reasoning via LangGraph). Optional
        # dependency: fall back to the classic engine when not installed so a
        # slim image still runs.
        try:
            from langchain_engine import LangChainEngine
            return LangChainEngine(config, tool_manager)
        except ImportError as e:
            logger.warning(
                f"LLM_BACKEND=langgraph but LangChain deps unavailable ({e}); "
                "falling back to the classic engine")
            return LLMEngine(config, tool_manager)
    else:  # vllm or default
        return LLMEngine(config, tool_manager)
