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
from dataclasses import dataclass, field
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

import grounding
from config import Config
from logging_utils import log_event
from sentence_stream import SentenceAssembler, split_into_sentences
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


@dataclass
class TurnContext:
    """Per-turn tool bookkeeping, owned by one generate_response /
    stream_response invocation and threaded explicitly through every path
    that executes tools.

    Replaces the old engine instance fields (_turn_spoken_results /
    _turn_tool_calls): two concurrent turns on the same engine each carry
    their own context and can no longer corrupt each other's bookkeeping.
    """

    # speak_result tool messages executed during this turn by the
    # native/agent paths, pending the relay check in _fold_unspoken_results.
    # (tool_name, model_facing_message, spoken_text) — see _collect_spoken_result.
    spoken_results: List[Tuple[str, str, str]] = field(default_factory=list)
    # Tool calls executed this turn (any engine path); drives the
    # text-mode grounding retry.
    tool_calls: int = 0


class _StreamedToolCall:
    """A tool call reconstructed from streamed deltas, shaped like the SDK's
    ChatCompletionMessageToolCall so _generate_native can consume it."""

    def __init__(self, call_id: str, name: str, arguments: str):
        from types import SimpleNamespace
        self.id = call_id
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=arguments)

    def model_dump(self) -> Dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name,
                             "arguments": self.function.arguments}}


class _StreamedMessage:
    """Assistant message reconstructed from streamed deltas (content +
    index-keyed tool_call accumulation)."""

    def __init__(self, content: Optional[str],
                 tool_calls: Optional[List[_StreamedToolCall]]):
        self.content = content
        self.tool_calls = tool_calls or None


class LLMEngine:
    """LLM inference engine with tool support."""
    
    def __init__(self, config: Config, tool_manager: 'ToolManager'):
        self.config = config
        self.tool_manager = tool_manager
        self.client: Optional[AsyncOpenAI] = None
        # Per-turn tool bookkeeping lives in a TurnContext created at the top
        # of generate_response/stream_response and threaded explicitly —
        # never on the engine instance (see TurnContext).
        
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

        ctx = TurnContext()

        # Generate response
        if self._native_tools_active():
            response_text = await self._generate_native(messages, ctx)
            # Marker safety net for models that ignore the tools param and
            # emit [TOOL:...] markers anyway.
            response_text = await self._apply_marker_tools(response_text, ctx)
        else:
            response_text = await self._generate(messages)
            # Parse and execute the text-marker tool calls.
            response_text = await self._apply_marker_tools(response_text, ctx)
            # "Never guess" enforcement for the default text-marker mode
            # (the native/agent paths run their own retry inside the loop).
            response_text = await self._grounding_retry_text(
                messages, response_text, ctx)
        return self._fold_unspoken_results(response_text, ctx)

    # ------------------------------------------------------------------
    # Token streaming (LLM→TTS): stream_response and its helpers
    # ------------------------------------------------------------------

    async def stream_response(
        self,
        conversation_history: List[Dict[str, str]],
        call_context: Optional[Dict[str, Any]] = None,
    ):
        """Async generator: the streaming counterpart of generate_response.

        Yields, in order:
          - zero or more ``{"type": "sentence", "text": str}`` events — the
            caller speaks each immediately;
          - exactly one terminal ``{"type": "final", "text": str}`` event —
            the definitive full response for history/metrics, never spoken.

        Invariant: the whitespace-normalized concatenation of all yielded
        sentence texts equals the whitespace-normalized final text.

        Real token streaming happens only when the config/engine/turn allow
        it (see _can_stream); otherwise the default path runs the existing
        generate_response and replays its text through the sentence splitter,
        so every engine (Ollama, LangChain, mock) supports this one
        consumption path.
        """
        if not self._can_stream(conversation_history):
            async for event in self._stream_via_generate(
                    conversation_history, call_context):
                yield event
            return

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(call_context)}
        ]
        messages.extend(self._history_window(conversation_history, call_context))

        # Same per-turn bookkeeping as generate_response, owned by this turn.
        ctx = TurnContext()

        if self._native_tools_active():
            inner = self._stream_native(messages, ctx)
        else:
            inner = self._stream_classic(messages, ctx)
        try:
            async for event in inner:
                yield event
        finally:
            await inner.aclose()

    def _streaming_supported(self) -> bool:
        """True when this engine generates through the OpenAI client and can
        therefore token-stream. Engines that override _generate away from the
        OpenAI client (Ollama, LangChain agent) return False."""
        return True

    def _can_stream(self, conversation_history: List[Dict[str, str]]) -> bool:
        """Should this turn use real token streaming?"""
        if not self.config.llm_streaming:
            return False
        if not self.config.tts_sentence_streaming:
            # Sentence-by-sentence TTS is the whole point of token streaming;
            # with TTS_SENTENCE_STREAMING off the operator asked for
            # whole-text TTS, which only the default (non-streaming) path
            # honors — streaming would silently re-enable per-sentence TTS.
            return False
        if self.client is None or not self._streaming_supported():
            return False
        # Grounding pre-classification: the text-mode grounding retry
        # REPLACES the response after the fact, which is impossible once
        # sentence 1 is audible. A live-data question whose tools are loaded
        # takes the default (non-streaming) path — those turns are tool-bound
        # anyway and keep their full grounding-retry behavior. Probing
        # grounding_category with an empty reply reduces to
        # live_data_category (promised_action("") is False), which is exactly
        # the response-independent detector we need here.
        if self.config.grounding_retry_enabled and self.tool_manager.tools:
            last_user = self._last_user_text(conversation_history)
            category = grounding.grounding_category(last_user, "")
            if category and self._category_tools_available(category):
                return False
        return True

    @staticmethod
    def _last_user_text(messages: List[Dict[str, Any]]) -> str:
        return str(next((m.get("content") or "" for m in reversed(messages)
                         if m.get("role") == "user"), ""))

    async def _stream_via_generate(
        self,
        conversation_history: List[Dict[str, str]],
        call_context: Optional[Dict[str, Any]] = None,
    ):
        """Default path: run the existing generate_response, then replay its
        text as sentence events + the final event. Preserves today's sentence
        pipelining (and the TTS_SENTENCE_STREAMING switch) for non-streaming
        engines and turns."""
        text = await self.generate_response(conversation_history, call_context)
        text = text or ""
        if not text.strip():
            chunks = []
        elif (not self.config.tts_sentence_streaming
                or self._is_precached_phrase(text)):
            chunks = [text]
        else:
            chunks = split_into_sentences(text)
        for chunk in chunks:
            yield {"type": "sentence", "text": chunk}
        yield {"type": "final", "text": text}

    def _is_precached_phrase(self, text: str) -> bool:
        """True when `text` is one of the whole-phrase TTS pre-cache
        candidates (config.phrases). The audio pipeline caches these keyed on
        the whole lowercased phrase, so they must reach main.py as a SINGLE
        chunk for its per-chunk cache lookup to hit — the non-streaming
        _speak path checks the whole-text cache before splitting, and
        pre-splitting a multi-sentence error phrase would turn the failure
        path into live TTS calls."""
        key = text.lower().strip()
        return any(p.lower().strip() == key
                   for p in self.config.phrases.get_all_phrases_for_cache())

    async def _replay_text(self, text: str):
        """Yield a fixed text (error phrase, fallback) as sentence events plus
        the final event. Pre-cached phrases stay whole so main's TTS cache
        lookup hits (see _is_precached_phrase)."""
        chunks = ([text] if self._is_precached_phrase(text)
                  else split_into_sentences(text))
        for chunk in chunks:
            yield {"type": "sentence", "text": chunk}
        yield {"type": "final", "text": text}

    @staticmethod
    async def _close_stream(stream) -> None:
        """Close an AsyncOpenAI stream so a cancelled turn (barge-in) stops
        the backend's generation instead of leaking it."""
        if stream is None:
            return
        try:
            close = getattr(stream, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
                return
            response = getattr(stream, "response", None)
            if response is not None:
                await response.aclose()
        except Exception as e:
            logger.debug(f"Error closing LLM stream: {e}")

    def _log_grounding_skipped(self, ctx: TurnContext, last_user: str,
                               final_text: str) -> None:
        """Known, accepted loss of the streaming path: PROMISED_ACTION is
        detected from the *response* text and cannot be pre-classified. When
        a streamed response turns out to be an ungrounded promise (or a
        live-data answer that slipped past pre-classification), the
        replace-style retry is impossible — sentence 1 is already audible —
        so log it instead (observable, tunable later)."""
        if (not self.config.grounding_retry_enabled
                or not self.tool_manager.tools):
            return
        category = self._grounding_category(last_user, final_text, ctx)
        if category and self._category_tools_available(category):
            log_event(logger, logging.INFO,
                      f"Grounding retry skipped (streaming): {category}",
                      event="grounding_skipped_streaming", category=category)

    async def _finalize_stream(self, assembler: SentenceAssembler,
                               emitted: List[str], finish_reason: Optional[str],
                               last_user: str, ctx: TurnContext):
        """Shared end-of-stream tail for both streaming paths (no pending
        tool round): process the un-emitted remainder (marker execution +
        speak_result folding), yield its sentences, then the final event.

        The remainder postprocess only APPENDS text after the emitted prefix
        (markers were only ever in the un-emitted part; _apply_marker_tools
        strips them and appends speak_result messages at the end;
        _fold_unspoken_results prepends only within the remainder), so the
        emitted prefix stays a prefix of final — the invariant holds.
        """
        remainder = assembler.flush()
        if not emitted and not remainder:
            # Mirror _generate's empty-content handling.
            logger.warning(
                f"LLM returned empty streamed content. Reason: {finish_reason}")
            Metrics.record_llm_error(self.config.llm_model, "empty_response")
            if finish_reason == "length":
                text = ("I'm sorry, I was thinking too hard and ran out of "
                        "time. Could you ask that again?")
            else:
                text = self._fallback_error()
            async for event in self._replay_text(text):
                yield event
            return

        tail = await self._apply_marker_tools(remainder, ctx) if remainder else ""
        tail = self._fold_unspoken_results(tail, ctx)
        for chunk in split_into_sentences(tail):
            emitted.append(chunk)
            yield {"type": "sentence", "text": chunk}

        final_text = " ".join(emitted)
        self._log_grounding_skipped(ctx, last_user, final_text)
        yield {"type": "final", "text": final_text}

    async def _stream_classic(self, messages: List[Dict[str, Any]],
                              ctx: TurnContext):
        """Token-stream the classic (text-marker) path.

        Sentences are emitted as they form; anything that could belong to a
        [TOOL:...] marker is held by the assembler, and the marker/tool
        postprocess runs on the un-emitted remainder at end of stream.
        """
        last_user = self._last_user_text(messages)
        assembler = SentenceAssembler()
        emitted: List[str] = []
        finish_reason: Optional[str] = None
        start_time = time.time()
        first_token_time: Optional[float] = None
        failed_before_output = False

        try:
            stream = await self.client.chat.completions.create(
                messages=messages, stream=True, **self._sampling_kwargs())
        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            Metrics.record_llm_error(self.config.llm_model, type(e).__name__)
            async for event in self._replay_text(self._fallback_error()):
                yield event
            return

        try:
            try:
                async for chunk in stream:
                    choices = getattr(chunk, "choices", None)
                    if not choices:
                        continue
                    choice = choices[0]
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if not content:
                        continue
                    if first_token_time is None:
                        first_token_time = time.time()
                        Metrics.record_llm_ttft(
                            (first_token_time - start_time) * 1000,
                            self.config.llm_model)
                    for sentence in assembler.feed(content):
                        emitted.append(sentence)
                        yield {"type": "sentence", "text": sentence}
            except Exception as e:
                # Mid-stream failure: already-spoken sentences can't be
                # unspoken, so keep what we have; if nothing has been SPOKEN
                # yet (tokens may have arrived but still sit in the
                # assembler's buffer), fall back to an error phrase like
                # _generate does — flushing the buffered fragment would speak
                # a dangling half-sentence (possibly an unterminated [TOOL:
                # marker) and record it as the definitive turn.
                logger.error(f"LLM stream error: {e}")
                Metrics.record_llm_error(self.config.llm_model, type(e).__name__)
                if not emitted:
                    failed_before_output = True
        finally:
            # Runs on normal completion, errors AND generator aclose()
            # (barge-in): the backend must stop generating either way.
            await self._close_stream(stream)

        if failed_before_output:
            async for event in self._replay_text(self._fallback_error()):
                yield event
            return

        Metrics.record_llm_latency(
            (time.time() - start_time) * 1000, self.config.llm_model)
        async for event in self._finalize_stream(
                assembler, emitted, finish_reason, last_user, ctx):
            yield event

    async def _stream_native(self, messages: List[Dict[str, Any]],
                             ctx: TurnContext):
        """Token-stream the native tool-calling path with hold-then-decide.

        Round 0 streams with `tools` attached. Content is held until
        ~_STREAM_HOLD_TOKENS content deltas arrive with no tool_call delta
        (models emit tool calls up front); then the hold releases and
        sentences stream to the end. A tool_call delta — before or after
        release — stops emission: the full assistant message is reconstructed
        from the deltas and the turn continues through the existing
        (non-streaming) _generate_native loop, preserving the round budget,
        grounding force and fold behavior. Because the post-tool rounds never
        stream, _fold_unspoken_results only ever rearranges text that comes
        AFTER the emitted prefix, keeping the invariant.
        """
        last_user = self._last_user_text(messages)
        # Round 0 counts against the tool-round budget exactly as in
        # _generate_native: with the budget already spent
        # (LLM_MAX_TOOL_ROUNDS=0) the request must go out WITHOUT the tools
        # param so the model has to answer in text — otherwise the streamed
        # path could execute tools the non-streaming engine never could.
        tools = (self._build_native_tools()
                 if self.config.llm_max_tool_rounds > 0 else [])
        assembler = SentenceAssembler()
        emitted: List[str] = []
        held: List[str] = []
        content_parts: List[str] = []
        tool_deltas: Dict[int, Dict[str, str]] = {}
        released = False
        finish_reason: Optional[str] = None
        start_time = time.time()
        first_token_time: Optional[float] = None
        failed_before_output = False

        try:
            stream = await self.client.chat.completions.create(
                messages=messages, tools=tools or None, stream=True,
                **self._sampling_kwargs())
        except Exception as e:
            logger.error(f"LLM native stream error: {e}")
            Metrics.record_llm_error(self.config.llm_model, type(e).__name__)
            async for event in self._replay_text(self._fallback_error()):
                yield event
            return

        try:
            try:
                async for chunk in stream:
                    choices = getattr(chunk, "choices", None)
                    if not choices:
                        continue
                    choice = choices[0]
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    for tcd in (getattr(delta, "tool_calls", None) or []):
                        acc = tool_deltas.setdefault(
                            getattr(tcd, "index", 0) or 0,
                            {"id": "", "name": "", "arguments": ""})
                        if getattr(tcd, "id", None):
                            acc["id"] = tcd.id
                        fn = getattr(tcd, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                acc["name"] += fn.name
                            if getattr(fn, "arguments", None):
                                acc["arguments"] += fn.arguments
                    content = getattr(delta, "content", None)
                    if not content:
                        continue
                    if first_token_time is None:
                        first_token_time = time.time()
                        Metrics.record_llm_ttft(
                            (first_token_time - start_time) * 1000,
                            self.config.llm_model)
                    content_parts.append(content)
                    if tool_deltas:
                        # Tool round decided: accumulate silently.
                        continue
                    if released:
                        for sentence in assembler.feed(content):
                            emitted.append(sentence)
                            yield {"type": "sentence", "text": sentence}
                    else:
                        held.append(content)
                        if len(held) >= self._STREAM_HOLD_TOKENS:
                            released = True
                            for sentence in assembler.feed("".join(held)):
                                emitted.append(sentence)
                                yield {"type": "sentence", "text": sentence}
                            held = []
            except Exception as e:
                logger.error(f"LLM native stream error: {e}")
                Metrics.record_llm_error(self.config.llm_model, type(e).__name__)
                if not emitted:
                    failed_before_output = True
                # A half-received tool round can't be executed safely.
                tool_deltas = {}
        finally:
            await self._close_stream(stream)

        if failed_before_output:
            async for event in self._replay_text(self._fallback_error()):
                yield event
            return

        Metrics.record_llm_latency(
            (time.time() - start_time) * 1000, self.config.llm_model)

        if tool_deltas:
            # Tool round: rebuild the assistant message and continue through
            # the existing non-streaming native loop (round budget, grounding
            # force, marker safety net, fold — all as generate_response).
            tool_calls = [
                _StreamedToolCall(acc["id"] or f"stream-tc-{idx}",
                                  acc["name"], acc["arguments"])
                for idx, acc in sorted(tool_deltas.items())
            ]
            msg = _StreamedMessage("".join(content_parts) or None, tool_calls)
            text = await self._generate_native(messages, ctx, first_message=msg)
            text = await self._apply_marker_tools(text, ctx)
            text = self._fold_unspoken_results(text, ctx)
            for chunk in split_into_sentences(text):
                emitted.append(chunk)
                yield {"type": "sentence", "text": chunk}
            yield {"type": "final", "text": " ".join(emitted)}
            return

        if held and not released:
            # Short pure-content answer: the stream ended inside the hold.
            for sentence in assembler.feed("".join(held)):
                emitted.append(sentence)
                yield {"type": "sentence", "text": sentence}

        async for event in self._finalize_stream(
                assembler, emitted, finish_reason, last_user, ctx):
            yield event

    # Content deltas released before deciding a streamed native round is a
    # pure-content answer (models emit tool calls up front, so a tool_call
    # delta after this many content tokens is rare — and still handled).
    _STREAM_HOLD_TOKENS = 16

    async def _grounding_retry_text(
        self,
        messages: List[Dict[str, Any]],
        response_text: str,
        ctx: TurnContext,
    ) -> str:
        """One nudged re-run when a live-data question ended the turn with
        zero executed tool markers (or the reply merely promised to check).
        Mirrors the native path's forced retry; keeps the original reply on
        any failure so the retry can only add grounding, never latency-error.
        """
        if not self.config.grounding_retry_enabled:
            return response_text
        if not self.tool_manager.tools:
            return response_text
        last_user = next((m.get("content") or "" for m in reversed(messages)
                          if m.get("role") == "user"), "")
        category = self._grounding_category(str(last_user), response_text, ctx)
        if not category or not self._category_tools_available(category):
            return response_text

        start_time = time.time()
        outcome = "error"
        retried: Optional[str] = None
        try:
            retried = await asyncio.wait_for(
                self._generate(messages + [
                    {"role": "system", "content": grounding.NUDGE}]),
                timeout=self.config.grounding_retry_timeout_s)
            retried = await self._apply_marker_tools(retried, ctx)
            outcome = "tool_used" if ctx.tool_calls else "no_tool_call"
        except asyncio.TimeoutError:
            outcome = "timeout"
        except Exception as e:
            logger.error(f"Grounding retry error (text): {e}")
        log_event(logger, logging.INFO,
                  f"Grounding retry ({category}): {outcome}",
                  event="grounding_retry", category=category, outcome=outcome,
                  latency_ms=round((time.time() - start_time) * 1000))
        # Only adopt the retry when it actually grounded itself in a tool.
        if outcome == "tool_used" and retried and retried.strip():
            return retried
        return response_text

    async def _apply_marker_tools(self, response_text: str,
                                  ctx: TurnContext) -> str:
        """Execute [TOOL:...] markers in a reply and fold in spoken results.

        Shared marker postprocess used by every generation path (classic,
        native safety net, langgraph agent). Tool executions are counted on
        the caller's TurnContext.
        """
        response_text, tool_results = await self._process_tool_calls(response_text)
        ctx.tool_calls += len(tool_results)

        # Append results from informational tools (like WEATHER)
        # These tools return data that should be spoken to the user
        for result in tool_results:
            tool_name = result.get("tool", "")
            tool_result = result.get("result")

            # For informational tools (speak_result=True), append the message
            tool = self.tool_manager.get_tool(tool_name)
            if getattr(tool, "speak_result", False) and tool_result:
                # to_speech(), not .message: a tool's model-facing result can be
                # material that reads fine but is unspeakable (scraped snippets,
                # bullets, URLs). to_speech() falls back to .message.
                spoken = (tool_result.to_speech()
                          if hasattr(tool_result, 'to_speech') else
                          getattr(tool_result, 'message', ''))
                if spoken:
                    # Add the result to the response
                    if response_text:
                        response_text = f"{response_text} {spoken}"
                    else:
                        response_text = spoken

        return response_text

    def _grounding_category(self, last_user: str, reply: str,
                            ctx: TurnContext) -> Optional[str]:
        """The category for a forced-tool retry this turn, or None.

        Zero tool calls -> the full check: a live-data question, or any promise
        to go and look.

        Tools already ran -> only a TRAILING promise counts. Calling a tool is
        not the same as answering it: the model can pick the wrong tool, or the
        right one can come back empty, and it then signs off with "Let me get
        that for you right away" — dead air, since the turn is over. But a
        promise mid-reply ("Let me check... it's 71 degrees") was kept, so only
        the final sentence is evidence of a broken one.
        """
        if ctx.tool_calls:
            return "PROMISED_ACTION" if grounding.trailing_promise(reply) else None
        return grounding.grounding_category(last_user, reply)

    def _collect_spoken_result(self, ctx: TurnContext, tool_name: str,
                               result: Any) -> None:
        """Record a successful speak_result tool execution from the native or
        agent tool loop onto the turn's context, for the relay check in
        _fold_unspoken_results."""
        tool = self.tool_manager.get_tool(tool_name)
        if not getattr(tool, "speak_result", False):
            return
        status = getattr(result, "status", None)
        if getattr(status, "value", status) != "success":
            return
        # Two different texts, and the difference matters. The relay check runs
        # against what the MODEL saw (did it deliver that content?), but what we
        # fold in when it didn't is what the CALLER should HEAR. For most tools
        # these are the same string; for search they are not.
        message = getattr(result, "message", "") or ""
        spoken = (result.to_speech() if hasattr(result, "to_speech")
                  else message) or message
        if spoken:
            ctx.spoken_results.append((tool_name, message, spoken))

    def _fold_unspoken_results(self, response_text: str,
                               ctx: TurnContext) -> str:
        """Prepend speak_result tool messages the model failed to relay.

        In native/agent mode the model sees an informational tool's result and
        is expected to deliver it, but it can just comment on it instead (e.g.
        praising a joke without telling it). When the final text shares almost
        no content with the tool message, the caller never heard the actual
        result — so speak the tool message first, then the model's text.
        """
        pending, ctx.spoken_results = ctx.spoken_results, []
        for tool_name, message, spoken in pending:
            if self._relayed(message, response_text):
                continue
            log_event(logger, logging.INFO,
                      f"Prepending unrelayed {tool_name} result to response",
                      event="tool_result_folded", tool=tool_name)
            response_text = (f"{spoken} {response_text}".strip()
                             if response_text else spoken)
        return response_text

    @staticmethod
    def _relayed(message: str, response_text: str) -> bool:
        """Did the response plausibly deliver the tool message's content?

        Verbatim inclusion counts; otherwise require a modest overlap of the
        message's significant words (paraphrases keep station names, units,
        conditions etc., while mere commentary shares almost nothing).
        """
        if not response_text:
            return False
        if message in response_text:
            return True
        words = {w for w in re.findall(r"[a-z0-9']+", message.lower())
                 if len(w) > 3}
        if not words:
            return True
        have = set(re.findall(r"[a-z0-9']+", response_text.lower()))
        return len(words & have) / len(words) >= 0.3

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
        
        # Add time context in the configured local timezone (the container
        # clock may be UTC; a naive now() here told callers UTC times).
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo(self.config.local_timezone))
        except Exception:
            now = datetime.now()
        prompt += f"\n\nCurrent time: {now.strftime('%I:%M %p %Z on %A, %B %d, %Y')}"

        # Static home/base address so "here", "home", and directions questions
        # have a fixed reference point. The MAP tool routes from this location.
        location = getattr(self.config, "agent_location", "") or ""
        if location.strip():
            prompt += (
                f"\n\nYour location (where \"here\" and \"home\" are): "
                f"{location.strip()}")

        # Caller-chosen demeanor for this call, layered over the base prompt.
        # Placed right after the base persona and before the call facts so it
        # colors the whole reply, but it can only shape TONE — the base prompt's
        # rules (grounding, tool use, safety) still stand.
        if call_context:
            persona = call_context.get("persona")
            if persona:
                prompt += (
                    "\n\nFor this call, adopt the following demeanor and speaking"
                    " style. It changes HOW you speak, not what you're allowed to"
                    " do — keep following all instructions above:\n" + persona)

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

            # This call arrived on a temporary virtual number provisioned for
            # a specific purpose — the caller dialed in expecting exactly this.
            virtual_purpose = call_context.get("virtual_number_context")
            if virtual_purpose:
                prompt += (
                    "\n\nThis call came in on a temporary number set up for a"
                    " specific purpose. Handle the call with that purpose in"
                    " mind:\n" + virtual_purpose)

            # Identity verification: some actions require a verified caller. Tell
            # the model where this caller stands so it routes through the VERIFY
            # tool before a gated action rather than refusing or guessing.
            if call_context.get("verification_required"):
                if call_context.get("verified"):
                    prompt += (
                        "\n\nThe caller has verified their identity on this call;"
                        " you may proceed with sensitive actions.")
                else:
                    prompt += (
                        "\n\nThe caller has NOT verified their identity. Before any"
                        " sensitive or restricted action, verify them using the"
                        " VERIFY tool (they enter a PIN or one-time code on the"
                        " keypad — do not ask them to say it aloud).")


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
            # Tools may carry a full JSON schema (e.g. an MCP inputSchema);
            # use it verbatim so native/langgraph modes get real nested
            # schemas. Otherwise synthesize one from the flat parameters dict.
            json_schema = getattr(tool, "json_schema", None)
            if json_schema:
                parameters = json_schema
            else:
                properties: Dict[str, Any] = {}
                required: List[str] = []
                for pname, spec in (getattr(tool, "parameters", {}) or {}).items():
                    prop: Dict[str, Any] = {"type": spec.get("type", "string")}
                    if spec.get("description"):
                        prop["description"] = spec["description"]
                    properties[pname] = prop
                    if spec.get("required"):
                        required.append(pname)
                parameters = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": getattr(tool, "description", ""),
                    "parameters": parameters,
                },
            })
        return tools

    async def _generate_native(self, messages: List[Dict[str, Any]],
                               ctx: TurnContext,
                               first_message: Optional[Any] = None) -> str:
        """Native function-calling loop (LLM_TOOL_CALLING=native).

        Sends tool schemas via the OpenAI `tools` param, executes any returned
        tool_calls through the ToolManager, feeds results back as `tool`
        messages, and repeats until the model answers in plain text.

        ``first_message`` lets the streaming path hand over a pre-consumed
        round-0 assistant message (reconstructed from stream deltas): the
        first loop iteration dispatches it instead of making a request, so
        the round budget and grounding behavior stay identical.
        """
        tools = self._build_native_tools()
        messages = list(messages)
        # Bound on tool-call round trips per turn so a model that keeps asking
        # for tools can't loop forever on a live phone call. The budget buys
        # max_rounds *tool* rounds, plus one final completion to turn the last
        # tool result into a spoken answer — that final call withholds the
        # tools param so the model has to answer rather than ask again.
        max_rounds = self.config.llm_max_tool_rounds

        last_user = next((m.get("content") or "" for m in reversed(messages)
                          if m.get("role") == "user"), "")
        grounding_retried = False

        with create_span("llm.generate_native", {
            "llm.model": self.config.llm_model,
            "llm.tools_count": len(tools),
        }) as span:
            try:
                for round_no in range(max_rounds + 1):
                    budget_spent = round_no == max_rounds
                    if first_message is not None:
                        # Round 0 was already consumed by the streaming path.
                        msg, first_message = first_message, None
                    else:
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

                    # "Never guess": a first-round zero-tool answer to a
                    # live-data question (or a promise to check) gets ONE
                    # forced-tool retry; the replacement msg falls through to
                    # the normal dispatch below.
                    if (not tool_calls and round_no == 0 and tools
                            and not grounding_retried
                            and self.config.grounding_retry_enabled):
                        category = grounding.grounding_category(
                            str(last_user), msg.content or "")
                        if category and self._category_tools_available(category):
                            grounding_retried = True
                            forced, outcome = await self._grounding_force_native(
                                messages, tools)
                            log_event(logger, logging.INFO,
                                      f"Grounding retry ({category}): {outcome}",
                                      event="grounding_retry",
                                      category=category, outcome=outcome)
                            span.set_attribute("llm.grounding_retry", outcome)
                            if forced is not None and getattr(forced, "tool_calls", None):
                                msg = forced
                                tool_calls = forced.tool_calls
                            elif outcome == "unsupported":
                                # Backend rejected tool_choice: nudge and loop.
                                messages.append({
                                    "role": "system",
                                    "content": grounding.NUDGE})
                                continue
                            # timeout/no_tool_call/error: keep the original
                            # reply rather than stacking more latency.

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
                            self._collect_spoken_result(
                                ctx, tc.function.name, result)
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
        
    def _category_tools_available(self, category: str) -> bool:
        """Is at least one tool that can answer this category loaded?"""
        names = grounding.CATEGORY_TOOLS.get(category, ())
        if not names:  # PROMISED_ACTION: any loaded tool qualifies
            return bool(self.tool_manager.tools)
        return any(n in self.tool_manager.tools for n in names)

    async def _grounding_force_native(self, messages: List[Dict[str, Any]],
                                      tools: List[Dict[str, Any]]):
        """One completion with tool_choice="required".

        Returns (message, outcome): outcome "tool_used"/"no_tool_call" with a
        message, or (None, "unsupported"/"timeout"/"error").
        """
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    messages=messages,
                    tools=tools,
                    tool_choice="required",
                    **self._sampling_kwargs(),
                ),
                timeout=self.config.grounding_retry_timeout_s)
        except asyncio.TimeoutError:
            logger.warning("Grounding retry timed out (native)")
            return None, "timeout"
        except Exception as e:
            if "400" in str(e) or "BadRequest" in type(e).__name__:
                logger.warning(f"tool_choice=required rejected by backend: {e}")
                return None, "unsupported"
            logger.error(f"Grounding retry error (native): {e}")
            return None, "error"
        msg = response.choices[0].message
        has_calls = bool(getattr(msg, "tool_calls", None))
        return msg, ("tool_used" if has_calls else "no_tool_call")

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
        # The first alternative tolerates one level of balanced [...] inside a
        # value (JSON arrays in "JSON object as a string" params, e.g. MCP
        # tools); the second is the legacy fallback so anything else behaves
        # exactly as before.
        pattern_with_params = r'\[TOOL:(\w+):((?:[^\[\]]|\[[^\[\]]*\])+|[^\]]+)\]'
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

    def _streaming_supported(self) -> bool:
        # _generate goes through the Ollama HTTP API, not the OpenAI client;
        # stream_response uses the default generate_response replay path.
        return False

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
