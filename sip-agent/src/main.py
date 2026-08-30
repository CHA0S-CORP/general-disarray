#!/usr/bin/env python3
"""
SIP AI Assistant - API-based Architecture
==========================================
All ML inference offloaded to dedicated services:
- Speaches API for STT (Whisper) and TTS (Piper/Kokoro)
- vLLM for LLM

This container is lightweight - just orchestration.
"""

import json
import os
import time
import random
import signal
import asyncio
import logging
import ipaddress
from typing import Any, Dict, List, Optional, Tuple

import call_events
import context_manager
import endpointing
from admin_events import EventBus
from call_session import (CallSession, TurnLedger, spoken_text,
                          set_current_session, get_current_session)
from caller_memory import CallerMemoryStore, caller_id_from_uri
from persona_store import PersonaStore
from identity_verification import IdentityVerifier, VerificationStore
from virtual_numbers import VirtualNumberRegistry, extension_from_uri
from earcons import generate_chime, generate_thinking_tick
from knowledge_base import KnowledgeBase
from mcp_tools import MCPManager
from sip_handler import SIPHandler
from tool_manager import ToolManager
from transcript_store import TranscriptStore
from config import Config, get_config
from llm_engine import create_llm_engine
from sentence_stream import split_into_sentences  # noqa: F401 (re-export; also used below)
from audio_pipeline import LowLatencyAudioPipeline
from logging_utils import log_event, HANGUP_DELAY_SECONDS
from speech_text import is_farewell

# How often to poll the playlist player while waiting for enqueued playback
# to drain (seconds). send_audio()/enqueue_file() are non-blocking, so _speak
# returns when the last chunk is merely ENQUEUED — seconds before the caller
# has heard it. The turn parks (cancellably) in _wait_for_playback_drain
# until the player runs dry; a slightly stale poll is fine, so this is a
# module constant, not deployment configuration.
PLAYBACK_DRAIN_POLL_S = 0.1


class _SpeechRun:
    """Gap-tolerant accumulator of speech duration, for the barge-in and
    cancel-merge debounces.

    Counts VAD-positive audio (ms) toward ``min_ms``, tolerating gaps of up to
    ``max_gap_ms`` INSIDE the run. The tolerance is the whole point: real
    speech is not VAD-positive end to end — pauses between syllables and
    unvoiced consonants read as silence — so a counter that resets on the
    first negative chunk needs the caller to talk for several times
    barge_in_min_duration_ms before it ever trips, and the finer the chunks,
    the worse it gets (measured on real TTS speech: firing at 500ms with 100ms
    chunks, but 1660ms with 20ms chunks). Tolerating the gaps makes the gate
    mean what it says — "the caller has been talking for min_ms" — at any
    chunk size.

    update() reports whether the gate is met; the caller resets after acting.
    """

    def __init__(self, min_ms: float, max_gap_ms: float):
        self.min_ms = min_ms
        self.max_gap_ms = max_gap_ms
        self.speech_ms = 0.0
        self.gap_ms = 0.0

    def update(self, chunk_ms: float, is_speech: bool) -> bool:
        """Feed one chunk. True once the run has reached ``min_ms``."""
        if is_speech:
            self.speech_ms += chunk_ms
            self.gap_ms = 0.0
        elif self.speech_ms > 0.0:
            # Only gaps INSIDE a run are tracked; silence before any speech
            # leaves the run at zero and costs nothing.
            self.gap_ms += chunk_ms
            if self.gap_ms > self.max_gap_ms:
                self.reset()
        return self.speech_ms >= self.min_ms

    def reset(self) -> None:
        self.speech_ms = 0.0
        self.gap_ms = 0.0


# Initialize OpenTelemetry early (before other modules)
from telemetry import init_telemetry, is_enabled as otel_enabled, TraceContextFilter, Metrics, get_otel_log_handler
init_telemetry("sip-agent")

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record):
        log_data = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        
        # Add trace context if available (from OpenTelemetry)
        if hasattr(record, 'trace_id'):
            log_data['trace_id'] = record.trace_id
        if hasattr(record, 'span_id'):
            log_data['span_id'] = record.span_id
        
        # Add extra fields if present (set by log_event)
        if hasattr(record, 'event_type'):
            log_data['event'] = record.event_type
        if hasattr(record, 'event_data') and record.event_data:
            log_data['data'] = record.event_data
            
        # Add exception info if present
        if record.exc_info:
            log_data['exc'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


# Configure JSON logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
# handler.addFilter(TraceContextFilter())  # Add trace context to logs
otel_handler = get_otel_log_handler()
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler, otel_handler],
    force=True  # Override any existing config
)

# Reduce noise from libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# split_into_sentences (the sentence splitter feeding streaming TTS) now
# lives in sentence_stream.py — shared with the LLM engine's token-streaming
# assembler — and is re-exported above for compatibility.


class SIPAIAssistant:
    """
    SIP AI Assistant with API-based ML inference.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        
        logger.info("Initializing SIP AI Assistant...")
        
        # Cross-call caller memory (data/caller_memory) and the RAG knowledge
        # base (data/knowledge). Constructed before the ToolManager so the
        # KNOWLEDGE tool can see the knowledge base when tools load. Both are
        # fail-open no-ops when disabled or their optional deps are missing.
        self.caller_memory = CallerMemoryStore(config)
        self.knowledge_base = KnowledgeBase(config)
        self.virtual_numbers = VirtualNumberRegistry(config)
        # Named demeanor profiles (data/personas.json), for the PERSONA tool's
        # save/load. The active persona lives on the CallSession; this is just
        # persistence. Fail-open.
        self.persona_store = PersonaStore(config)
        # Optional caller identity verification (static PIN + rolling TOTP over
        # DTMF). The store persists per-caller credentials; the verifier holds
        # the check logic (with a global config fallback). Shared by the VERIFY
        # tool and the /verify REST endpoints. Fail-open.
        self.verify_store = VerificationStore(config)
        self.verifier = IdentityVerifier(config, self.verify_store)
        # MCP client (external tool servers). Connected inside
        # ToolManager.start() so MCP tools register through the same
        # wrapper path as plugins. No-op unless MCP_ENABLED.
        self.mcp_manager = MCPManager(config)

        # Core components
        self.tool_manager = ToolManager(self)
        self.llm_engine = create_llm_engine(config, self.tool_manager)
        self.audio_pipeline = LowLatencyAudioPipeline(config)
        self.sip_handler = SIPHandler(config, self._on_call_received)
        
        # State: everything owned by a live call (history, turn task, held
        # transcripts, the audio loop) lives on its CallSession, so a stale
        # task from a replaced call can never write into the next call's
        # conversation. Sessions are registered here keyed by the call's SIP
        # call-id (transcript id when unavailable); up to
        # config.max_concurrent_calls entries (default 1). _call_lock guards
        # registry mutation only.
        self.sessions: Dict[str, CallSession] = {}
        self._call_lock = asyncio.Lock()

        # Per-call conversation transcripts (bounded memory + data/transcripts).
        self.transcripts = TranscriptStore(config)
        self._session_counter = 0

        # In-process event bus feeding the admin dashboard's SSE stream
        # (publish is a no-op with no subscribers; a full subscriber drops
        # its oldest events, so it can never block the call path).
        self.events = EventBus()

        # In-flight call-event webhook tasks (fire-and-forget; held so the
        # event loop doesn't garbage-collect them mid-delivery).
        self._event_tasks: set = set()
        
        # Pre-cached phrases for instant playback - loaded from config
        self.thinking_phrases = self.config.phrases.thinking
        self.greeting_phrases = self.config.phrases.greetings
        self.goodbye_phrases = self.config.phrases.goodbyes
        self.error_phrases = self.config.phrases.errors
        self.followup_phrases = self.config.phrases.followups
        
        # Combined list for pre-caching
        self._phrases_to_cache = self.config.phrases.get_all_phrases_for_cache()
        # The VERIFY tool's keypad prompt is a fixed phrase too: pre-cache it
        # so it plays instantly instead of being synthesized mid-turn.
        if getattr(self.config, "enable_verify_tool", False):
            self._phrases_to_cache = list(dict.fromkeys(
                self._phrases_to_cache + [self.config.verify_call_prompt]))

        # Confirmation earcon (in-memory PCM), generated once; played via the
        # same send_audio path as TTS so barge-in/flush semantics are identical.
        self._chime_pcm: bytes = generate_chime(
            sample_rate=config.sample_rate, volume=config.chime_volume)
        # Softer periodic tick played while a slow LLM turn is in flight.
        self._thinking_pcm: bytes = generate_thinking_tick(
            sample_rate=config.sample_rate, volume=config.chime_volume)

    @property
    def session(self) -> Optional[CallSession]:
        """The session the current task acts on behalf of, else the sole one.

        Tasks bound to one call (set_current_session at the top of the task
        body: response turns, the audio loop, outbound interactive sessions,
        drain goodbyes) resolve their own session even with several calls
        live — but only for as long as it is still registered, so a stale
        task from a replaced/ended call sees None and can never touch a live
        call's state. Unbound callers (REST handlers, the scheduler) get the
        sole active session when unambiguous; with zero — or more than one —
        registered sessions they get None.
        """
        bound = get_current_session()
        if bound is not None:
            for existing in self.sessions.values():
                if existing is bound:
                    return bound
            return None
        if len(self.sessions) == 1:
            return next(iter(self.sessions.values()))
        return None

    @staticmethod
    def _session_key(session: CallSession) -> str:
        """Registry key for a session: the SIP call-id when the CallInfo has
        one, else the (unique) transcript id."""
        return (getattr(session.call_info, "call_id", "")
                or session.transcript_id)

    def _detach_session(self, session: CallSession) -> None:
        """Remove exactly this session's registry entry, if still present.

        Pure registry op (no task cancellation) — callers hold _call_lock.
        A session already replaced by a newer call is simply absent, so this
        can never clobber another session's entry.
        """
        for key, existing in list(self.sessions.items()):
            if existing is session:
                del self.sessions[key]
                break

    @property
    def current_call(self):
        """The live call's CallInfo (None when idle).

        Read-only compat shim for external callers (REST /speak, timer tool);
        per-call state lives on self.session.
        """
        return self.session.call_info if self.session else None

    @property
    def conversation_history(self) -> List[Dict]:
        """The live call's conversation history (empty when idle)."""
        return self.session.conversation_history if self.session else []

    def _publish_admin_event(self, event: str, data: Optional[Dict] = None,
                             session: Optional[CallSession] = None) -> None:
        """Publish to the in-process admin event bus. Never raises into the
        call path; call_id is the session's transcript id ("-" when idle)."""
        try:
            sess = session or self.session
            self.events.publish(
                event, sess.transcript_id if sess else "-", data or {})
        except Exception as e:
            logger.debug(f"Admin event publish failed: {e}")

    def _publish_call_ended(self, session: CallSession) -> None:
        """Publish call.ended to the admin bus exactly once per session (the
        audio-loop tail and _teardown_session both reach this point)."""
        if session.admin_ended_published:
            return
        session.admin_ended_published = True
        self._publish_admin_event(
            "call.ended",
            {"direction": session.direction,
             "duration_seconds": round(time.time() - session.start_time, 1)},
            session)

    def _begin_session(self, call_info, direction: str, remote: str,
                       virtual_number=None) -> CallSession:
        """Create and install a new active session (caller holds _call_lock)."""
        prefix = "in" if direction == "inbound" else "out"
        self._session_counter += 1
        session = CallSession(
            call_info=call_info,
            direction=direction,
            # Counter suffix keeps ids unique for calls in the same second.
            transcript_id=f"{prefix}-{int(time.time())}-{self._session_counter}",
            virtual_number=virtual_number,
        )
        # Per-call audio state (VAD + utterance buffer + latency metrics).
        session.audio_state = self.audio_pipeline.new_session_state()
        self.sessions[self._session_key(session)] = session
        self.transcripts.start(session.transcript_id, direction, remote)
        # Load what we remember about this caller (one small disk read).
        if self.config.caller_memory_enabled:
            try:
                caller_id = caller_id_from_uri(remote)
                if caller_id:
                    session.caller_id = caller_id
                    session.caller_memory_prompt = (
                        self.caller_memory.format_for_prompt(caller_id))
            except Exception as e:
                logger.warning(f"Could not load caller memory: {e}")
        self._emit_call_event("call.started", session)
        self._publish_admin_event(
            "call.started", {"direction": direction, "remote": remote}, session)
        return session

    async def _teardown_session(self, session: Optional[CallSession] = None):
        """Stop one session's tasks, remove exactly its registry entry, and
        close its transcript.

        With no argument, tears down every registered session (used by
        stop()). Detaches the session from the registry first so its tasks
        observe `self.session is not them` and cannot touch a replacement
        session's state.
        """
        if session is None:
            for existing in list(self.sessions.values()):
                await self._teardown_session(existing)
            return
        self._detach_session(session)
        if session.audio_loop_task and not session.audio_loop_task.done():
            session.audio_loop_task.cancel()
            try:
                await session.audio_loop_task
            except asyncio.CancelledError:
                pass
        await self._cancel_turn(session)
        # Close this session's realtime STT connection, if any (idempotent;
        # no-op in batch mode).
        try:
            await self.audio_pipeline.stop_session_stt(session.audio_state)
        except Exception as e:
            logger.debug(f"Session STT teardown failed: {e}")
        self.transcripts.end(session.transcript_id)
        self._emit_call_event("call.ended", session)
        self._publish_call_ended(session)
        self._finalize_virtual_number(session)
        self._start_memory_update(session)

    def _start_memory_update(self, session: CallSession) -> None:
        """Fire-and-forget the post-call caller-memory update. Both teardown
        paths (forced teardown and the audio-loop tail) reach here; the
        session flag makes it run once."""
        if not self.config.caller_memory_enabled or session.memory_update_started:
            return
        session.memory_update_started = True
        remote_uri = getattr(session.call_info, "remote_uri", "") or ""
        transcript = self.transcripts.get(session.transcript_id)
        if not remote_uri or transcript is None:
            return
        task = asyncio.create_task(
            self.caller_memory.update_from_call(
                remote_uri, transcript, self.llm_engine))
        self._event_tasks.add(task)
        task.add_done_callback(self._event_tasks.discard)

    def _emit_call_event(self, event: str, session: CallSession) -> None:
        """Fire-and-forget a call-lifecycle webhook. Never blocks call setup
        or teardown; deliver_webhook never raises and retries internally."""
        url = self.config.call_event_webhook_url
        if not url or event not in call_events.enabled_events(self.config):
            return
        if event == "call.ended":
            # Both the audio-loop tail and _teardown_session reach this point.
            if session.ended_event_emitted:
                return
            session.ended_event_emitted = True
        payload = call_events.build_call_event_payload(
            event, session,
            self.transcripts.get(session.transcript_id),
            self.config.call_event_include_transcript)
        log_event(logger, logging.INFO,
                  f"Emitting {event} for {session.transcript_id}",
                  event="call_event", call_event=event,
                  call_id=session.transcript_id)
        from api import deliver_webhook  # lazy, mirrors the create_api import
        task = asyncio.create_task(
            deliver_webhook(url, payload, self.config,
                            api_name="call_event_webhook"))
        self._event_tasks.add(task)
        task.add_done_callback(self._event_tasks.discard)

    def _finalize_virtual_number(self, session: CallSession) -> None:
        """Complete a virtual number after its call: fire the result webhook
        and consume the entry (single-use). Both call.ended sites (teardown
        and the audio-loop tail) reach here; the session flag makes it run
        once. Never blocks teardown."""
        entry = session.virtual_number
        if entry is None or session.virtual_number_finalized:
            return
        session.virtual_number_finalized = True

        consumed = self.virtual_numbers.consume(entry.id)
        log_event(logger, logging.INFO,
                  f"Virtual number completed: {entry.number}",
                  event="virtual_number_completed", number=entry.number,
                  virtual_number_id=entry.id, consumed=consumed is not None,
                  persistent=entry.persistent)

        if not entry.wants("completed"):
            return
        extra = self._virtual_number_call_fields(session)
        extra["duration_seconds"] = round(time.time() - session.start_time, 1)
        if entry.include_transcript:
            transcript = self.transcripts.get(session.transcript_id)
            if transcript is not None:
                extra["transcript"] = transcript
        self.virtual_numbers.fire_webhook(entry, status="completed", extra=extra)

    @staticmethod
    def _virtual_number_call_fields(session: CallSession) -> Dict[str, Any]:
        """Per-call fields shared by every virtual-number webhook."""
        return {
            "caller": getattr(session.call_info, "remote_uri", "") or "",
            "call_id": session.transcript_id,
        }

    def _emit_virtual_number_speech(self, session: CallSession, text: str) -> None:
        """Trigger-number speech hooks: `first_speech` fires once per call,
        `speech` on every utterance. Fire-and-forget; never on the speaking
        path's critical section."""
        entry = session.virtual_number
        if entry is None:
            return
        session.virtual_number_speech_count += 1
        first = session.virtual_number_speech_count == 1
        if not ((first and entry.wants("first_speech")) or entry.wants("speech")):
            return
        extra = self._virtual_number_call_fields(session)
        extra.update({
            "text": text,
            "utterance_index": session.virtual_number_speech_count,
            "first": first,
        })
        self.virtual_numbers.fire_webhook(
            entry, status="first_speech" if first else "speech", extra=extra)

    async def start(self):
        """Start all components and run main loop."""
        await self.start_components()
        
        # Keep running
        while self.running:
            await asyncio.sleep(1)
            
    async def start_components(self):
        """Start all components (without main loop)."""
        log_event(logger, logging.INFO, "Starting SIP AI Assistant...",
                 event="warming_up", phase="init")
        self.running = True
        
        # Start components
        log_event(logger, logging.INFO, "Starting LLM engine...",
                 event="warming_up", phase="llm")
        await self.llm_engine.start()
        
        log_event(logger, logging.INFO, "Starting audio pipeline...",
                 event="warming_up", phase="audio")
        await self.audio_pipeline.start()

        # Index the knowledge base in the background; calls can start (and
        # the KNOWLEDGE tool reports "still indexing") while it builds.
        if self.config.knowledge_enabled:
            task = asyncio.create_task(self.knowledge_base.start())
            self._event_tasks.add(task)
            task.add_done_callback(self._event_tasks.discard)
        
        # Pre-cache common phrases
        log_event(logger, logging.INFO, "Pre-caching TTS phrases...",
                 event="warming_up", phase="tts_cache")
        await self._precache_phrases()
        
        log_event(logger, logging.INFO, "Starting SIP handler...",
                 event="warming_up", phase="sip")
        await self.sip_handler.start()
        
        log_event(logger, logging.INFO, "Starting tool manager...",
                 event="warming_up", phase="tools")
        await self.tool_manager.start()

        # Virtual-number registry (no-op unless VIRTUAL_NUMBERS_ENABLED).
        await self.virtual_numbers.start()
        
        sip_uri = f"sip:{self.config.sip_user}@{self.config.sip_domain}"
        log_event(logger, logging.INFO, f"SIP AI Assistant ready! URI: {sip_uri}",
                 event="ready", sip_uri=sip_uri)
    
    async def run_loop(self, shutdown_event: asyncio.Event = None):
        """Run main loop until shutdown."""
        try:
            while self.running:
                if shutdown_event and shutdown_event.is_set():
                    break
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
            
    async def drain_active_call(self, turn_timeout: float = 15.0):
        """Wind down every active call gracefully before shutdown.

        Each active session drains concurrently in its own task (bounded per
        call): the in-flight response turn may finish (``turn_timeout``), a
        goodbye is spoken so the caller isn't cut off mid-sentence, then the
        call hangs up cleanly. No-op when idle.
        """
        sessions = [
            s for s in self.sessions.values()
            if s.call_info and getattr(s.call_info, 'is_active', False)]
        if not sessions:
            return

        async def _drain_one(session: CallSession):
            # Top of the task body: everything below (goodbye TTS, playback,
            # hangup) acts on behalf of this call.
            set_current_session(session)
            call = session.call_info
            logger.info("Draining active call before shutdown")
            task = session.turn_task
            if task and not task.done():
                try:
                    # shield: a timeout should move on to the goodbye, not
                    # leave a half-cancelled turn behind (_cancel_turn
                    # finishes the job).
                    await asyncio.wait_for(asyncio.shield(task), turn_timeout)
                except (asyncio.TimeoutError, Exception):
                    await self._cancel_turn(session)

            try:
                goodbye = self.get_random_goodbye()
                log_event(logger, logging.INFO, f"Assistant: {goodbye}",
                         event="assistant_response", text=goodbye)
                await self._speak(goodbye)
                # Goodbyes are pre-cached and short; give playback a moment.
                await asyncio.sleep(2.0)
            except Exception as e:
                logger.warning(f"Failed to speak goodbye during drain: {e}")

            try:
                await self.sip_handler.hangup_call(call)
            except Exception as e:
                logger.warning(f"Hangup during drain failed: {e}")

        # Per-call bound: a wedged call can delay shutdown by at most the
        # turn wait plus the goodbye; the others drain in parallel.
        per_call_timeout = turn_timeout + 15.0
        await asyncio.gather(
            *(asyncio.wait_for(_drain_one(s), per_call_timeout)
              for s in sessions),
            return_exceptions=True)

    async def stop(self):
        """Stop all components."""
        logger.info("Stopping...")
        self.running = False

        # Stop all registered sessions (audio loop + in-flight turn + transcript)
        await self._teardown_session()

        # Give in-flight call-event webhooks a moment to finish delivery.
        if self._event_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._event_tasks, return_exceptions=True),
                    timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Call-event webhook delivery still pending at shutdown")

        await self.virtual_numbers.stop()
        await self.tool_manager.stop()
        await self.mcp_manager.stop()
        await self.sip_handler.stop()
        await self.audio_pipeline.stop()
        await self.llm_engine.stop()
        
        logger.info("Stopped.")
        
    async def _precache_phrases(self):
        """Pre-generate audio for common phrases concurrently."""
        logger.info(f"Pre-caching {len(self._phrases_to_cache)} phrases...")
        
        # Use semaphore to limit concurrent TTS requests
        MAX_CONCURRENT_TTS = 5
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TTS)
        
        async def cache_phrase(phrase: str) -> bool:
            async with semaphore:
                try:
                    audio = await self.audio_pipeline.synthesize(phrase)
                    return audio is not None
                except Exception as e:
                    logger.warning(f"Failed to cache '{phrase}': {e}")
                    return False
        
        # Run all precaching concurrently with limited concurrency
        results = await asyncio.gather(
            *[cache_phrase(phrase) for phrase in self._phrases_to_cache],
            return_exceptions=True
        )
        
        cached = sum(1 for r in results if r is True)
        logger.info(f"Pre-cached {cached}/{len(self._phrases_to_cache)} phrases")
        
    def get_random_thinking(self) -> str:
        """Get a random thinking/processing phrase."""
        return random.choice(self.thinking_phrases)
        
    def get_random_greeting(self) -> str:
        """Get a random greeting phrase."""
        return random.choice(self.greeting_phrases)
        
    def get_random_goodbye(self) -> str:
        """Get a random goodbye phrase."""
        return random.choice(self.goodbye_phrases)
        
    def get_random_error(self) -> str:
        """Get a random error/retry phrase."""
        return random.choice(self.error_phrases)
        
    def get_random_followup(self) -> str:
        """Get a random follow-up phrase."""
        return random.choice(self.followup_phrases)
        
    async def _on_call_received(self, call_info):
        """Handle incoming call."""
        try:
            remote_uri = getattr(call_info, 'remote_uri', 'unknown')
            log_event(logger, logging.INFO, f"Call received from: {remote_uri}",
                     event="call_start", caller=remote_uri, direction="inbound")

            # Record call started metric
            Metrics.record_call_started("inbound")

            # The lock guards registry mutation only (greeting playback and
            # the audio loop run outside it, so a second call can be admitted
            # while the first is still greeting).
            async with self._call_lock:
                # Duplicate-INVITE suppression keys on the SIP call id: a
                # second REAL call while another is live is a new session,
                # not a duplicate.
                dup_key = getattr(call_info, 'call_id', '') or ''
                if dup_key and dup_key in self.sessions:
                    logger.warning(
                        f"Call {dup_key} already has a session, "
                        "ignoring duplicate callback")
                    return

                # Capacity: at the cap, evict a session. Prefer one whose SIP
                # call has already ended — PJSIP drops a hung-up call from
                # active_calls immediately, but its session stays registered
                # until its audio loop notices (>=50ms poll), and in that
                # window _busy() (which counts PJSIP-alive calls) can admit a
                # new INVITE. Blindly evicting the oldest registry entry here
                # would tear down a LIVE call while the dead session lingered.
                # Only when every registered session is still live fall back
                # to replacing the oldest (the pre-existing
                # SIP_BUSY_REJECT=false replace-the-call semantics; with the
                # busy gate on this is only reachable in a tight INVITE race).
                while len(self.sessions) >= max(
                        1, self.config.max_concurrent_calls):
                    victim = next(
                        (s for s in self.sessions.values()
                         if not getattr(s.call_info, 'is_active', False)),
                        None)
                    if victim is None:
                        victim = next(iter(self.sessions.values()))
                    await self._teardown_session(victim)

                # Match the dialed extension against the virtual-number
                # registry (the PJSIP thread only captured the URI string).
                virtual_number = None
                dialed = extension_from_uri(getattr(call_info, 'local_uri', '') or '')
                if dialed:
                    virtual_number = self.virtual_numbers.claim(dialed)
                    if virtual_number:
                        log_event(logger, logging.INFO,
                                 f"Call matched virtual number {virtual_number.number}",
                                 event="virtual_number_matched",
                                 number=virtual_number.number,
                                 virtual_number_id=virtual_number.id)

                session = self._begin_session(call_info, "inbound", remote_uri,
                                              virtual_number=virtual_number)
                if virtual_number and virtual_number.wants("answered"):
                    # Trigger-number hook: kick the workflow off the moment
                    # the call is matched, before the greeting plays.
                    self.virtual_numbers.fire_webhook(
                        virtual_number, status="answered",
                        extra=self._virtual_number_call_fields(session))

            # Everything below acts on behalf of the new call: bind it so
            # greeting playback (and anything else reaching self.session /
            # current_call) resolves this session even with other calls live.
            set_current_session(session)

            # Per-session realtime STT connection (no-op in batch mode).
            await self.audio_pipeline.start_session_stt(session.audio_state)

            # Play greeting
            await self._play_greeting(session)

            # Start listening (single task per session)
            logger.info("Listening...")
            session.audio_loop_task = asyncio.create_task(
                self._audio_processing_loop(session))
        except Exception as e:
            logger.error(f"Error handling call: {e}", exc_info=True)
        
    async def _play_greeting(self, session: CallSession):
        """Play initial greeting (uses pre-cached audio)."""
        greeting = self.get_random_greeting()
        # A virtual number may carry its own greeting (not pre-cached; one
        # TTS round-trip is acceptable for these provisioned calls).
        if session.virtual_number and session.virtual_number.greeting:
            greeting = session.virtual_number.greeting

        try:
            logger.info(f"Playing greeting: {greeting}")
            # This should hit the cache since we pre-cached it
            audio = await self.audio_pipeline.synthesize(greeting)
            if audio:
                await self._play_audio(audio)
                self.transcripts.add_turn(session.transcript_id, "assistant", greeting)
                self._publish_admin_event(
                    "assistant_turn", {"text": greeting}, session)
        except Exception as e:
            logger.error(f"Error playing greeting: {e}")
            
    async def _audio_processing_loop(self, session: CallSession):
        """Main audio processing loop for one call session."""
        # Task body top: this loop (and every task it spawns — response
        # turns, tickers) acts on behalf of exactly this call.
        set_current_session(session)
        logger.info("Audio processing loop started")

        audio_received_count = 0
        last_log_time = time.time()
        # Speech (ms) heard while the assistant is speaking; a barge-in only
        # triggers once this reaches barge_in_min_duration_ms, so clicks/pops/
        # short noise bursts can't cancel an in-flight turn.
        barge_in_run = _SpeechRun(self.config.barge_in_min_duration_ms,
                                  self.config.barge_in_max_gap_ms)
        # Same debounce for the speculative cancel-merge branch (speech while
        # the LLM is thinking, nothing playing): cancelling an in-flight turn
        # is at least as destructive as a barge-in, so a single VAD-positive
        # chunk of line noise must not trigger it either.
        cancel_merge_run = _SpeechRun(self.config.barge_in_min_duration_ms,
                                      self.config.barge_in_max_gap_ms)

        # `self.session is session` guards against a replaced session: once a
        # new call takes over, this loop exits instead of touching its state.
        while self.running and self.session is session:
            try:
                # Check call state
                if not getattr(session.call_info, 'is_active', False):
                    log_event(logger, logging.INFO, "Call ended, stopping audio loop",
                             event="call_end")
                    # Record call metrics (outbound sessions record theirs in
                    # make_outbound_call, matching the previous behavior)
                    if session.direction == "inbound":
                        duration_ms = (time.time() - session.start_time) * 1000
                        Metrics.record_call_duration(duration_ms, "inbound")
                        Metrics.record_call_ended("inbound", "completed")
                    break

                # Wait for media to be ready
                if not getattr(session.call_info, 'media_ready', False):
                    await asyncio.sleep(0.1)
                    continue

                # Dispatch speech that was held while the previous turn was
                # still in flight, now that the turn has finished.
                if session.pending_transcription and (
                        session.turn_task is None or session.turn_task.done()):
                    pending = session.pending_transcription
                    session.pending_transcription = None
                    self._dispatch_turn(session, pending)

                # Speculative endpointing: an incomplete-looking fragment is
                # never held forever — once ENDPOINT_MAX_SILENCE_MS has
                # passed with no follow-up speech (and the VAD isn't mid-
                # collection of new speech), dispatch it as-is.
                if (session.held_fragment
                        and self.config.endpoint_mode == "speculative"
                        and (session.turn_task is None
                             or session.turn_task.done())
                        and not session.audio_state.vad.is_speaking
                        and (time.monotonic() - session.held_fragment_at) * 1000
                            >= self.config.endpoint_max_silence_ms):
                    held = session.held_fragment
                    session.held_fragment = None
                    log_event(logger, logging.INFO,
                              f"Held fragment dispatched at deadline: {held}",
                              event="endpoint_fragment_deadline", text=held)
                    self._dispatch_turn(session, held)

                audio_chunk = None
                try:
                    # Try to receive audio. The short timeout is the idle poll
                    # interval — it also sets the steady-state chunk size (the
                    # audio accrued since the last read), and hence how finely
                    # barge-in and end-of-utterance can be resolved.
                    audio_chunk = await self.sip_handler.receive_audio(
                        session.call_info,
                        timeout=0.02
                    )

                    if audio_chunk:
                        audio_received_count += 1

                        # Log periodically (debug level - not interesting for filtering)
                        if time.time() - last_log_time > 5:
                            logger.debug(f"Audio chunks received: {audio_received_count}")
                            last_log_time = time.time()

                        # Check for barge-in — only while the assistant is
                        # audibly speaking. Speech while the LLM is still
                        # thinking (nothing playing) is a follow-up, not an
                        # interruption: it must NOT cancel the in-flight turn,
                        # or a slow LLM could be starved by a talkative caller;
                        # it is held via pending_transcription instead.
                        # Exception (speculative endpointing): a turn that was
                        # dispatched at the short threshold and has produced NO
                        # audio yet is cancelled when the caller resumes, and
                        # its text is merged with the new speech (cancel-merge
                        # — a third branch, distinct from both barge-in and
                        # pending_transcription).
                        chunk_ms = (
                            len(audio_chunk) / 2 / self.config.sample_rate * 1000)
                        speech = self.audio_pipeline.has_speech(
                            session.audio_state, audio_chunk)

                        if session.dtmf_collecting:
                            # A tool is collecting a keypad code: the caller's
                            # own DTMF tones / an "okay" must not cancel the
                            # turn. (Digits mute the prompt themselves.)
                            barge_in_run.reset()
                            cancel_merge_run.reset()
                        elif self._playback_active(session):
                            cancel_merge_run.reset()
                            if barge_in_run.update(chunk_ms, speech):
                                barge_in_run.reset()
                                log_event(logger, logging.INFO, "Barge-in detected",
                                         event="barge_in")
                                Metrics.record_barge_in()
                                self._publish_admin_event("barge_in", {}, session)
                                await self._handle_barge_in(session)
                        else:
                            barge_in_run.reset()
                            # speculative_turn_text is cleared the moment the
                            # turn's first audio is enqueued, so this only ever
                            # cancels turns nothing was heard from; the duration
                            # gate (same threshold as barge-in) keeps
                            # clicks/pops/coughs from cancelling in-flight LLM
                            # work.
                            tripped = cancel_merge_run.update(chunk_ms, speech)
                            if (tripped
                                    and self.config.endpoint_mode == "speculative"
                                    and session.speculative_turn_text
                                    and session.turn_task
                                    and not session.turn_task.done()):
                                cancel_merge_run.reset()
                                await self._speculative_cancel_merge(session)

                        # Process through VAD/STT
                        transcription = await self.audio_pipeline.process_audio(
                            session.audio_state, audio_chunk)

                        # Speculative endpointing: merge with any held
                        # fragment and hold back fragments that don't look
                        # like a finished thought yet (None = held).
                        if transcription and self.config.endpoint_mode == "speculative":
                            transcription = self._speculative_gate(
                                session, transcription)

                        if transcription:
                            # Run the response turn (ack + LLM + TTS) as a separate
                            # task so this loop keeps reading RTP and can detect
                            # barge-in (via session.processing) during playback.
                            if session.turn_task and not session.turn_task.done():
                                # A turn is already in flight; hold the transcript
                                # and answer it when the current turn finishes.
                                session.pending_transcription = (
                                    f"{session.pending_transcription} {transcription}"
                                    if session.pending_transcription else transcription
                                )
                                logger.debug(f"Turn in progress, holding transcript: {transcription}")
                            else:
                                self._dispatch_turn(session, transcription)

                except Exception as e:
                    logger.debug(f"Audio read error: {e}")

                # Back off only when there was nothing to read — receive_audio()
                # has already slept out its timeout on every empty path. With
                # audio still pending, go straight back for the next chunk so a
                # backlog drains: the old unconditional 50ms sleep, against a
                # read capped at 100ms, let the recording file outrun the reader
                # and settle at a steady ~250ms backlog, which delayed barge-in
                # (and every transcript) by that much on top of the debounce.
                if audio_chunk is None:
                    await asyncio.sleep(0.01)
                else:
                    await asyncio.sleep(0)  # yield to the event loop

            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                await asyncio.sleep(0.1)

        # Cancel any in-flight response turn now that the loop is ending
        await self._cancel_turn(session)

        # A naturally-ended call must free its registry slot (with several
        # concurrent calls allowed, nothing else replaces it): remove exactly
        # this session's entry. Idempotent — a forced teardown, or the
        # outbound caller's finally block, may already have detached it.
        async with self._call_lock:
            self._detach_session(session)

        # Close this session's realtime STT connection (idempotent; no-op in
        # batch mode).
        try:
            await self.audio_pipeline.stop_session_stt(session.audio_state)
        except Exception as e:
            logger.debug(f"Session STT teardown failed: {e}")

        # Close out and persist this call's transcript (idempotent — a
        # forced teardown may already have done it).
        self.transcripts.end(session.transcript_id)
        self._emit_call_event("call.ended", session)
        self._publish_call_ended(session)
        self._finalize_virtual_number(session)
        self._start_memory_update(session)

        # Record conversation turns (count user messages as turns)
        turns = len([m for m in session.conversation_history if m.get("role") == "user"])
        if turns > 0:
            Metrics.record_conversation_turns(turns)

        logger.info("Audio processing loop ended")

    async def _run_turn(self, session: CallSession, transcription: str):
        """Run a full response turn (ack + LLM + TTS) as a cancellable task.

        Runs outside the audio loop so the loop keeps reading RTP and can
        cancel this turn on barge-in or call end via _cancel_turn().
        """
        # Task body top: tools and playback inside this turn resolve their
        # session through the contextvar (create_task snapshots context, so
        # this must be set here, not by the spawner).
        set_current_session(session)
        try:
            # Acknowledge so the caller knows we heard them: an instant earcon
            # (default), a spoken filler phrase, or silence per TURN_ACK_MODE.
            mode = self.config.turn_ack_mode
            if mode == "chime":
                log_event(logger, logging.INFO, "Turn ack: chime",
                         event="assistant_ack", kind="chime")
                await self._play_audio(self._chime_pcm)
            elif mode == "phrase":
                ack = self.get_random_thinking()
                log_event(logger, logging.INFO, f"Assistant: {ack}",
                         event="assistant_ack", kind="phrase", text=ack)
                await self._speak(ack)
            await self._handle_transcription(session, transcription)
        except asyncio.CancelledError:
            logger.info("Response turn cancelled (barge-in or call end)")
            raise
        except Exception as e:
            logger.error(f"Response turn error: {e}")

    async def _cancel_turn(self, session: CallSession):
        """Cancel the session's in-flight response turn task, if any."""
        task = session.turn_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        session.turn_task = None
        # Drop any held transcript: after a barge-in the caller is redirecting
        # the conversation, and after call end there is no one to answer.
        # (The speculative cancel-merge path re-seeds held_fragment AFTER
        # this call, deliberately.)
        session.pending_transcription = None
        session.held_fragment = None
        session.speculative_turn_text = None

    def _dispatch_turn(self, session: CallSession, text: str) -> None:
        """Start a response turn task for ``text``.

        In speculative endpointing mode the dispatched text is recorded on
        the session so the cancel-merge branch can rebuild the utterance if
        the caller resumes speaking before any audio plays."""
        if self.config.endpoint_mode == "speculative":
            session.speculative_turn_text = text
        session.turn_task = asyncio.create_task(self._run_turn(session, text))

    def _speculative_gate(self, session: CallSession,
                          transcription: str) -> Optional[str]:
        """Speculative endpointing: decide whether a fresh transcript is
        dispatched now or held for merging with follow-up speech.

        Merges with any held fragment first. Returns the text to dispatch,
        or None when the (possibly merged) fragment was held because it does
        not look like a finished thought yet. A turn already in flight
        bypasses the hold — the existing pending_transcription machinery
        owns that case."""
        merged = (f"{session.held_fragment} {transcription}".strip()
                  if session.held_fragment else transcription)
        turn_in_flight = bool(session.turn_task and not session.turn_task.done())
        if turn_in_flight or endpointing.looks_complete(merged):
            session.held_fragment = None
            return merged
        session.held_fragment = merged
        session.held_fragment_at = time.monotonic()
        log_event(logger, logging.INFO,
                  f"Fragment looks incomplete, holding: {merged}",
                  event="endpoint_fragment_held", text=merged)
        return None

    async def _speculative_cancel_merge(self, session: CallSession) -> None:
        """The caller resumed speaking after a speculative dispatch but
        BEFORE the assistant was audibly speaking: cancel the in-flight turn
        (it has produced no audio, so nothing was heard and nothing is
        recorded) and hold the dispatched text so the follow-up speech's
        transcript merges with it into a single utterance.

        Distinct from barge-in (playback active -> cancel + discard) and
        from pending_transcription (turn keeps running -> answered after)."""
        # _handle_transcription strips before recording, so compare/re-seed
        # the stripped text (realtime STT can hand over unstripped
        # transcripts with e.g. a leading space).
        first = (session.speculative_turn_text or "").strip()
        session.speculative_turn_text = None
        # Speech that completed while playback was active (e.g. during the
        # ack phrase) was parked in pending_transcription; _cancel_turn is
        # about to wipe it, so fold it into the merged utterance instead of
        # silently dropping it.
        pending = (session.pending_transcription or "").strip()
        log_event(logger, logging.INFO,
                  "Caller resumed before playback; cancelling turn to merge",
                  event="speculative_cancel_merge", text=first)
        await self._cancel_turn(session)
        # The cancelled turn already appended its user message to history and
        # the transcript store; the merged re-dispatch will append the full
        # utterance, so drop the fragment's entries to avoid duplicates.
        history = session.conversation_history
        if (history and history[-1].get("role") == "user"
                and history[-1].get("content") == first):
            history.pop()
            self.transcripts.remove_last_turn(
                session.transcript_id, "user", first)
        # Re-seed AFTER _cancel_turn — it clears all held transcript state.
        session.held_fragment = " ".join(
            part for part in (first, pending) if part)
        session.held_fragment_at = time.monotonic()

    async def _handle_transcription(self, session: CallSession, text: str):
        """Handle transcribed text for one session."""
        text = text.strip()
        if not text or len(text) < 2:
            return

        # Prevent overlapping processing
        if session.processing:
            logger.debug(f"Already processing, queuing: {text}")
            return

        # Record user utterance word count
        word_count = len(text.split())
        Metrics.record_user_utterance(word_count)

        log_event(logger, logging.INFO, f"User: {text}",
                 event="user_speech", text=text)

        # Add to history
        session.conversation_history.append({
            "role": "user",
            "content": text
        })
        self.transcripts.add_turn(session.transcript_id, "user", text)
        self._publish_admin_event("user_turn", {"text": text}, session)
        self._emit_virtual_number_speech(session, text)

        # Deterministic farewell: when the whole utterance is just a goodbye,
        # don't gamble on the model invoking HANGUP — answer with a goodbye
        # phrase and end the call ourselves. Mixed utterances still go to the
        # LLM (and its HANGUP tool).
        if self.config.farewell_hangup_enabled and is_farewell(text):
            session.processing = True
            try:
                await self._farewell_hangup(session)
            finally:
                session.processing = False
            return

        # Generate response: consume the engine's sentence/final event stream.
        # Sentences are spoken the moment they form (with LLM_STREAMING that
        # is roughly first-sentence tokens + one TTS call); the terminal
        # `final` event is the definitive full text for history/metrics and
        # is never spoken — the spoken sentences concatenate to it.
        session.processing = True
        ticker = self._start_thinking_ticker(session)

        try:
            turn_start = time.time()
            history_slice, call_context = await self._build_llm_turn_inputs(
                session, text)
            stream = self.llm_engine.stream_response(history_slice, call_context)

            # Speak first, record after: history must reflect what the caller
            # actually heard. On barge-in, CancelledError lands at an await
            # inside _synthesize_and_play or — since audio is merely ENQUEUED
            # by it — at the playback-drain wait below (before
            # _handle_barge_in flushes the player), and only the heard prefix
            # is recorded.
            ledger = TurnLedger()
            session.active_ledger = ledger
            tts_complete = True
            final_text: Optional[str] = None
            got_first_sentence = False
            first_audio_ms: Optional[float] = None
            try:
                async for event in stream:
                    if event["type"] == "sentence" and event["text"].strip():
                        if not got_first_sentence:
                            got_first_sentence = True
                            # Stop the thinking ticks at first audio, not at
                            # full completion.
                            ticker.cancel()
                            if session.audio_state is not None:
                                session.audio_state.metrics.llm_first_token = (
                                    time.time())
                        ok = await self._synthesize_and_play(
                            event["text"], ledger=ledger)
                        if ok and first_audio_ms is None:
                            first_audio_ms = (time.time() - turn_start) * 1000
                            Metrics.record_time_to_first_audio(
                                first_audio_ms, self.config.llm_model)
                            # The turn is audible from here on: the
                            # speculative cancel-merge precondition ("has
                            # produced no audio") no longer holds — even in
                            # an inter-sentence playback gap — so disarm it.
                            # Later caller speech is a barge-in or a
                            # pending_transcription, never a cancel-merge.
                            session.speculative_turn_text = None
                        tts_complete = tts_complete and ok
                    elif event["type"] == "final":
                        final_text = event["text"]
                ticker.cancel()  # no-sentence safety (empty final)
                # Keep the turn alive (and cancellable) until the queued
                # audio has actually played out; without this, a barge-in
                # during the playback tail would find the turn already
                # done and history would keep the full unheard response.
                await self._wait_for_playback_drain(session)

                if final_text and final_text.strip():
                    # Full text known only now: metrics + response log here.
                    Metrics.record_assistant_response(len(final_text.split()))
                    log_event(logger, logging.INFO, f"Assistant: {final_text}",
                             event="assistant_response", text=final_text,
                             time_to_first_audio_ms=(
                                 round(first_audio_ms)
                                 if first_audio_ms is not None else None))

                    if tts_complete:
                        # Normal completion: full text, exactly as before.
                        session.conversation_history.append({
                            "role": "assistant",
                            "content": final_text
                        })
                        self.transcripts.add_turn(
                            session.transcript_id, "assistant", final_text)
                        self._publish_admin_event(
                            "assistant_turn", {"text": final_text}, session)
                    else:
                        # TTS failed part-way (no audio / swallowed error for
                        # some chunk): the caller did NOT hear the full text,
                        # so record only what actually played.
                        self._record_partial_response(session, ledger)

                    # Off the speaking path: fold overflow turns into the rolling
                    # summary once the history outgrows the LLM window.
                    context_manager.maybe_schedule_summary(self, session)
            except asyncio.CancelledError:
                self._record_interrupted_response(session, ledger)
                raise
            finally:
                # Closing the event stream tears down the engine's underlying
                # HTTP stream too — a barge-in must stop vLLM generation.
                await stream.aclose()
                session.active_ledger = None

        except Exception as e:
            logger.error(f"Response error: {e}")
            await self._speak(self.get_random_error())

        finally:
            ticker.cancel()
            session.processing = False

    async def _farewell_hangup(self, session: CallSession):
        """Speak a goodbye and end the call (FAREWELL_HANGUP_ENABLED path).

        Runs inside the turn task, so the delay before the actual hangup is
        cancellable: a barge-in during the goodbye (the caller changed their
        mind) aborts the hangup along with the rest of the turn.
        """
        goodbye = self.get_random_goodbye()
        log_event(logger, logging.INFO, f"Assistant: {goodbye}",
                 event="assistant_response", text=goodbye)
        # Speak first, record after (same barge-in truthfulness pattern as
        # _handle_transcription): a barge-in during the goodbye cancels the
        # hangup, and history must then show the goodbye as interrupted.
        ledger = TurnLedger()
        session.active_ledger = ledger
        try:
            complete = await self._speak(goodbye, ledger=ledger)
            # Goodbyes are pre-cached, so _speak returns the moment the WAV
            # is enqueued. Park here (cancellably) until it has actually
            # played: a barge-in during the audible goodbye must land inside
            # this try block — not at the sleep below — so the reconstruction
            # path marks the goodbye as interrupted.
            await self._wait_for_playback_drain(session)
            if complete:
                session.conversation_history.append({
                    "role": "assistant",
                    "content": goodbye
                })
                self.transcripts.add_turn(session.transcript_id, "assistant", goodbye)
                self._publish_admin_event(
                    "assistant_turn", {"text": goodbye}, session)
            else:
                self._record_partial_response(session, ledger)
        except asyncio.CancelledError:
            self._record_interrupted_response(session, ledger)
            raise
        finally:
            session.active_ledger = None

        call_info = session.call_info
        # Grace period between the end of the goodbye and dropping RTP (also
        # covers mock/no-player mode, where the drain above returns at once).
        await asyncio.sleep(HANGUP_DELAY_SECONDS)
        if self.current_call is call_info:
            log_event(logger, logging.INFO, "Ending call after caller farewell",
                     event="farewell_hangup")
            await self.sip_handler.hangup_call(call_info)

    def _start_thinking_ticker(self, session: CallSession) -> asyncio.Task:
        """Background task: after thinking_sound_delay_s of LLM silence, play
        a soft tick every thinking_sound_interval_s so the caller knows the
        assistant is still working. Caller cancels it when the response lands.
        """
        async def _tick():
            # Explicit bind (normally inherited from the turn task's context,
            # but the ticker must never play into another call).
            set_current_session(session)
            delay = self.config.thinking_sound_delay_s
            if delay <= 0:
                return
            # A zero/negative interval would spin playing back-to-back ticks.
            interval = max(0.5, self.config.thinking_sound_interval_s)
            try:
                await asyncio.sleep(delay)
                while self.session is session:
                    await self._play_audio(self._thinking_pcm)
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"Thinking ticker stopped: {e}")

        return asyncio.create_task(_tick())

    async def _build_llm_turn_inputs(
            self, session: CallSession,
            user_input: str) -> Tuple[List[Dict], Dict]:
        """Assemble one turn's LLM inputs: (history slice, call_context)."""
        # Snapshot the summary pair BEFORE any await: the background
        # summarizer sets rolling_summary and summarized_upto together,
        # and reading them either side of an await can pick up the new
        # index with the old (empty) summary — silently dropping the
        # turns that were just folded in.
        rolling_summary = session.rolling_summary
        summarized_upto = session.summarized_upto

        call_context = {
            "remote_uri": getattr(session.call_info, 'remote_uri', 'unknown'),
            "duration": time.time() - session.start_time,
        }
        # Refresh caller memory each turn (one tiny local file read):
        # a REMEMBER earlier in this call, or the post-call extraction
        # from a call that ended seconds ago (hangup -> immediate
        # callback), must be visible now — not on the next call.
        if self.config.caller_memory_enabled and session.caller_id:
            try:
                session.caller_memory_prompt = (
                    self.caller_memory.format_for_prompt(session.caller_id))
            except Exception as e:
                logger.warning(f"Could not refresh caller memory: {e}")
        if session.persona:
            call_context["persona"] = session.persona
        if session.caller_memory_prompt:
            call_context["caller_memory"] = session.caller_memory_prompt
        if session.virtual_number and session.virtual_number.purpose:
            call_context["virtual_number_context"] = session.virtual_number.purpose
        # Identity verification state: tell the model whether the caller has
        # verified, and whether any tool is gated behind verification (so it
        # knows to route through the VERIFY tool before a sensitive action).
        if getattr(self.config, "verify_required_tools_set", None):
            call_context["verification_required"] = True
            call_context["verified"] = bool(session.verified)
        if rolling_summary:
            call_context["conversation_summary"] = rolling_summary
        if self.config.knowledge_auto_inject:
            knowledge = await self.knowledge_base.format_for_prompt(user_input)
            if knowledge:
                call_context["knowledge_context"] = knowledge

        # Turns already folded into the rolling summary stay out of the raw
        # history window (the summary travels in call_context).
        return session.conversation_history[summarized_upto:], call_context

    async def _generate_response(self, session: CallSession, user_input: str) -> str:
        """Generate one full LLM response (non-streaming).

        The live speaking path consumes llm_engine.stream_response instead;
        this wrapper remains for callers that need the complete text in one
        piece.
        """
        try:
            history_slice, call_context = await self._build_llm_turn_inputs(
                session, user_input)
            return await self.llm_engine.generate_response(
                history_slice, call_context)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self.get_random_error()


    def _allocate_tag(self, ledger: Optional[TurnLedger], chunk_text: str) -> Optional[int]:
        """Allocate a playback tag for one chunk and record it in the ledger.

        Returns None (untagged playback, today's behavior) when no ledger is
        in play or there is no live session to allocate from.
        """
        session = self.session
        if ledger is None or session is None:
            return None
        tag = session.next_playback_tag
        session.next_playback_tag += 1
        ledger.entries[tag] = chunk_text
        return tag

    async def _speak(self, text: str, ledger: Optional[TurnLedger] = None) -> bool:
        """Synthesize and play text, streaming long responses sentence-by-sentence.

        send_audio() enqueues to the playlist player, so synthesizing sentence
        N+1 while sentence N is still playing cuts time-to-first-audio for long
        LLM responses roughly to the TTS time of the first sentence. Barge-in
        still works: the turn task is cancelled between awaits and the player
        queue is flushed.

        When a ``ledger`` is passed, every enqueued chunk is tagged and
        recorded so a barge-in can reconstruct what the caller actually heard
        (see _handle_transcription). Untagged calls (greeting, acks, earcons)
        behave exactly as before and never touch the ledger.

        Returns True when every chunk was synthesized and enqueued; False when
        TTS failed for any chunk (empty audio or a swallowed error) — callers
        recording history must then fall back to the ledger, because the
        caller did not hear the full text.
        """
        try:
            # Whole-text cache hit (pre-cached phrases) plays instantly.
            cached = self.audio_pipeline.get_cached_audio(text)
            if cached:
                await self._play_audio(cached, tag=self._allocate_tag(ledger, text))
                return True

            sentences = (split_into_sentences(text)
                         if self.config.tts_sentence_streaming else [text])
            if len(sentences) <= 1:
                return await self._synthesize_and_play(text, ledger=ledger)
            complete = True
            for sentence in sentences:
                if not await self._synthesize_and_play(sentence, ledger=ledger):
                    complete = False
            return complete

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False

    async def _synthesize_and_play(self, text: str,
                                   ledger: Optional[TurnLedger] = None) -> bool:
        """Synthesize one chunk of text (cache-aware) and enqueue it.

        Returns True when the chunk was enqueued, False when TTS produced no
        audio (so the chunk was never heard)."""
        cached = self.audio_pipeline.get_cached_audio(text)
        if cached:
            await self._play_audio(cached, tag=self._allocate_tag(ledger, text))
            return True

        start = time.time()
        audio = await self.audio_pipeline.synthesize(text)
        elapsed = (time.time() - start) * 1000

        if audio:
            logger.info(f"TTS: {elapsed:.0f}ms for {len(text)} chars")
            await self._play_audio(audio, tag=self._allocate_tag(ledger, text))
            return True
        logger.warning("TTS returned no audio")
        return False

    async def _play_audio(self, audio: bytes, tag: Optional[int] = None):
        """Play audio to caller. ``tag`` feeds the playback ledger (barge-in
        truthfulness); untagged audio (earcons, greetings) passes through the
        original two-argument send_audio call unchanged."""
        try:
            if self.current_call:
                if tag is None:
                    await self.sip_handler.send_audio(self.current_call, audio)
                else:
                    await self.sip_handler.send_audio(self.current_call, audio,
                                                      tag=tag)
        except Exception as e:
            logger.error(f"Playback error: {e}")

    def _reconstruct_spoken(self, session: CallSession,
                            ledger: TurnLedger) -> str:
        """Best-effort reconstruction of what the caller actually heard of an
        interrupted response, from the ledger + the player's snapshot().

        With no player (mock mode / call already torn down) nothing can be
        proven completed, so this returns "" (append nothing, the safe side).
        """
        completed: List[int] = []
        current: Optional[int] = None
        fraction = 0.0
        get_player = getattr(self.sip_handler, 'get_playlist_player', None)
        if get_player:
            try:
                player = get_player(session.call_info)
                if player is not None and hasattr(player, 'snapshot'):
                    completed, current, fraction = player.snapshot()
            except Exception as e:
                logger.debug(f"Playback snapshot unavailable: {e}")
        return spoken_text(ledger, completed, current, fraction)

    def _record_interrupted_response(self, session: CallSession,
                                     ledger: TurnLedger) -> None:
        """On barge-in, write only the actually-heard prefix into history and
        the transcript (marked as interrupted). Nothing heard -> nothing
        recorded."""
        spoken = self._reconstruct_spoken(session, ledger)
        if not spoken:
            return
        truncated = spoken + " [interrupted by caller]"
        session.conversation_history.append({
            "role": "assistant",
            "content": truncated
        })
        self.transcripts.add_turn(session.transcript_id, "assistant", truncated)
        self._publish_admin_event("assistant_turn", {"text": truncated}, session)
        log_event(logger, logging.INFO,
                  f"Assistant (interrupted): {truncated}",
                  event="barge_in_truncated", text=truncated,
                  chunks_total=len(ledger.entries))

    def _record_partial_response(self, session: CallSession,
                                 ledger: TurnLedger) -> None:
        """TTS failed part-way through a response: some chunks never made it
        into the player, so the full text was NOT heard. Record only what the
        ledger + player snapshot can prove actually played. Nothing heard ->
        nothing recorded."""
        spoken = self._reconstruct_spoken(session, ledger)
        if not spoken:
            return
        session.conversation_history.append({
            "role": "assistant",
            "content": spoken
        })
        self.transcripts.add_turn(session.transcript_id, "assistant", spoken)
        self._publish_admin_event("assistant_turn", {"text": spoken}, session)
        log_event(logger, logging.WARNING,
                  f"Assistant (TTS incomplete): {spoken}",
                  event="tts_incomplete", text=spoken,
                  chunks_total=len(ledger.entries))

    async def _wait_for_playback_drain(self, session: CallSession):
        """Cancellably wait until the playlist player has played out all
        enqueued audio.

        send_audio()/enqueue_file() are non-blocking, so _speak returns when
        the last chunk is merely ENQUEUED — for cache hits and short
        responses, before the caller has heard a single word. The turn must
        stay alive (and cancellable) until playback resolves: a barge-in
        during this wait is delivered here as CancelledError, inside the
        caller's ledger try-block, while the player snapshot is still intact
        (_handle_barge_in cancels the turn BEFORE flushing the player).

        With no player (mock mode / call torn down) there is nothing to wait
        for; errors fail open so a broken snapshot can never wedge a turn.
        """
        get_player = getattr(self.sip_handler, 'get_playlist_player', None)
        if not get_player:
            return
        while True:
            try:
                player = get_player(session.call_info)
                if player is None or not player.has_audio():
                    return
            except Exception as e:
                logger.debug(f"Playback drain check unavailable: {e}")
                return
            await asyncio.sleep(PLAYBACK_DRAIN_POLL_S)

    def _playback_active(self, session: CallSession) -> bool:
        """True while the playlist player has audio playing or queued."""
        get_player = getattr(self.sip_handler, 'get_playlist_player', None)
        if not get_player:
            return False
        player = get_player(session.call_info)
        return bool(player and player.has_audio())

    async def _handle_barge_in(self, session: CallSession):
        """Handle user interruption."""
        logger.info("Stopping playback for barge-in")
        # Cancel the in-flight response turn (LLM+TTS) so it stops producing audio
        await self._cancel_turn(session)
        # Flush playback via playlist player. Use clear() (transient) NOT stop_all()
        # (terminal): stop_all latches the player as permanently stopped, which
        # would silently drop every later turn's audio for the rest of the call.
        player = self.sip_handler.get_playlist_player(session.call_info)
        if player:
            player.clear()
        session.processing = False
        
    async def make_outbound_call(self, uri: str, message: str):
        """Make an outbound call and play a message, then start interactive session."""
        import re
        
        # Parse SIP URI - handle formats like:
        # "Display Name" <sip:user@domain>
        # <sip:user@domain>
        # sip:user@domain
        # user@domain
        # extension
        
        original_uri = uri
        
        # Extract URI from angle brackets if present (e.g., "Name" <sip:420@domain>)
        angle_match = re.search(r'<(sip:[^>]+)>', uri)
        if angle_match:
            uri = angle_match.group(1)
        elif '<' in uri and '>' in uri:
            # Try to extract anything in angle brackets
            angle_match = re.search(r'<([^>]+)>', uri)
            if angle_match:
                uri = angle_match.group(1)
                if not uri.startswith('sip:'):
                    uri = f"sip:{uri}"
        
        # If still no sip: prefix, build the URI
        if not uri.startswith('sip:'):
            # Strip any remaining angle brackets or quotes
            clean_uri = uri.replace('<', '').replace('>', '').replace('"', '').strip()
            # If it doesn't have an @, add the domain
            if '@' not in clean_uri:
                uri = f"sip:{clean_uri}@{self.config.sip_domain}"
            else:
                uri = f"sip:{clean_uri}"

        # Strip control characters (CR/LF/NUL and friends) before this URI is
        # handed to PJSIP's makeCall, so a crafted destination cannot inject
        # extra SIP headers/parameters.
        uri = re.sub(r'[\x00-\x1f\x7f]', '', uri).strip()

        logger.info(f"Making outbound call to {uri} (from: {original_uri})")
        try:
            call_info = await self.sip_handler.make_call(uri)
            if call_info:
                # Wait for call to connect (configurable ring timeout)
                ring_timeout = self.config.callback_ring_timeout_s
                start_time = asyncio.get_event_loop().time()
                
                # Poll for call to be answered
                while asyncio.get_event_loop().time() - start_time < ring_timeout:
                    if getattr(call_info, 'is_active', False):
                        break
                    await asyncio.sleep(0.5)
                else:
                    # Timed out waiting for answer
                    log_event(logger, logging.WARNING, f"Call to {uri} not answered",
                             event="call_timeout", uri=uri, timeout=ring_timeout)
                    Metrics.record_call_failed("outbound", "timeout")
                    await self.sip_handler.hangup_call(call_info)
                    return
                
                log_event(logger, logging.INFO, f"Outbound call connected to {uri}",
                         event="call_start", caller=uri, direction="outbound")
                
                # Record call started metric
                Metrics.record_call_started("outbound")
                outbound_call_start_time = time.time()
                
                # Small delay after answer for audio to stabilize
                await asyncio.sleep(1)
                
                # Play the callback message
                audio = await self.audio_pipeline.synthesize(message)
                if audio:
                    await self.sip_handler.send_audio(call_info, audio)
                    # Wait for audio to play (estimate based on audio length)
                    audio_duration = len(audio) / (self.config.sample_rate * 2)
                    await asyncio.sleep(audio_duration + 0.5)

                # Now start interactive session
                try:
                    # The lock guards registry mutation only: this call ADDS
                    # a session — existing sessions (an inbound call in
                    # progress) are no longer torn down.
                    async with self._call_lock:
                        logger.info(f"Starting interactive session with: {uri}")
                        session = self._begin_session(call_info, "outbound", uri)

                    # Everything below acts on behalf of the new call (the
                    # followup TTS and the audio loop resolve the session
                    # through the contextvar).
                    set_current_session(session)

                    self.transcripts.add_turn(session.transcript_id, "assistant", message)
                    self._publish_admin_event(
                        "assistant_turn", {"text": message}, session)

                    # Per-session realtime STT (no-op in batch mode).
                    await self.audio_pipeline.start_session_stt(
                        session.audio_state)

                    # Ask if they need anything else
                    followup = self.get_random_followup()
                    log_event(logger, logging.INFO, f"Assistant: {followup}",
                             event="assistant_response", text=followup)
                    await self._speak(followup)

                    # Start listening loop (runs until call ends)
                    logger.info("Listening...")
                    session.audio_loop_task = asyncio.create_task(
                        self._audio_processing_loop(session))

                    # Wait for the audio loop to complete (call ends)
                    await session.audio_loop_task

                except asyncio.CancelledError:
                    logger.info("Outbound call session cancelled")
                except Exception as e:
                    logger.error(f"Error in interactive session: {e}", exc_info=True)
                finally:
                    # Clean up when session ends
                    if call_info.is_active:
                        await self.sip_handler.hangup_call(call_info)
                    # Only detach THIS session's registry entry (identity
                    # match), so we don't clobber an inbound call that
                    # arrived meanwhile and replaced it.
                    async with self._call_lock:
                        self._detach_session(session)
                    
                logger.info(f"Outbound call to {uri} completed")
                
                # Record call end metrics
                if 'outbound_call_start_time' in locals():
                    duration_ms = (time.time() - outbound_call_start_time) * 1000
                    Metrics.record_call_duration(duration_ms, "outbound")
                    Metrics.record_call_ended("outbound", "completed")
            else:
                logger.error(f"Failed to connect outbound call to {uri}")
        except Exception as e:
            logger.error(f"Outbound call failed: {e}")
            raise
        
    async def schedule_callback(self, delay: int, message: str = "This is your scheduled callback.", destination: str = None):
        """Schedule a callback to the caller."""
        if destination == "CALLER_NUMBER" or destination is None:
            # Get caller's number from current call
            if self.current_call:
                destination = getattr(self.current_call, 'remote_uri', None)
                
        if not destination:
            logger.warning("No destination for callback")
            return
            
        log_event(logger, logging.INFO, f"Callback scheduled: {delay}s to {destination}",
                 event="callback_scheduled", delay=delay, destination=destination, message=message)
        
        # Use tool_manager's scheduler for proper task management
        await self.tool_manager.schedule_task(
            task_type="callback",
            delay_seconds=delay,
            message=message,
            target_uri=destination
        )


def _api_bind_is_exposed(host: str) -> bool:
    """True if the API bind address is reachable from outside the local host.

    Loopback (127.0.0.1 / ::1 / localhost) is considered safe; 0.0.0.0 / :: (all
    interfaces) and any specific routable address are treated as exposed.
    """
    h = (host or "").strip()
    if not h or h == "localhost":
        return False
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        # A non-IP hostname we can't classify — treat as exposed to be safe.
        return True
    return ip.is_unspecified or not ip.is_loopback


async def main():
    """Main entry point."""
    config = get_config()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))
    
    assistant = SIPAIAssistant(config)
    
    # Handle shutdown
    loop = asyncio.get_event_loop()
    
    shutdown_event = asyncio.Event()
    
    def shutdown_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()
        
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)
    
    # Start API server
    api_port = int(os.environ.get("API_PORT", "8080"))
    api_task = None
    call_queue = None
    
    try:
        # Start assistant
        await assistant.start_components()
        
        # Create and connect call queue
        redis_url = os.environ.get("REDIS_URL")
        max_concurrent = int(os.environ.get("CALL_QUEUE_MAX_CONCURRENT", "1"))
        
        if redis_url:
            from call_queue import CallQueue
            call_queue = CallQueue(redis_url=redis_url, max_concurrent=max_concurrent)
            await call_queue.connect()
            log_event(logger, logging.INFO, f"Call queue connected (max_concurrent={max_concurrent})",
                     event="queue_connected", max_concurrent=max_concurrent)
        else:
            logger.warning("REDIS_URL not set - call queue disabled, calls will execute directly")
        
        # Create and start API
        from api import create_api
        from telemetry import instrument_fastapi
        import uvicorn
        
        app = create_api(assistant, call_queue)

        # Sanity-check the call-event webhook URL against the webhook policy so
        # a misconfiguration is visible at startup instead of silently dropping
        # every event (deliver_webhook re-validates per send and never raises).
        if assistant.config.call_event_webhook_url:
            from api import RequestRejected, validate_callback_url
            try:
                await validate_callback_url(
                    assistant.config.call_event_webhook_url, assistant.config)
            except RequestRejected as e:
                logger.warning(
                    "CALL_EVENT_WEBHOOK_URL %s is rejected by webhook policy (%s); "
                    "call events will NOT be delivered. For private URLs like "
                    "http://n8n:5678/... set WEBHOOK_ALLOW_PRIVATE=true.",
                    assistant.config.call_event_webhook_url, e.detail)

        # Instrument FastAPI with OpenTelemetry
        instrument_fastapi(app)
        
        # Start queue worker (uses handler from app.state)
        if call_queue:
            await call_queue.start(app.state.handler)

        # Fail closed: an unauthenticated, externally-bound REST API exposes
        # outbound dialing (toll fraud), arbitrary tool execution and SSRF-capable
        # webhooks to anyone who can reach the port. Refuse to start in that
        # configuration unless the operator explicitly opts out.
        #
        # In Docker the container must bind 0.0.0.0 for the published port to
        # work, so exposure is judged by the host-side publish address
        # (API_BIND_ADDRESS, forwarded by the compose files) when available.
        exposure_host = assistant.config.api_published_host or assistant.config.api_host
        if _api_bind_is_exposed(exposure_host) and not assistant.config.api_auth_token:
            if assistant.config.allow_unauthenticated:
                logger.warning(
                    "REST API reachable on %s:%s with NO auth token "
                    "(ALLOW_UNAUTHENTICATED=true) - anyone who can reach it can place "
                    "calls and execute tools.", exposure_host, api_port)
            else:
                raise RuntimeError(
                    f"Refusing to start: REST API is reachable on {exposure_host} "
                    f"(externally reachable) with no API_AUTH_TOKEN set. Set API_AUTH_TOKEN to a "
                    f"secret, or bind to loopback (API_HOST=127.0.0.1, or API_BIND_ADDRESS=127.0.0.1 "
                    f"under Docker), or set ALLOW_UNAUTHENTICATED=true to override (not recommended)."
                )

        uvicorn_config = uvicorn.Config(
            app,
            host=assistant.config.api_host,
            port=api_port,
            log_level="warning",
            access_log=False
        )
        server = uvicorn.Server(uvicorn_config)
        
        log_event(logger, logging.INFO, f"API server starting on port {api_port}",
                 event="api_started", port=api_port)
        
        # Run server in background
        api_task = asyncio.create_task(server.serve())
        
        # Run assistant main loop
        await assistant.run_loop(shutdown_event)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        # Let an active call finish its turn and hear a goodbye before the
        # SIP stack is torn down. Stop the queue first so no NEW outbound
        # calls start while we drain.
        if call_queue:
            await call_queue.stop()
            await call_queue.disconnect()

        try:
            await assistant.drain_active_call()
        except Exception as e:
            logger.warning(f"Call drain failed: {e}")

        # Stop API
        if api_task:
            api_task.cancel()
            try:
                await api_task
            except asyncio.CancelledError:
                pass

        await assistant.stop()


if __name__ == "__main__":
    import os
    asyncio.run(main())