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
import re
import time
import random
import signal
import asyncio
import logging
import ipaddress
from typing import List, Dict, Optional

import call_events
import context_manager
from call_session import CallSession
from caller_memory import CallerMemoryStore, caller_id_from_uri
from earcons import generate_chime
from knowledge_base import KnowledgeBase
from sip_handler import SIPHandler
from tool_manager import ToolManager
from transcript_store import TranscriptStore
from config import Config, get_config
from llm_engine import create_llm_engine
from audio_pipeline import LowLatencyAudioPipeline
from logging_utils import log_event

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

# Split after sentence punctuation followed by whitespace ("3.5" is safe: no
# whitespace after the dot).
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def split_into_sentences(text: str, min_chars: int = 25) -> List[str]:
    """Split text into speakable sentence chunks for streaming TTS.

    Fragments shorter than ``min_chars`` merge into the following one so
    abbreviations and one-word sentences don't produce choppy TTS calls.
    """
    parts = [p for p in _SENTENCE_SPLIT.split(text.strip()) if p]
    merged: List[str] = []
    for part in parts:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


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

        # Core components
        self.tool_manager = ToolManager(self)
        self.llm_engine = create_llm_engine(config, self.tool_manager)
        self.audio_pipeline = LowLatencyAudioPipeline(config)
        self.sip_handler = SIPHandler(config, self._on_call_received)
        
        # State: everything owned by the live call (history, turn task, held
        # transcripts, the audio loop) lives on the CallSession, so a stale
        # task from a replaced call can never write into the next call's
        # conversation. self.session is the single active session.
        self.session: Optional[CallSession] = None
        self._call_lock = asyncio.Lock()

        # Per-call conversation transcripts (bounded memory + data/transcripts).
        self.transcripts = TranscriptStore(config)
        self._session_counter = 0

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

        # Confirmation earcon (in-memory PCM), generated once; played via the
        # same send_audio path as TTS so barge-in/flush semantics are identical.
        self._chime_pcm: bytes = generate_chime(
            sample_rate=config.sample_rate, volume=config.chime_volume)

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

    def _begin_session(self, call_info, direction: str, remote: str) -> CallSession:
        """Create and install a new active session (caller holds _call_lock)."""
        prefix = "in" if direction == "inbound" else "out"
        self._session_counter += 1
        session = CallSession(
            call_info=call_info,
            direction=direction,
            # Counter suffix keeps ids unique for calls in the same second.
            transcript_id=f"{prefix}-{int(time.time())}-{self._session_counter}",
        )
        self.session = session
        self.transcripts.start(session.transcript_id, direction, remote)
        # Load what we remember about this caller (one small disk read).
        if self.config.caller_memory_enabled:
            try:
                caller_id = caller_id_from_uri(remote)
                if caller_id:
                    session.caller_memory_prompt = (
                        self.caller_memory.format_for_prompt(caller_id))
            except Exception as e:
                logger.warning(f"Could not load caller memory: {e}")
        self._emit_call_event("call.started", session)
        return session

    async def _teardown_session(self):
        """Stop the active session's tasks and close its transcript.

        Detaches the session first so its tasks observe `self.session is not
        them` and cannot touch the replacement session's state.
        """
        session = self.session
        if session is None:
            return
        self.session = None
        if session.audio_loop_task and not session.audio_loop_task.done():
            session.audio_loop_task.cancel()
            try:
                await session.audio_loop_task
            except asyncio.CancelledError:
                pass
        await self._cancel_turn(session)
        self.transcripts.end(session.transcript_id)
        self._emit_call_event("call.ended", session)
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
        """Wind down an active call gracefully before shutdown.

        Lets the in-flight response turn finish (bounded by ``turn_timeout``),
        speaks a goodbye so the caller isn't cut off mid-sentence, then hangs
        up cleanly. No-op when idle.
        """
        session = self.session
        call = session.call_info if session else None
        if not (call and getattr(call, 'is_active', False)):
            return

        logger.info("Draining active call before shutdown")
        task = session.turn_task
        if task and not task.done():
            try:
                # shield: a timeout should move on to the goodbye, not leave a
                # half-cancelled turn behind (_cancel_turn finishes the job).
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

    async def stop(self):
        """Stop all components."""
        logger.info("Stopping...")
        self.running = False

        # Stop the active session (audio loop + in-flight turn + transcript)
        await self._teardown_session()

        # Give in-flight call-event webhooks a moment to finish delivery.
        if self._event_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._event_tasks, return_exceptions=True),
                    timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Call-event webhook delivery still pending at shutdown")

        await self.tool_manager.stop()
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
        # Prevent duplicate handling
        if self._call_lock.locked():
            logger.warning("Call already being handled, ignoring duplicate callback")
            return
            
        async with self._call_lock:
            try:
                remote_uri = getattr(call_info, 'remote_uri', 'unknown')
                log_event(logger, logging.INFO, f"Call received from: {remote_uri}",
                         event="call_start", caller=remote_uri, direction="inbound")

                # Record call started metric
                Metrics.record_call_started("inbound")

                # Replace any existing session: its audio loop and in-flight
                # turn are cancelled and its transcript closed first.
                await self._teardown_session()
                session = self._begin_session(call_info, "inbound", remote_uri)

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

        try:
            logger.info(f"Playing greeting: {greeting}")
            # This should hit the cache since we pre-cached it
            audio = await self.audio_pipeline.synthesize(greeting)
            if audio:
                await self._play_audio(audio)
                self.transcripts.add_turn(session.transcript_id, "assistant", greeting)
        except Exception as e:
            logger.error(f"Error playing greeting: {e}")
            
    async def _audio_processing_loop(self, session: CallSession):
        """Main audio processing loop for one call session."""
        logger.info("Audio processing loop started")

        audio_received_count = 0
        last_log_time = time.time()
        # Consecutive speech (ms) heard while the assistant is speaking; a
        # barge-in only triggers once this reaches barge_in_min_duration_ms,
        # so clicks/pops/short noise bursts can't cancel an in-flight turn.
        barge_in_speech_ms = 0.0

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
                    session.turn_task = asyncio.create_task(
                        self._run_turn(session, pending))

                try:
                    # Try to receive audio
                    audio_chunk = await self.sip_handler.receive_audio(
                        session.call_info,
                        timeout=0.1
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
                        if self.audio_pipeline.has_speech(audio_chunk) and \
                                self._playback_active(session):
                            barge_in_speech_ms += (
                                len(audio_chunk) / 2 / self.config.sample_rate * 1000)
                            if barge_in_speech_ms >= self.config.barge_in_min_duration_ms:
                                barge_in_speech_ms = 0.0
                                log_event(logger, logging.INFO, "Barge-in detected",
                                         event="barge_in")
                                Metrics.record_barge_in()
                                await self._handle_barge_in(session)
                        else:
                            barge_in_speech_ms = 0.0

                        # Process through VAD/STT
                        transcription = await self.audio_pipeline.process_audio(audio_chunk)

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
                                session.turn_task = asyncio.create_task(
                                    self._run_turn(session, transcription)
                                )

                except Exception as e:
                    logger.debug(f"Audio read error: {e}")

                await asyncio.sleep(0.05)  # 50ms polling interval

            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                await asyncio.sleep(0.1)

        # Cancel any in-flight response turn now that the loop is ending
        await self._cancel_turn(session)

        # Close out and persist this call's transcript (idempotent — a
        # forced teardown may already have done it).
        self.transcripts.end(session.transcript_id)
        self._emit_call_event("call.ended", session)
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
        session.pending_transcription = None

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

        # Generate response
        session.processing = True

        try:
            response = await self._generate_response(session, text)

            if response:
                # Record assistant response word count
                response_word_count = len(response.split())
                Metrics.record_assistant_response(response_word_count)

                log_event(logger, logging.INFO, f"Assistant: {response}",
                         event="assistant_response", text=response)

                # Add to history
                session.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                self.transcripts.add_turn(session.transcript_id, "assistant", response)

                # Synthesize and play response
                await self._speak(response)

                # Off the speaking path: fold overflow turns into the rolling
                # summary once the history outgrows the LLM window.
                context_manager.maybe_schedule_summary(self, session)

        except Exception as e:
            logger.error(f"Response error: {e}")
            await self._speak(self.get_random_error())

        finally:
            session.processing = False

    async def _generate_response(self, session: CallSession, user_input: str) -> str:
        """Generate LLM response."""
        try:
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
            if session.caller_memory_prompt:
                call_context["caller_memory"] = session.caller_memory_prompt
            if rolling_summary:
                call_context["conversation_summary"] = rolling_summary
            if self.config.knowledge_auto_inject:
                knowledge = await self.knowledge_base.format_for_prompt(user_input)
                if knowledge:
                    call_context["knowledge_context"] = knowledge

            response = await self.llm_engine.generate_response(
                # Turns already folded into the rolling summary stay out of
                # the raw history window (the summary travels in call_context).
                session.conversation_history[summarized_upto:],
                call_context,
            )
            return response
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self.get_random_error()
            
    async def _speak(self, text: str):
        """Synthesize and play text, streaming long responses sentence-by-sentence.

        send_audio() enqueues to the playlist player, so synthesizing sentence
        N+1 while sentence N is still playing cuts time-to-first-audio for long
        LLM responses roughly to the TTS time of the first sentence. Barge-in
        still works: the turn task is cancelled between awaits and the player
        queue is flushed.
        """
        try:
            # Whole-text cache hit (pre-cached phrases) plays instantly.
            cached = self.audio_pipeline.get_cached_audio(text)
            if cached:
                await self._play_audio(cached)
                return

            sentences = (split_into_sentences(text)
                         if self.config.tts_sentence_streaming else [text])
            if len(sentences) <= 1:
                await self._synthesize_and_play(text)
                return
            for sentence in sentences:
                await self._synthesize_and_play(sentence)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"TTS error: {e}")

    async def _synthesize_and_play(self, text: str):
        """Synthesize one chunk of text (cache-aware) and enqueue it."""
        cached = self.audio_pipeline.get_cached_audio(text)
        if cached:
            await self._play_audio(cached)
            return

        start = time.time()
        audio = await self.audio_pipeline.synthesize(text)
        elapsed = (time.time() - start) * 1000

        if audio:
            logger.info(f"TTS: {elapsed:.0f}ms for {len(text)} chars")
            await self._play_audio(audio)
        else:
            logger.warning("TTS returned no audio")
            
    async def _play_audio(self, audio: bytes):
        """Play audio to caller."""
        try:
            if self.current_call:
                await self.sip_handler.send_audio(self.current_call, audio)
        except Exception as e:
            logger.error(f"Playback error: {e}")
            
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
                    # Guard session setup with the call lock (consistent with
                    # _on_call_received) so an inbound call can't interleave.
                    async with self._call_lock:
                        # Replace any existing session (cancels its tasks and
                        # closes its transcript).
                        await self._teardown_session()

                        logger.info(f"Starting interactive session with: {uri}")

                        session = self._begin_session(call_info, "outbound", uri)
                        self.transcripts.add_turn(session.transcript_id, "assistant", message)

                        # Ask if they need anything else
                        followup = self.get_random_followup()
                        log_event(logger, logging.INFO, f"Assistant: {followup}",
                                 event="assistant_response", text=followup)
                        await self._speak(followup)

                        # Start listening loop (runs until call ends)
                        logger.info("Listening...")
                        session.audio_loop_task = asyncio.create_task(
                            self._audio_processing_loop(session))

                    # Wait for the audio loop to complete (call ends) - outside lock
                    await session.audio_loop_task

                except asyncio.CancelledError:
                    logger.info("Outbound call session cancelled")
                except Exception as e:
                    logger.error(f"Error in interactive session: {e}", exc_info=True)
                finally:
                    # Clean up when session ends
                    if call_info.is_active:
                        await self.sip_handler.hangup_call(call_info)
                    # Only detach if the active session is still THIS one, so
                    # we don't clobber an inbound call that arrived meanwhile.
                    async with self._call_lock:
                        if self.session is session:
                            self.session = None
                    
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