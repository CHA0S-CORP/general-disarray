"""
WebSocket Realtime Client for Speaches API
===========================================
Uses the /v1/realtime WebSocket endpoint for low-latency streaming STT.

This implements the OpenAI Realtime API protocol which Speaches supports:
- WebSocket connection to /v1/realtime?model=<model>&intent=transcription
- Send audio via input_audio_buffer.append events
- Receive transcriptions via conversation.item.input_audio_transcription.completed events

This provides significantly lower latency compared to batch transcription
by streaming audio in real-time and receiving transcription results as they're ready.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass
from math import gcd
import urllib.parse

import numpy as np
from scipy import signal

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketClientProtocol = None

from config import Config
from telemetry import create_span, Metrics
from logging_utils import (
    log_event, 
    RECONNECT_BASE_DELAY_SECONDS, 
    RECONNECT_MAX_DELAY_SECONDS
)

logger = logging.getLogger(__name__)


# Speaches' /v1/realtime follows the OpenAI Realtime spec: wire audio is PCM16
# mono @ 24 kHz. The agent works at config.sample_rate (16 kHz), so audio must be
# resampled up to this rate before it is base64-encoded onto the WebSocket.
REALTIME_WIRE_SAMPLE_RATE = 24000


class _StreamingResampler:
    """Streaming polyphase resampler that keeps FIR filter state across chunks.

    scipy.signal.resample_poly redesigns its anti-aliasing filter on every call
    and treats each buffer as a complete signal, so calling it per ~20 ms RTP
    chunk both burns CPU on the hot audio path and creates discontinuities at
    every chunk seam. This designs the same kaiser-windowed low-pass once, then
    zero-stuffs, filters with carried state and decimates with a carried phase,
    so a chunked stream resamples identically to one continuous signal.
    """

    def __init__(self, up: int, down: int):
        self.up = up
        self.down = down
        # Same filter design resample_poly uses internally, scaled by `up` to
        # preserve amplitude after zero-stuffing.
        max_rate = max(up, down)
        half_len = 10 * max_rate
        self._taps = signal.firwin(2 * half_len + 1, 1.0 / max_rate,
                                   window=('kaiser', 5.0)) * up
        self._zi = np.zeros(len(self._taps) - 1)
        self._phase = 0  # global upsampled-sample index modulo `down`

    def process(self, samples: np.ndarray) -> np.ndarray:
        upsampled = np.zeros(samples.size * self.up)
        upsampled[::self.up] = samples
        filtered, self._zi = signal.lfilter(self._taps, 1.0, upsampled, zi=self._zi)
        offset = (-self._phase) % self.down
        self._phase = (self._phase + upsampled.size) % self.down
        return filtered[offset::self.down]


@dataclass
class TranscriptionResult:
    """Result from realtime transcription."""
    text: str
    is_final: bool
    item_id: Optional[str] = None
    confidence: float = 1.0
    latency_ms: float = 0.0


class RealtimeWebSocketClient:
    """
    WebSocket-based realtime STT client for Speaches API.
    
    Uses the /v1/realtime WebSocket endpoint with OpenAI Realtime API protocol
    for streaming audio and receiving transcriptions in real-time.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.speaches_api_url.rstrip('/')
        self.model = config.whisper_model
        self.language = config.whisper_language
        
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._session_id: Optional[str] = None
        
        self._transcription_callback: Optional[Callable[[TranscriptionResult], Awaitable[None]]] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

        # Resolved by the receive loop when a transcript (or failure) arrives for
        # the buffer we just committed. Lets commit_and_wait() return deterministically
        # instead of racing a temporary callback swap.
        self._pending_transcript: Optional[asyncio.Future] = None

        # Correlate each commit with the server item_id so a late transcript from
        # a timed-out turn cannot resolve a later turn's waiter. turn_detection is
        # None, so each commit produces exactly one item. _commit_seq is a
        # monotonically increasing turn token; _committed_items maps an item_id to
        # the turn token it was committed under.
        self._commit_seq = 0
        self._pending_seq: Optional[int] = None
        self._inflight_item_id: Optional[str] = None
        self._committed_items: dict = {}

        # Audio buffering for batch fallback
        self._audio_buffer = bytearray()
        self._last_audio_time = 0.0

        # Lazily-built streaming resampler for the 16 kHz -> 24 kHz wire rate
        # conversion (filter designed once, state carried across chunks).
        self._resampler: Optional[_StreamingResampler] = None
        
        # Track connection state for metrics
        self._connection_attempts = 0
        self._last_connection_error: Optional[str] = None
        
        self.available = WEBSOCKETS_AVAILABLE
        
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("websockets package not installed - WebSocket realtime mode unavailable")
            logger.warning("Install with: pip install websockets")
            
    def _build_ws_url(self) -> str:
        """Build the WebSocket URL with query parameters."""
        # Convert http(s) to ws(s)
        ws_base = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://')
        
        # Build query parameters
        params = {
            'model': self.model,
            'intent': 'transcription',  # Transcription-only mode
        }
        if self.language:
            params['language'] = self.language
            
        query_string = urllib.parse.urlencode(params)
        return f"{ws_base}/v1/realtime?{query_string}"
            
    async def initialize(self):
        """Initialize the WebSocket client and establish connection."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("Cannot initialize realtime client - websockets not available")
            return
            
        # Establish WebSocket connection
        await self._connect()
        
    async def _connect(self):
        """Establish WebSocket connection to Speaches realtime endpoint."""
        if not WEBSOCKETS_AVAILABLE:
            return
            
        self._connection_attempts += 1
        Metrics.record_realtime_connection_attempt()
        
        try:
            ws_url = self._build_ws_url()
            logger.info(f"Connecting to Speaches realtime API: {ws_url}")
            
            # Connect with reasonable timeouts
            self._ws = await websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
                max_size=10 * 1024 * 1024,  # 10MB max message size
            )
            
            self._connected = True
            # Re-arm availability: a prior failed connect may have cleared this,
            # and a later successful (re)connect must restore it.
            self.available = True
            self._last_connection_error = None
            Metrics.record_realtime_connection_state("connected")
            logger.info("WebSocket connection established with Speaches realtime API")
            
            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            # Send session configuration
            await self._configure_session()
            
        except Exception as e:
            self._connected = False
            self._last_connection_error = str(e)
            Metrics.record_realtime_connection_state("failed")
            Metrics.record_realtime_connection_error(type(e).__name__)
            logger.error(f"WebSocket connection error: {e}")
            # Don't permanently disable realtime here: availability is re-armed on
            # the next successful (re)connect. Mutating it False would survive a
            # later reconnect and silently kill realtime STT for the process.
            
    async def _configure_session(self):
        """Send session configuration after connection."""
        if not self._ws or not self._connected:
            return
            
        # Configure the session for transcription.
        # turn_detection is disabled (null): the agent's local VAD owns turn
        # boundaries and explicitly commits the buffer, so the server must not
        # auto-commit. This keeps a single, deterministic trigger per utterance.
        session_config = {
            "type": "session.update",
            "session": {
                "input_audio_transcription": {
                    "model": self.model
                },
                "turn_detection": None
            }
        }
        
        if self.language:
            session_config["session"]["input_audio_transcription"]["language"] = self.language
            
        await self._ws.send(json.dumps(session_config))
        logger.debug("Session configuration sent")
        
    async def _receive_loop(self):
        """Receive and process messages from the WebSocket."""
        if not self._ws:
            return
            
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message[:100]}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self._connected = False
            Metrics.record_realtime_connection_state("disconnected")
            await self._handle_connection_failure()
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
            self._connected = False
            Metrics.record_realtime_connection_error(type(e).__name__)
            await self._handle_connection_failure()
            
    def _resolve_pending(self, value: str, item_id: Optional[str]):
        """Resolve the in-flight commit_and_wait() future, but only with a result
        that belongs to the current turn.

        A late completed/failed event from a previously timed-out commit names an
        item_id that maps (via _committed_items) to an earlier turn token. If that
        token doesn't match the current _pending_seq, the event is stale and is
        ignored instead of corrupting the next turn's transcript.
        """
        fut = self._pending_transcript
        if fut is None or fut.done():
            return
        if item_id is not None:
            seq = self._committed_items.get(item_id)
            if seq is not None and seq != self._pending_seq:
                logger.debug(
                    f"Ignoring stale transcript for item {item_id} "
                    f"(turn {seq}, current turn {self._pending_seq})"
                )
                return
        fut.set_result(value)

    async def _handle_message(self, data: dict):
        """Handle a message from the WebSocket."""
        msg_type = data.get("type", "")
        
        if msg_type == "session.created":
            self._session_id = data.get("session", {}).get("id")
            logger.info(f"Realtime session created: {self._session_id}")
            
        elif msg_type == "session.updated":
            logger.debug("Session configuration updated")
            
        elif msg_type == "input_audio_buffer.speech_started":
            log_event(logger, logging.DEBUG, "Speech started detected",
                     event="realtime_speech_started")
            
        elif msg_type == "input_audio_buffer.speech_stopped":
            log_event(logger, logging.DEBUG, "Speech stopped detected",
                     event="realtime_speech_stopped")
            
        elif msg_type == "input_audio_buffer.committed":
            # turn_detection is None, so this commit produced exactly one item.
            # Record which turn token it belongs to so a slow transcript can later
            # be correlated back to its commit and not leak into a different turn.
            item_id = data.get("item_id")
            if item_id is not None and self._pending_seq is not None:
                self._inflight_item_id = item_id
                self._committed_items[item_id] = self._pending_seq
                # Bound the map: only the current and a few recent turns matter.
                if len(self._committed_items) > 8:
                    cutoff = self._pending_seq - 4
                    self._committed_items = {
                        k: v for k, v in self._committed_items.items() if v >= cutoff
                    }
            logger.debug("Audio buffer committed")
            
        elif msg_type == "conversation.item.input_audio_transcription.completed":
            # This is the final transcription result for the committed buffer.
            transcript = data.get("transcript", "")
            item_id = data.get("item_id")

            # Resolve the in-flight commit_and_wait() deterministically (even for
            # an empty transcript, so the waiter returns immediately rather than
            # timing out) — but only if this item_id belongs to the current turn.
            self._resolve_pending(transcript, item_id)

            if transcript:
                result = TranscriptionResult(
                    text=transcript,
                    is_final=True,
                    item_id=item_id
                )

                log_event(logger, logging.DEBUG, f"Transcription received: {transcript[:50]}...",
                         event="realtime_transcription", text_length=len(transcript))

                # Record metrics
                Metrics.record_stt_latency(0, self.model)  # Latency tracked elsewhere

                if self._transcription_callback:
                    await self._transcription_callback(result)

        elif msg_type == "conversation.item.input_audio_transcription.failed":
            # Unblock the waiter with an empty result instead of stalling.
            error = data.get("error", {})
            logger.warning(f"Realtime transcription failed: {error.get('message', 'unknown')}")
            Metrics.record_stt_error(self.model, "realtime_transcription_failed")
            self._resolve_pending("", data.get("item_id"))

        elif msg_type == "conversation.item.input_audio_transcription.delta":
            # Partial transcription (streaming)
            delta = data.get("delta", "")
            if delta and self._transcription_callback:
                result = TranscriptionResult(
                    text=delta,
                    is_final=False,
                    item_id=data.get("item_id")
                )
                await self._transcription_callback(result)
                
        elif msg_type == "error":
            error = data.get("error", {})
            error_msg = error.get("message", "Unknown error")
            error_type = error.get("type", "unknown")
            logger.error(f"Realtime API error: {error_type} - {error_msg}")
            Metrics.record_stt_error(self.model, f"realtime_{error_type}")
            # A commit can fail server-side (e.g. buffer too short); don't make
            # commit_and_wait() block for the full timeout — resolve it empty.
            # Generic error events carry no item_id, so this resolves the current
            # in-flight waiter only (a completed future is left untouched).
            self._resolve_pending("", data.get("item_id"))
            
        else:
            logger.debug(f"Unhandled message type: {msg_type}")
            
    async def _handle_connection_failure(self):
        """Handle WebSocket connection failure - attempt reconnect."""
        logger.warning("WebSocket connection failed, attempting reconnect...")
        self._connected = False
        
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())
            
    async def _reconnect_loop(self):
        """Attempt to reconnect with exponential backoff."""
        backoff = RECONNECT_BASE_DELAY_SECONDS
        
        while not self._connected:
            try:
                await asyncio.sleep(backoff)
                await self._connect()
                if self._connected:
                    logger.info("WebSocket reconnection successful")
                    Metrics.record_realtime_reconnection()
                    return
            except Exception as e:
                logger.warning(f"Reconnection attempt failed: {e}")
                
            backoff = min(backoff * 2, RECONNECT_MAX_DELAY_SECONDS)
            
    def set_transcription_callback(self, callback: Callable[[TranscriptionResult], Awaitable[None]]):
        """Set callback for receiving transcription results."""
        self._transcription_callback = callback
        
    def _to_wire_audio(self, pcm16_bytes: bytes) -> bytes:
        """
        Resample agent PCM (config.sample_rate, 16 kHz) up to the 24 kHz mono
        PCM16 that the realtime wire format expects. Without this, Speaches reads
        16 kHz samples as 24 kHz and the transcript is distorted/garbled.
        """
        src_rate = self.config.sample_rate
        if not pcm16_bytes or src_rate == REALTIME_WIRE_SAMPLE_RATE:
            return pcm16_bytes

        samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
        if samples.size == 0:
            return pcm16_bytes

        # 16k -> 24k reduces to up=3, down=2; use gcd so other rates also work.
        if self._resampler is None:
            g = gcd(REALTIME_WIRE_SAMPLE_RATE, src_rate)
            self._resampler = _StreamingResampler(
                REALTIME_WIRE_SAMPLE_RATE // g, src_rate // g)
        resampled = self._resampler.process(samples.astype(np.float64))
        return np.clip(np.round(resampled), -32768, 32767).astype(np.int16).tobytes()

    async def push_audio(self, audio_data: bytes):
        """
        Push audio data to be transcribed via WebSocket.

        Expects 16-bit PCM mono at config.sample_rate (16 kHz); it is resampled to
        24 kHz here to match the OpenAI Realtime / Speaches wire format.
        """
        if not self._ws or not self._connected:
            return

        try:
            # Resample to the 24 kHz wire rate, then base64-encode for the
            # OpenAI Realtime API input_audio_buffer.append event.
            wire_audio = self._to_wire_audio(audio_data)
            audio_base64 = base64.b64encode(wire_audio).decode('utf-8')
            
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }
            
            await self._ws.send(json.dumps(message))
            self._last_audio_time = time.time()
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket closed while sending audio")
            self._connected = False
            await self._handle_connection_failure()
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            
    async def commit_audio_buffer(self):
        """
        Commit the current audio buffer for transcription.

        Call this when you want to force transcription of buffered audio
        (e.g., after the local VAD detects end-of-turn). Arms a fresh future that
        the receive loop resolves with the resulting transcript.
        """
        if not self._ws or not self._connected:
            # Arm a resolved-empty future so a waiting commit_and_wait() returns
            # immediately rather than blocking on a stale future from a prior turn.
            self._commit_seq += 1
            self._pending_seq = self._commit_seq
            self._inflight_item_id = None
            self._pending_transcript = asyncio.get_running_loop().create_future()
            self._pending_transcript.set_result("")
            return

        try:
            # Advance the turn token before sending the commit so the receive loop
            # can correlate the resulting item/transcript with this turn.
            self._commit_seq += 1
            self._pending_seq = self._commit_seq
            self._inflight_item_id = None
            self._pending_transcript = asyncio.get_running_loop().create_future()
            message = {"type": "input_audio_buffer.commit"}
            await self._ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error committing audio buffer: {e}")
            if self._pending_transcript and not self._pending_transcript.done():
                self._pending_transcript.set_result("")
            
    async def clear_audio_buffer(self):
        """Clear the audio buffer without transcribing."""
        if not self._ws or not self._connected:
            return
            
        try:
            message = {"type": "input_audio_buffer.clear"}
            await self._ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error clearing audio buffer: {e}")
            
    async def commit_and_wait(self, timeout: float) -> str:
        """
        Commit the streamed audio buffer and wait for its transcript.

        Audio is already streaming via push_audio(); the caller invokes this when
        the local VAD detects end-of-turn. Sends input_audio_buffer.commit and
        awaits the future resolved by the receive loop. Returns "" on timeout.
        """
        if not self._connected or not self._ws:
            logger.warning("Realtime connection not available")
            Metrics.record_stt_error(self.model, "connection_unavailable")
            return ""

        with create_span("stt.transcribe.realtime", {
            "stt.model": self.model,
            "stt.mode": "realtime",
        }) as span:
            start_time = time.time()
            await self.commit_audio_buffer()

            result_text = ""
            try:
                result_text = await asyncio.wait_for(self._pending_transcript, timeout)
            except asyncio.TimeoutError:
                logger.debug(f"Realtime transcription timed out after {timeout}s")
                span.set_attribute("stt.timeout", True)

            latency_ms = (time.time() - start_time) * 1000
            span.set_attribute("stt.latency_ms", latency_ms)
            span.set_attribute("stt.text_length", len(result_text))

            if result_text:
                Metrics.record_stt_latency(latency_ms, self.model)

            return result_text
            
    async def close(self):
        """Close the WebSocket connection."""
        self._connected = False
        
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
                
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
                
        if self._ws:
            await self._ws.close()
            self._ws = None
            
        Metrics.record_realtime_connection_state("closed")
        logger.info("Realtime WebSocket client closed")


class RealtimeSTTManager:
    """
    Manager for STT that can use either realtime (WebSocket) or batch mode.
    
    Provides a unified interface regardless of the underlying mode.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.mode = config.stt_mode
        
        self._realtime_client: Optional[RealtimeWebSocketClient] = None
        self._batch_client = None  # WhisperAPIClient - imported lazily to avoid circular imports
        
        self._transcription_callback: Optional[Callable[[str], Awaitable[None]]] = None
        
    async def initialize(self):
        """Initialize the appropriate STT client based on configuration."""
        if self.config.use_realtime_stt and WEBSOCKETS_AVAILABLE:
            logger.info("Initializing STT in realtime (WebSocket) mode")
            self._realtime_client = RealtimeWebSocketClient(self.config)
            await self._realtime_client.initialize()
            
            if self._realtime_client.available and self._realtime_client._connected:
                # Set up callback wrapper
                async def callback_wrapper(result: TranscriptionResult):
                    if result.is_final and result.text and self._transcription_callback:
                        await self._transcription_callback(result.text)
                        
                self._realtime_client.set_transcription_callback(callback_wrapper)
                logger.info("Realtime STT initialized successfully")
                Metrics.record_stt_mode("realtime")
                return
            else:
                logger.warning("Realtime STT unavailable, falling back to batch mode")
                
        # Fallback to batch mode
        logger.info("Initializing STT in batch mode")
        Metrics.record_stt_mode("batch")
        from audio_pipeline import WhisperAPIClient
        self._batch_client = WhisperAPIClient(self.config)
        await self._batch_client.initialize()
        
    @property
    def available(self) -> bool:
        """Check if STT is available."""
        if self._realtime_client and self._realtime_client.available and self._realtime_client._connected:
            return True
        if self._batch_client and self._batch_client.available:
            return True
        return False
        
    @property 
    def is_realtime(self) -> bool:
        """Check if using realtime mode."""
        return (self._realtime_client is not None and 
                self._realtime_client.available and 
                self._realtime_client._connected)
        
    def set_transcription_callback(self, callback: Callable[[str], Awaitable[None]]):
        """Set callback for receiving transcription results (realtime mode only)."""
        self._transcription_callback = callback
        
    async def push_audio(self, audio_data: bytes):
        """Push audio for realtime transcription."""
        if self._realtime_client and self._realtime_client._connected:
            await self._realtime_client.push_audio(audio_data)
            
    async def commit_audio(self):
        """Commit audio buffer for transcription (realtime mode)."""
        if self._realtime_client and self._realtime_client._connected:
            await self._realtime_client.commit_audio_buffer()

    async def clear_audio(self):
        """Discard the streamed server-side audio buffer without transcribing.

        Used when an utterance is dropped (e.g. below min speech duration) so the
        already-streamed audio doesn't bleed into the next turn's transcript.
        No-op in batch mode (there is no server-side streamed buffer)."""
        if self._realtime_client and self._realtime_client._connected:
            await self._realtime_client.clear_audio_buffer()

    async def commit_and_wait(self, timeout: float) -> str:
        """
        Realtime mode: commit the streamed buffer and wait for its transcript.
        Returns "" if not in realtime mode.
        """
        if self._realtime_client and self._realtime_client._connected:
            return await self._realtime_client.commit_and_wait(timeout)
        return ""

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe a full audio buffer. Realtime mode commits the streamed
        buffer (ignoring audio_data, which was already sent); batch uploads it."""
        if self._realtime_client and self._realtime_client._connected:
            return await self._realtime_client.commit_and_wait(self.config.realtime_commit_timeout_s)
        elif self._batch_client and self._batch_client.available:
            return await self._batch_client.transcribe(audio_data)
        else:
            logger.error("No STT client available")
            return ""
            
    async def close(self):
        """Close all clients."""
        if self._realtime_client:
            await self._realtime_client.close()
        if self._batch_client:
            await self._batch_client.close()
