"""
Low-Latency Audio Pipeline with Speaches (Unified STT + TTS)
=============================================================
All ML inference offloaded to a single Speaches API server:
- Whisper API for STT (OpenAI-compatible /v1/audio/transcriptions)
- Piper/Kokoro for TTS (OpenAI-compatible /v1/audio/speech)

This simplifies deployment to a single ML service container.
"""

import asyncio
import io
import logging
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional, Tuple

import httpx
import numpy as np

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from config import Config
from endpointing import suggest_timeout_ms
from speech_text import sanitize_for_speech
from telemetry import create_span, Metrics
from logging_utils import log_event
from retry_utils import retry_async, RetryError


logger = logging.getLogger(__name__)


def decode_audio_to_pcm16(data: bytes, target_rate: int) -> bytes:
    """Decode an audio file into 16-bit mono PCM at target_rate — the format
    SIPHandler.send_audio() plays.

    Accepts whatever libsndfile can read (WAV, FLAC, OGG; MP3 with
    libsndfile >= 1.1). CPU-bound: call via run_in_executor off the event
    loop. Raises ValueError on undecodable or empty input.
    """
    import soundfile as sf

    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:
        raise ValueError(f"could not decode audio: {e}") from e
    if audio.size == 0:
        raise ValueError("audio file contains no samples")

    mono = audio.mean(axis=1)
    if int(rate) != int(target_rate):
        if SCIPY_AVAILABLE:
            import math
            g = math.gcd(int(rate), int(target_rate))
            mono = scipy.signal.resample_poly(
                mono, int(target_rate) // g, int(rate) // g)
        else:
            new_indices = np.linspace(0, len(mono) - 1,
                                      int(len(mono) * target_rate / rate))
            mono = np.interp(new_indices, np.arange(len(mono)), mono)

    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


# ============================================================================
# Latency Tracking
# ============================================================================

@dataclass
class LatencyMetrics:
    """Track latency at each stage."""
    vad_start: float = 0
    speech_end: float = 0
    stt_start: float = 0
    stt_end: float = 0
    llm_first_token: float = 0
    llm_complete: float = 0
    tts_first_chunk: float = 0
    audio_start: float = 0
    
    def log_summary(self):
        """Log latency breakdown."""
        if self.speech_end and self.stt_end:
            stt_latency = (self.stt_end - self.speech_end) * 1000
            logger.info(f"STT latency: {stt_latency:.0f}ms")
        if self.stt_end and self.llm_first_token:
            llm_ttft = (self.llm_first_token - self.stt_end) * 1000
            logger.info(f"LLM TTFT: {llm_ttft:.0f}ms")
        if self.llm_first_token and self.tts_first_chunk:
            tts_latency = (self.tts_first_chunk - self.llm_first_token) * 1000
            logger.info(f"TTS first chunk: {tts_latency:.0f}ms")
        if self.speech_end and self.audio_start:
            total = (self.audio_start - self.speech_end) * 1000
            logger.info(f"Total response latency: {total:.0f}ms")


# ============================================================================
# Optimized VAD with Shorter Timeouts
# ============================================================================

class FastVoiceActivityDetector:
    """
    Optimized VAD with aggressive silence detection.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.sample_rate
        
        self.vad = None
        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(3)  # Mode 3 = most aggressive
            
        self.speech_frames = deque(maxlen=50)
        self.silence_frames = 0
        # Trailing silence (ms) since the last speech chunk. Measured from the
        # chunks' real durations, NOT frames * chunk_duration_ms: the audio
        # loop's reads are variable-length (sip_handler.receive_audio returns
        # up to 100ms at a time), so a frame count says nothing about elapsed
        # audio time.
        self.silence_ms = 0.0
        self.is_speaking = False
        # Cumulative speech (ms) in the current utterance — feeds the
        # adaptive endpointing timeout (endpointing.suggest_timeout_ms).
        self.speech_ms = 0.0

        self.noise_floor = 200
        self.noise_samples = deque(maxlen=100)
        
        self.silence_timeout_ms = config.silence_duration_ms
        
    # Lower bound for the adaptive noise floor. Without it, sustained digital
    # silence (zero-energy RTP) drives the floor to 0.0, after which any
    # nonzero dither/line noise exceeds "floor * 2" and reads as speech —
    # phantom barge-ins on a silent line.
    MIN_NOISE_FLOOR = 50.0

    def is_speech(self, audio_chunk: bytes, update_noise: bool = True) -> bool:
        """Check if chunk contains speech with energy pre-filter.

        ``update_noise=False`` makes the check side-effect-free on the
        adaptive noise floor: the audio loop calls both has_speech()
        (barge-in check) and process_audio() on the SAME chunk, and letting
        both feed noise_samples would double-count every chunk in the floor.
        """
        samples = np.frombuffer(audio_chunk, dtype=np.int16)
        energy = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

        if not self.is_speaking and update_noise:
            self.noise_samples.append(energy)
            if len(self.noise_samples) >= 10:
                self.noise_floor = max(
                    float(np.percentile(list(self.noise_samples), 30)),
                    self.MIN_NOISE_FLOOR)

        if energy < self.noise_floor * 1.5:
            return False
            
        if self.vad:
            try:
                frame_size = int(self.sample_rate * 0.03) * 2  # 30ms
                for i in range(0, len(audio_chunk), frame_size):
                    frame = audio_chunk[i:i + frame_size]
                    if len(frame) == frame_size:
                        if self.vad.is_speech(frame, self.sample_rate):
                            return True
                return False
            except Exception:
                pass
                
        return bool(energy > self.noise_floor * 2)

    def process_audio(self, audio_chunk: bytes,
                      silence_timeout_ms: Optional[int] = None) -> Tuple[bool, bool]:
        """Process audio with faster end-of-utterance detection.

        ``silence_timeout_ms`` overrides the configured fixed timeout for
        this chunk (adaptive/speculative endpointing); None keeps the
        configured value (existing behavior).
        """
        is_speech = self.is_speech(audio_chunk)

        # This chunk's real duration. receive_audio() hands us anything from
        # one 20ms frame up to 100ms, so both timers must be driven by the
        # actual byte count; crediting a flat chunk_duration_ms per chunk made
        # a 100ms read count as 20ms and stretched every configured timeout by
        # up to 5x (a 750ms hangover became ~3.8s of dead air before STT ran).
        chunk_ms = len(audio_chunk) / 2 / self.sample_rate * 1000

        if is_speech:
            self.speech_frames.append(audio_chunk)
            self.silence_frames = 0
            self.silence_ms = 0.0
            if not self.is_speaking:
                # Speech just started - record VAD event
                Metrics.record_vad_speech_segment()
                self.speech_ms = 0.0
            self.is_speaking = True
            self.speech_ms += chunk_ms
        else:
            self.silence_frames += 1
            self.silence_ms += chunk_ms

        timeout_ms = (silence_timeout_ms if silence_timeout_ms is not None
                      else self.silence_timeout_ms)
        end_of_utterance = (
            self.is_speaking and
            self.silence_ms >= timeout_ms
        )

        if end_of_utterance:
            self.is_speaking = False

        return is_speech, end_of_utterance

    def reset(self):
        """Reset state."""
        self.speech_frames.clear()
        self.silence_frames = 0
        self.silence_ms = 0.0
        self.is_speaking = False
        self.speech_ms = 0.0


# ============================================================================
# Per-session audio state
# ============================================================================

@dataclass
class SessionAudioState:
    """All mutable per-call audio state, owned by one CallSession.

    The pipeline itself holds only shared, concurrency-safe resources (HTTP
    clients, the TTS phrase cache, config); everything a single call mutates
    per chunk/utterance — the VAD state machine, the utterance buffer, the
    latency metrics — lives here so two sessions can never corrupt each
    other. Created via LowLatencyAudioPipeline.new_session_state().
    """

    vad: FastVoiceActivityDetector
    buffer: bytearray = field(default_factory=bytearray)
    metrics: LatencyMetrics = field(default_factory=LatencyMetrics)
    # Per-session realtime STT connection (a RealtimeWebSocketClient), attached
    # by LowLatencyAudioPipeline.start_session_stt() in realtime mode and
    # closed by stop_session_stt() at teardown. None in batch mode, when the
    # connection cap is reached, or when the connect failed (the session then
    # falls back to the shared batch client).
    realtime: Optional[Any] = None


# ============================================================================
# Whisper API Client (OpenAI-compatible) - via Speaches
# ============================================================================

class WhisperAPIClient:
    """
    Whisper API client using OpenAI-compatible endpoints.
    Uses Speaches server for transcription.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.speaches_api_url.rstrip('/')
        self.model = config.whisper_model
        self.language = config.whisper_language
        self.client: Optional[httpx.AsyncClient] = None
        self.available = False
        
    async def initialize(self):
        """Initialize the API client and ensure model is downloaded."""
        self.client = httpx.AsyncClient(timeout=120.0)  # Longer timeout for model download
        
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                logger.info(f"Whisper API (Speaches) available at {self.base_url}")
                
                # Ensure the STT model is downloaded
                await self._ensure_model_downloaded()

                self.available = True

                # Force the model into memory now; Speaches loads Whisper
                # lazily on the first transcription, which on a busy GPU can
                # take minutes and would otherwise stall the first real call.
                await self._warm_up()
            else:
                logger.warning(f"Whisper API returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Whisper API not available: {e}")
            self.available = False
            
    async def _ensure_model_downloaded(self):
        """
        Ensure the Whisper model is downloaded.
        Speaches will auto-download on first use, but we can trigger it early.
        """
        try:
            import urllib.parse
            encoded_model = urllib.parse.quote(self.model, safe='')
            
            # Check if model exists by trying to get model info
            response = await self.client.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                models = response.json().get('data', [])
                model_ids = [m.get('id', '') for m in models]
                
                if self.model in model_ids:
                    logger.info(f"STT model '{self.model}' is already available")
                    return
                    
            # Model not found, trigger download
            logger.info(f"Downloading STT model: {self.model}")
            logger.info("This may take a few minutes on first run...")
            
            response = await self.client.post(
                f"{self.base_url}/v1/models/{encoded_model}",
                timeout=300.0
            )
            
            if response.status_code in (200, 201):
                logger.info(f"STT model '{self.model}' download initiated/completed")
            else:
                logger.warning(f"STT model download response: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Could not pre-download STT model: {e}")
            # Continue anyway - Speaches will download on first use

    async def _warm_up(self):
        """Transcribe a short silent clip so the model is loaded before the first call."""
        try:
            silence = b'\x00\x00' * int(self.config.sample_rate * 0.5)
            start = time.time()
            await self.transcribe(silence)
            logger.info(f"STT warm-up completed in {(time.time() - start) * 1000:.0f}ms")
        except Exception as e:
            logger.warning(f"STT warm-up failed: {e}")

    async def close(self):
        """Close the client."""
        if self.client:
            await self.client.aclose()
            
    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio using OpenAI-compatible API with retry logic.
        """
        if not self.available or not self.client:
            logger.warning("Whisper API not available")
            Metrics.record_stt_error(self.model, "api_unavailable")
            return ""
        
        # Calculate audio duration for metrics
        audio_duration_s = len(audio_data) / (self.config.sample_rate * 2)  # 16-bit audio = 2 bytes per sample
        
        with create_span("stt.transcribe", {
            "stt.model": self.model,
            "stt.language": self.language,
            "audio.bytes": len(audio_data),
            "audio.duration_s": audio_duration_s
        }) as span:
            start_time = time.time()
            
            async def do_transcribe():
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(self.config.sample_rate)
                    wav.writeframes(audio_data)
                wav_buffer.seek(0)
                
                files = {
                    'file': ('audio.wav', wav_buffer, 'audio/wav')
                }
                data = {
                    'model': self.model,
                    'language': self.language,
                    'response_format': 'json'
                }
                
                response = await self.client.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                return response.json()
            
            try:
                result = await retry_async(
                    do_transcribe,
                    api_name="stt",
                    config=self.config,
                    retryable_exceptions=(httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException),
                )
                
                latency_ms = (time.time() - start_time) * 1000
                text = result.get('text', '').strip()
                
                # Record metrics
                span.set_attribute("stt.text_length", len(text))
                span.set_attribute("stt.latency_ms", latency_ms)
                Metrics.record_stt_latency(latency_ms, self.model)
                Metrics.record_stt_audio_duration(audio_duration_s)
                
                # Record confidence if available (Whisper API may not always provide this)
                if 'confidence' in result:
                    confidence = result.get('confidence', 1.0)
                    Metrics.record_stt_confidence(confidence, self.model)
                    span.set_attribute("stt.confidence", confidence)
                
                return text
                
            except RetryError as e:
                latency_ms = (time.time() - start_time) * 1000
                span.set_attribute("stt.latency_ms", latency_ms)
                span.set_attribute("error", str(e))
                Metrics.record_stt_error(self.model, "retry_exhausted")
                logger.error(f"STT transcription failed after retries: {e}")
                return ""
            except asyncio.TimeoutError:
                logger.error("STT request timeout")
                Metrics.record_stt_error(self.model, "timeout")
                span.set_attribute("error.type", "timeout")
                return ""
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                span.record_exception(e)
                Metrics.record_stt_error(self.model, type(e).__name__)
                return ""


# ============================================================================
# Speaches TTS Client (OpenAI-compatible /v1/audio/speech)
# ============================================================================

class SpeachesTTSClient:
    """
    TTS client using Speaches OpenAI-compatible API.
    
    Uses the /v1/audio/speech endpoint with Piper or Kokoro models.
    Returns audio in the configured format (wav by default).
    
    Supported models:
    - speaches-ai/Kokoro-82M-v1.0-ONNX (recommended, high quality)
    - hexgrad/Kokoro-82M (alternative Kokoro)
    - Piper voices via rhasspy/* repos (e.g., rhasspy/piper-voice-en_US-lessac-medium)
    """
    
    # Sample rates for different TTS backends
    PIPER_SAMPLE_RATE = 22050
    KOKORO_SAMPLE_RATE = 24000

    # Response formats that are raw little-endian int16 PCM (wav is unwrapped to
    # raw PCM by _extract_wav_data). Anything else (mp3/opus/aac/flac) is a
    # compressed byte stream that must NOT be reinterpreted as int16 samples.
    RAW_PCM_FORMATS = frozenset({"wav", "pcm"})
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.speaches_api_url.rstrip('/')
        self.model = config.tts_model
        self.voice = config.tts_voice
        self.response_format = config.tts_response_format
        self.speed = config.tts_speed
        self.available = False
        
        # Audio cache for common phrases
        self.audio_cache: dict = {}
        self.cache_enabled = True
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
        # Determine expected sample rate based on model
        if 'kokoro' in self.model.lower():
            self.tts_sample_rate = self.KOKORO_SAMPLE_RATE
        else:
            self.tts_sample_rate = self.PIPER_SAMPLE_RATE
        
    async def initialize(self):
        """Test connection to Speaches TTS API and ensure model is downloaded."""
        self.client = httpx.AsyncClient(timeout=120.0)  # Longer timeout for model download
        
        try:
            # Test the health endpoint
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code != 200:
                logger.warning(f"Speaches TTS health check failed: {response.status_code}")
                return
                
            logger.info(f"Speaches TTS available at {self.base_url}")
            
            # Ensure the TTS model is downloaded
            if not await self._ensure_model_downloaded():
                logger.error(f"Failed to download TTS model: {self.model}")
                return
                
            self.available = True
            logger.info(f"TTS model: {self.model}, voice: {self.voice}")
            
            # Pre-cache common phrases
            if self.cache_enabled:
                await self._precache_phrases()
                
        except Exception as e:
            logger.warning(f"Speaches TTS not available: {e}")
            self.available = False
            
    async def _ensure_model_downloaded(self) -> bool:
        """
        Check if model is installed, download if not.
        
        Speaches uses POST /v1/models/{model_id} to download models.
        """
        try:
            # First, try a test synthesis to see if model is already available
            test_response = await self.client.post(
                f"{self.base_url}/v1/audio/speech",
                json={
                    "model": self.model,
                    "voice": self.voice,
                    "input": "test",
                    "response_format": self.response_format,
                },
                timeout=10.0
            )
            
            if test_response.status_code == 200:
                logger.info(f"TTS model '{self.model}' is already available")
                return True
                
            # Check if it's a "model not installed" error
            if test_response.status_code == 404:
                error_detail = test_response.json().get('detail', '')
                if 'not installed' in error_detail.lower():
                    logger.info(f"TTS model '{self.model}' not installed, downloading...")
                    return await self._download_model()
                    
            logger.error(f"TTS test failed: {test_response.status_code} - {test_response.text}")
            return False
            
        except Exception as e:
            logger.error(f"Error checking TTS model availability: {e}")
            return False
            
    async def _download_model(self) -> bool:
        """
        Download a model via POST /v1/models/{model_id}.
        """
        try:
            # URL-encode the model ID for the path
            import urllib.parse
            encoded_model = urllib.parse.quote(self.model, safe='')
            
            logger.info(f"Downloading TTS model: {self.model}")
            logger.info("This may take a few minutes on first run...")
            
            # POST to download the model
            response = await self.client.post(
                f"{self.base_url}/v1/models/{encoded_model}",
                timeout=300.0  # 5 minute timeout for model download
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully downloaded TTS model: {self.model}")
                return True
            elif response.status_code == 201:
                logger.info(f"TTS model download started: {self.model}")
                # Wait a bit for model to be ready
                await asyncio.sleep(5)
                return True
            else:
                logger.error(f"Failed to download TTS model: {response.status_code} - {response.text}")
                return False
                
        except asyncio.TimeoutError:
            logger.error("TTS model download timed out (>5 minutes)")
            return False
        except Exception as e:
            logger.error(f"Error downloading TTS model: {e}")
            return False
            
    async def _precache_phrases(self):
        """Pre-synthesize common phrases from config."""
        phrases = self.config.phrases.get_all_phrases_for_cache()
        
        logger.info(f"Pre-caching {len(phrases)} phrases...")
        
        for phrase in phrases:
            try:
                audio = await self._synthesize_raw(phrase)
                if audio:
                    # Only resample raw PCM; compressed formats must not be
                    # reinterpreted as int16 (see _is_raw_pcm_format).
                    if self._is_raw_pcm_format():
                        audio = self._resample(audio, self.tts_sample_rate, self.config.sample_rate)
                    self.audio_cache[phrase.lower()] = audio
            except Exception as e:
                logger.warning(f"Failed to cache '{phrase}': {e}")
                
        logger.info(f"Cached {len(self.audio_cache)} phrases")
        
    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
        
    def get_cached(self, text: str) -> Optional[bytes]:
        """Get pre-cached audio if available."""
        return self.audio_cache.get(text.lower().strip())

    def _is_raw_pcm_format(self) -> bool:
        """True when synthesized audio is raw int16 PCM and safe to resample.

        Compressed formats (mp3/opus/aac/flac) would be reinterpreted as int16
        samples by _resample, producing noise — so callers must skip resampling
        for those and treat the bytes as an opaque encoded stream.
        """
        return self.response_format.lower() in self.RAW_PCM_FORMATS
        
    async def _synthesize_raw(self, text: str) -> bytes:
        """
        Synthesize text using Speaches TTS API (OpenAI-compatible).
        """
        if not self.available or not self.client:
            Metrics.record_tts_error(self.model, "api_unavailable")
            return b''
        
        with create_span("tts.synthesize", {
            "tts.model": self.model,
            "tts.voice": self.voice,
            "tts.text_length": len(text)
        }) as span:
            start_time = time.time()
            try:
                # Build request payload (OpenAI-compatible format)
                payload = {
                    "model": self.model,
                    "voice": self.voice,
                    "input": text,
                    "response_format": self.response_format,
                    "speed": self.speed
                }
                
                response = await self.client.post(
                    f"{self.base_url}/v1/audio/speech",
                    json=payload,
                    timeout=30.0
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    audio_data = response.content
                    
                    # If response is WAV, extract raw PCM data
                    if self.response_format == "wav" and audio_data[:4] == b'RIFF':
                        audio_data = self._extract_wav_data(audio_data)
                    
                    # Calculate audio duration (16-bit mono at tts_sample_rate)
                    audio_duration_s = len(audio_data) / (self.tts_sample_rate * 2)
                    
                    span.set_attribute("tts.audio_bytes", len(audio_data))
                    span.set_attribute("tts.audio_duration_s", audio_duration_s)
                    span.set_attribute("tts.latency_ms", latency_ms)
                    
                    # Record metrics
                    Metrics.record_tts_latency(latency_ms, self.model)
                    Metrics.record_tts_characters(len(text), self.model)
                    Metrics.record_tts_audio_duration(audio_duration_s, self.model)
                    
                    return audio_data
                else:
                    logger.error(f"TTS API error: {response.status_code} - {response.text}")
                    span.set_attribute("error", True)
                    span.set_attribute("http.status_code", response.status_code)
                    Metrics.record_tts_error(self.model, f"http_{response.status_code}")
                    return b''
                    
            except asyncio.TimeoutError:
                logger.warning("TTS response timeout")
                span.set_attribute("error", True)
                span.set_attribute("error.type", "timeout")
                Metrics.record_tts_error(self.model, "timeout")
                return b''
            except Exception as e:
                logger.error(f"TTS synthesis error: {e}")
                span.record_exception(e)
                Metrics.record_tts_error(self.model, type(e).__name__)
                return b''
            
    def _extract_wav_data(self, wav_bytes: bytes) -> bytes:
        """Extract raw PCM data from WAV file, also detect sample rate."""
        try:
            wav_buffer = io.BytesIO(wav_bytes)
            with wave.open(wav_buffer, 'rb') as wav:
                # Update sample rate from actual file
                self.tts_sample_rate = wav.getframerate()
                return wav.readframes(wav.getnframes())
        except Exception as e:
            logger.warning(f"Failed to extract WAV data: {e}")
            # Return as-is if extraction fails
            return wav_bytes
            
    async def synthesize(self, text: str) -> bytes:
        """Synthesize with cache check and resampling."""
        # Check cache first
        cached = self.get_cached(text)
        if cached:
            logger.debug(f"Cache hit for: {text}")
            return cached
            
        if not self.available:
            logger.warning("Speaches TTS not available")
            return b''
            
        start = time.time()
        audio = await self._synthesize_raw(text)
        
        if audio:
            if self._is_raw_pcm_format():
                # Resample to target rate (usually 16000Hz for SIP)
                audio = self._resample(audio, self.tts_sample_rate, self.config.sample_rate)
            else:
                # Compressed format: resampling as int16 would emit noise. Pass the
                # encoded bytes through un-resampled rather than corrupting them.
                logger.warning(
                    f"TTS_RESPONSE_FORMAT='{self.response_format}' is not raw PCM; "
                    "skipping resample (audio left at native rate, not int16-resampled)"
                )

            elapsed = (time.time() - start) * 1000
            logger.info(f"Speaches TTS: {elapsed:.0f}ms for '{text[:30]}...'")

        return audio
        
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesis - synthesize whole thing and yield in chunks.
        
        Note: Speaches may support true streaming in future versions.
        """
        audio = await self.synthesize(text)
        
        if audio:
            # Yield in chunks for streaming playback
            chunk_size = 4096
            for i in range(0, len(audio), chunk_size):
                yield audio[i:i + chunk_size]
                
    def _resample(self, audio: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample audio to target sample rate."""
        if from_rate == to_rate:
            return audio
            
        samples = np.frombuffer(audio, dtype=np.int16)
        if len(samples) == 0:
            return audio
            
        if SCIPY_AVAILABLE:
            ratio = to_rate / from_rate
            new_len = int(len(samples) * ratio)
            resampled = scipy.signal.resample(samples.astype(np.float64), new_len)
        else:
            # Linear interpolation fallback
            ratio = to_rate / from_rate
            new_indices = np.linspace(0, len(samples) - 1, int(len(samples) * ratio))
            resampled = np.interp(new_indices, np.arange(len(samples)), samples.astype(np.float64))
            
        return resampled.astype(np.int16).tobytes()


# ============================================================================
# Optimized Audio Pipeline (API-based with Speaches)
# ============================================================================

class LowLatencyAudioPipeline:
    """
    Low-latency audio pipeline using Speaches for both STT and TTS.
    
    Supports two STT modes:
    - "realtime": WebRTC streaming for lowest latency (default)
    - "batch": Traditional file upload for compatibility
    
    Target latencies:
    - STT: < 200ms (realtime) / < 300ms (batch)
    - LLM TTFT: < 500ms  
    - TTS: < 150ms (Piper via Speaches)
    - Total: < 850ms (realtime) / < 950ms (batch)
    """
    
    def __init__(self, config: Config):
        self.config = config

        # Shared components (concurrency-safe across sessions)
        self.tts = SpeachesTTSClient(config)

        # STT - use RealtimeSTTManager which handles mode selection. The
        # manager stays the shared/singleton path (mode probing, batch
        # fallback); concurrent calls get their own RealtimeWebSocketClient
        # via start_session_stt(), attached to their SessionAudioState.
        # Sessions WITHOUT their own connection transcribe through the shared
        # batch client (_stt_batch_client) — never the manager's single
        # realtime WebSocket, which cannot be shared across calls.
        self._stt_manager = None  # Initialized in start()
        self._stt_batch_client = None  # Fallback for when realtime unavailable
        # Live per-session realtime clients (capped at MAX_CONCURRENT_CALLS).
        self._session_realtime_clients: set = set()

        self.max_buffer_size = int(config.max_speech_duration_s * config.sample_rate * 2)

        # Realtime mode state
        self._realtime_transcription_callback = None
        self._use_realtime = config.use_realtime_stt

    def new_session_state(self) -> SessionAudioState:
        """Fresh per-call audio state (VAD + utterance buffer + metrics).

        Every consumer of process_audio()/has_speech() owns one of these for
        the duration of its call, so per-utterance state never leaks between
        concurrent (or successive) calls.
        """
        return SessionAudioState(vad=FastVoiceActivityDetector(self.config))

    async def start_session_stt(self, state: SessionAudioState) -> None:
        """Attach a per-session realtime STT connection (realtime mode only).

        One RealtimeWebSocketClient per live session, capped at
        MAX_CONCURRENT_CALLS concurrent connections. On any failure (cap
        reached, connect error) the session simply keeps ``state.realtime``
        unset and transcribes through the shared batch path — the existing
        fallback pattern. No-op in batch mode (the default).
        """
        if state is None or not (self._stt_manager and self._stt_manager.is_realtime):
            return
        cap = max(1, getattr(self.config, "max_concurrent_calls", 1))
        if len(self._session_realtime_clients) >= cap:
            logger.warning(
                "Realtime STT connection cap reached "
                f"({cap}); session falls back to batch STT")
            return
        try:
            from realtime_client import RealtimeWebSocketClient
            client = RealtimeWebSocketClient(self.config)
            await client.initialize()
            if client.available and getattr(client, "_connected", False):
                state.realtime = client
                self._session_realtime_clients.add(client)
                logger.info("Per-session realtime STT connected")
            else:
                await client.close()
                logger.warning(
                    "Per-session realtime STT connect failed; "
                    "session falls back to batch STT")
        except Exception as e:
            logger.warning(f"Per-session realtime STT unavailable, "
                           f"session falls back to batch STT: {e}")

    async def stop_session_stt(self, state: Optional[SessionAudioState]) -> None:
        """Close and detach a session's realtime STT connection (idempotent —
        both teardown paths can reach it)."""
        client = getattr(state, "realtime", None) if state is not None else None
        if client is None:
            return
        state.realtime = None
        self._session_realtime_clients.discard(client)
        try:
            await client.close()
        except Exception as e:
            logger.debug(f"Error closing per-session realtime STT: {e}")

    @property
    def stt(self):
        """Get the active STT client for compatibility."""
        if self._stt_manager:
            return self._stt_manager
        return self._stt_batch_client
        
    async def start(self):
        """Initialize all components."""
        logger.info("Starting low-latency audio pipeline...")
        logger.info(f"STT mode: {self.config.stt_mode}")
        
        start = time.time()
        
        # Try to initialize realtime STT if configured
        if self._use_realtime:
            try:
                from realtime_client import RealtimeSTTManager
                self._stt_manager = RealtimeSTTManager(self.config)
                await self._stt_manager.initialize()
                
                if self._stt_manager.is_realtime:
                    logger.info("Using WebRTC realtime STT mode")
                else:
                    logger.info("Realtime unavailable, using batch STT mode")
            except ImportError as e:
                logger.warning(f"Realtime client not available: {e}")
                self._use_realtime = False
            except Exception as e:
                logger.warning(f"Failed to initialize realtime STT: {e}")
                self._use_realtime = False
                
        # Fallback to batch mode if realtime not available
        if not self._stt_manager or not self._stt_manager.available:
            logger.info("Initializing batch STT client")
            self._stt_batch_client = WhisperAPIClient(self.config)
            await self._stt_batch_client.initialize()
        elif self._stt_manager.is_realtime:
            # Realtime mode ALSO needs the shared batch client: sessions that
            # can't get their own realtime connection (cap reached, connect
            # failure) transcribe their locally buffered audio through it.
            # They must never share the manager's single realtime WebSocket —
            # two concurrent calls' audio would interleave in one server-side
            # buffer and cross-contaminate transcripts.
            logger.info("Initializing shared batch STT client "
                        "(fallback for sessions without a realtime connection)")
            self._stt_batch_client = WhisperAPIClient(self.config)
            await self._stt_batch_client.initialize()
            
        # Initialize TTS
        await self.tts.initialize()
        
        load_time = (time.time() - start) * 1000
        logger.info(f"Pipeline ready in {load_time:.0f}ms")
        
        # Log STT status
        if self._stt_manager and self._stt_manager.available:
            mode_str = "realtime (WebRTC)" if self._stt_manager.is_realtime else "batch"
            logger.info(f"STT ready in {mode_str} mode at {self.config.speaches_api_url}")
        elif self._stt_batch_client and self._stt_batch_client.available:
            logger.info(f"STT ready in batch mode at {self.config.speaches_api_url}")
        else:
            logger.warning("STT not available")
            
        # Log TTS status
        if self.tts.available:
            logger.info(f"Speaches TTS ready, {len(self.tts.audio_cache)} phrases cached")
        else:
            logger.warning("Speaches TTS not available")
            
    async def stop(self):
        """Cleanup."""
        # Any per-session realtime connections not yet detached by teardown.
        for client in list(self._session_realtime_clients):
            self._session_realtime_clients.discard(client)
            try:
                await client.close()
            except Exception as e:
                logger.debug(f"Error closing per-session realtime STT: {e}")
        if self._stt_manager:
            await self._stt_manager.close()
        if self._stt_batch_client:
            await self._stt_batch_client.close()
        await self.tts.close()
        
    def set_realtime_transcription_callback(self, callback):
        """
        Set callback for realtime transcription results.
        
        In realtime mode, transcriptions can arrive asynchronously.
        This callback is called with each transcription result.
        """
        self._realtime_transcription_callback = callback
        if self._stt_manager and hasattr(self._stt_manager, 'set_transcription_callback'):
            self._stt_manager.set_transcription_callback(callback)
        
    async def process_audio(self, state: SessionAudioState, audio_chunk: bytes,
                            endpoint_mode: Optional[str] = None) -> Optional[str]:
        """
        Process audio with fast end-of-utterance detection.

        ``state`` is the caller's per-call SessionAudioState (see
        new_session_state) — this method is stateless over the shared HTTP
        clients, so concurrent sessions can safely interleave calls.

        In realtime mode, audio is also streamed for continuous transcription.

        ``endpoint_mode`` overrides config.endpoint_mode for this call site
        (None = the configured mode). Speculative's short cutoff is only safe
        for consumers that also run main.py's hold/merge machinery; paths
        without it (api.py's choice collection) must opt out.
        """
        # In realtime mode, a session with its own connection (state.realtime)
        # streams audio there. Sessions WITHOUT one (per-session connect
        # failed or the connection cap was reached) do not stream at all:
        # they transcribe their locally buffered audio through the shared
        # BATCH client in _transcribe_buffer. Pushing them into the shared
        # realtime manager would interleave concurrent calls' audio in one
        # server-side buffer and cross-contaminate transcripts.
        if state.realtime is not None:
            await state.realtime.push_audio(audio_chunk)

        # Endpointing: choose this chunk's end-of-utterance silence timeout.
        # fixed -> None (the VAD's configured value, today's behavior).
        # adaptive -> scale by utterance length. No interim transcript exists
        # before the commit — the realtime session runs turn_detection=None
        # and only transcribes after the explicit commit — so adaptive runs
        # audio-only (partial_text=None).
        # speculative -> always the short threshold; main.py's audio loop
        # owns transcript-level completion, hold and merge.
        mode = endpoint_mode if endpoint_mode is not None else self.config.endpoint_mode
        silence_timeout_ms = None
        if mode == "adaptive":
            silence_timeout_ms = suggest_timeout_ms(
                None, state.vad.speech_ms,
                self.config.silence_duration_ms,
                self.config.endpoint_min_silence_ms,
                self.config.endpoint_max_silence_ms)
        elif mode == "speculative":
            silence_timeout_ms = self.config.endpoint_min_silence_ms

        is_speech, end_of_utterance = state.vad.process_audio(
            audio_chunk, silence_timeout_ms=silence_timeout_ms)

        if is_speech:
            state.buffer.extend(audio_chunk)

            if len(state.buffer) > self.max_buffer_size:
                logger.warning("Buffer overflow, forcing transcription")
                return await self._transcribe_buffer(state)

        if end_of_utterance and len(state.buffer) > 0:
            return await self._transcribe_buffer(state)

        return None

    async def _transcribe_buffer(self, state: SessionAudioState) -> str:
        """Transcribe buffered audio via API."""
        state.metrics.speech_end = time.time()

        audio_data = bytes(state.buffer)
        state.buffer.clear()
        state.vad.reset()
        
        duration_ms = len(audio_data) / (self.config.sample_rate * 2) * 1000
        if duration_ms < self.config.min_speech_duration_ms:
            # A session with its own realtime connection already streamed the
            # sub-threshold audio via push_audio(); clear that buffer so it
            # doesn't bleed into the next turn's transcript. Sessions on the
            # batch path never streamed anything, so there is nothing to
            # clear (and touching the shared realtime manager here could wipe
            # ANOTHER call's buffered utterance).
            if state.realtime is not None:
                await state.realtime.clear_audio_buffer()
            return ""

        state.metrics.stt_start = time.time()
        
        # Use the appropriate client. Only a session's OWN realtime
        # connection may use the realtime path: the shared manager's single
        # WebSocket must never be committed on behalf of one session (it
        # could contain another concurrent call's audio). Sessions without
        # their own connection transcribe through the shared batch client.
        if state.realtime is not None:
            # The audio was already streamed via push_audio(); the local VAD
            # just detected end-of-turn, so commit the buffer and wait for
            # the transcript deterministically.
            result = await state.realtime.commit_and_wait(
                self.config.realtime_commit_timeout_s
            )
        elif (self._stt_manager and self._stt_manager.available
                and not self._stt_manager.is_realtime):
            # Manager in batch mode (its internal fallback): safe to share.
            result = await self._stt_manager.transcribe(audio_data)
        elif self._stt_batch_client and self._stt_batch_client.available:
            result = await self._stt_batch_client.transcribe(audio_data)
        else:
            logger.error("No STT client available")
            result = ""

        state.metrics.stt_end = time.time()

        stt_latency = (state.metrics.stt_end - state.metrics.speech_end) * 1000
        mode_str = "realtime" if state.realtime is not None else "batch"
        logger.info(f"STT ({mode_str}): {stt_latency:.0f}ms for {duration_ms:.0f}ms audio")
        
        return result
        
    async def synthesize(self, text: str) -> bytes:
        """Synthesize with caching.

        Sanitizes here — the single choke point every spoken path flows
        through (call turns, greeting, outbound messages, REST /speak, timer
        announcements) — so formatting artifacts never reach the TTS engine.
        """
        return await self.tts.synthesize(sanitize_for_speech(text))

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream synthesis."""
        async for chunk in self.tts.synthesize_stream(sanitize_for_speech(text)):
            yield chunk
            
    def get_cached_audio(self, text: str) -> Optional[bytes]:
        """Get pre-cached audio for instant playback."""
        return self.tts.get_cached(text)
        
    def has_speech(self, state: SessionAudioState, audio_chunk: bytes,
                   update_noise: bool = False) -> bool:
        """Quick speech check against the caller's per-call VAD state.

        Side-effect-free by default (``update_noise=False``): main.py's audio
        loop calls both has_speech() and process_audio() on the SAME chunk,
        and letting both feed the adaptive noise floor would double-count
        every chunk. Callers for which this is the ONLY per-chunk check
        (answering-machine detection — process_audio never runs during the
        AMD window) must pass ``update_noise=True``, or the floor freezes at
        a stale value and steady line noise reads as continuous speech."""
        return state.vad.is_speech(audio_chunk, update_noise=update_noise)
