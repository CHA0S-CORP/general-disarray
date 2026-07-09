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
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Tuple

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
from speech_text import sanitize_for_speech
from telemetry import create_span, Metrics
from logging_utils import log_event
from retry_utils import retry_async, RetryError


logger = logging.getLogger(__name__)


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
        self.is_speaking = False
        
        self.noise_floor = 200
        self.noise_samples = deque(maxlen=100)
        
        self.silence_timeout_ms = config.silence_duration_ms
        
    # Lower bound for the adaptive noise floor. Without it, sustained digital
    # silence (zero-energy RTP) drives the floor to 0.0, after which any
    # nonzero dither/line noise exceeds "floor * 2" and reads as speech —
    # phantom barge-ins on a silent line.
    MIN_NOISE_FLOOR = 50.0

    def is_speech(self, audio_chunk: bytes) -> bool:
        """Check if chunk contains speech with energy pre-filter."""
        samples = np.frombuffer(audio_chunk, dtype=np.int16)
        energy = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

        if not self.is_speaking:
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

    def process_audio(self, audio_chunk: bytes) -> Tuple[bool, bool]:
        """Process audio with faster end-of-utterance detection."""
        is_speech = self.is_speech(audio_chunk)
        
        if is_speech:
            self.speech_frames.append(audio_chunk)
            self.silence_frames = 0
            if not self.is_speaking:
                # Speech just started - record VAD event
                Metrics.record_vad_speech_segment()
            self.is_speaking = True
        else:
            self.silence_frames += 1
            
        silence_ms = self.silence_frames * self.config.chunk_duration_ms
        end_of_utterance = (
            self.is_speaking and 
            silence_ms >= self.silence_timeout_ms
        )
        
        if end_of_utterance:
            self.is_speaking = False
            
        return is_speech, end_of_utterance
        
    def reset(self):
        """Reset state."""
        self.speech_frames.clear()
        self.silence_frames = 0
        self.is_speaking = False


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
        
        # Components
        self.vad = FastVoiceActivityDetector(config)
        self.tts = SpeachesTTSClient(config)
        
        # STT - use RealtimeSTTManager which handles mode selection
        self._stt_manager = None  # Initialized in start()
        self._stt_batch_client = None  # Fallback for when realtime unavailable
        
        # Audio buffer
        self.audio_buffer = bytearray()
        self.max_buffer_size = int(config.max_speech_duration_s * config.sample_rate * 2)
        
        # Realtime mode state
        self._realtime_transcription_callback = None
        self._use_realtime = config.use_realtime_stt
        
        # Metrics
        self.last_metrics = LatencyMetrics()
        
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
        
    async def process_audio(self, audio_chunk: bytes) -> Optional[str]:
        """
        Process audio with fast end-of-utterance detection.
        
        In realtime mode, audio is also streamed for continuous transcription.
        """
        # In realtime mode, push audio for streaming transcription
        if self._stt_manager and self._stt_manager.is_realtime:
            await self._stt_manager.push_audio(audio_chunk)
            
        is_speech, end_of_utterance = self.vad.process_audio(audio_chunk)
        
        if is_speech:
            self.audio_buffer.extend(audio_chunk)
            
            if len(self.audio_buffer) > self.max_buffer_size:
                logger.warning("Buffer overflow, forcing transcription")
                return await self._transcribe_buffer()
                
        if end_of_utterance and len(self.audio_buffer) > 0:
            return await self._transcribe_buffer()
            
        return None
        
    async def _transcribe_buffer(self) -> str:
        """Transcribe buffered audio via API."""
        self.last_metrics.speech_end = time.time()
        
        audio_data = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        self.vad.reset()
        
        duration_ms = len(audio_data) / (self.config.sample_rate * 2) * 1000
        if duration_ms < self.config.min_speech_duration_ms:
            # In realtime mode the sub-threshold audio was already streamed to the
            # server via push_audio(); clear that buffer so it doesn't bleed into
            # the next turn's transcript. Batch mode has no server-side buffer, so
            # this is a no-op there.
            if self._stt_manager and self._stt_manager.is_realtime:
                await self._stt_manager.clear_audio()
            return ""
            
        self.last_metrics.stt_start = time.time()
        
        # Use the appropriate client
        if self._stt_manager and self._stt_manager.available:
            if self._stt_manager.is_realtime:
                # In realtime mode the audio was already streamed via push_audio();
                # the local VAD just detected end-of-turn, so commit the buffer and
                # wait for the transcript deterministically.
                result = await self._stt_manager.commit_and_wait(
                    self.config.realtime_commit_timeout_s
                )
            else:
                result = await self._stt_manager.transcribe(audio_data)
        elif self._stt_batch_client and self._stt_batch_client.available:
            result = await self._stt_batch_client.transcribe(audio_data)
        else:
            logger.error("No STT client available")
            result = ""
            
        self.last_metrics.stt_end = time.time()
        
        stt_latency = (self.last_metrics.stt_end - self.last_metrics.speech_end) * 1000
        mode_str = "realtime" if (self._stt_manager and self._stt_manager.is_realtime) else "batch"
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
        
    def has_speech(self, audio_chunk: bytes) -> bool:
        """Quick speech check."""
        return self.vad.is_speech(audio_chunk)
