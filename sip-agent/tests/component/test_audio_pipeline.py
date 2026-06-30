"""Component tests for LowLatencyAudioPipeline against the mock Speaches server.

Drives STT and TTS over real HTTP to the in-process mock, so the multipart
upload, WAV unwrap/resample, and phrase precache code paths all execute.
"""
import numpy as np
import pytest
import pytest_asyncio

from audio_pipeline import LowLatencyAudioPipeline
from mock_speaches import MOCK_TRANSCRIPT

pytestmark = pytest.mark.component


@pytest_asyncio.fixture
async def pipeline(comp_config):
    p = LowLatencyAudioPipeline(comp_config)
    await p.start()
    yield p
    await p.stop()


def _audio(ms: int, rate: int = 16000) -> bytes:
    n = rate * ms // 1000
    return np.zeros(n, dtype=np.int16).tobytes()


async def test_tts_available_and_precached(pipeline):
    assert pipeline.tts.available is True
    # Greetings are precached at startup from config.phrases.
    greeting = pipeline.config.phrases.greetings[0]
    cached = pipeline.get_cached_audio(greeting)
    assert cached is not None and len(cached) > 0


async def test_synthesize_returns_resampled_pcm(pipeline):
    audio = await pipeline.synthesize("an uncached sentence please")
    assert isinstance(audio, (bytes, bytearray))
    assert len(audio) > 0
    # Raw int16 PCM -> even number of bytes.
    assert len(audio) % 2 == 0


async def test_stt_transcribes_via_speaches(pipeline):
    # Batch mode: pipeline.stt is the WhisperAPIClient.
    assert pipeline.stt.available is True
    text = await pipeline.stt.transcribe(_audio(300))
    assert text == MOCK_TRANSCRIPT


async def test_process_audio_returns_transcript_on_end_of_utterance(pipeline):
    # Simulate a buffered utterance, then drive silence to trigger end-of-turn.
    pipeline.audio_buffer.extend(_audio(300))  # > min_speech_duration_ms (200)
    pipeline.vad.is_speaking = True

    result = None
    for _ in range(200):
        result = await pipeline.process_audio(_audio(20))  # silent 20ms chunks
        if result:
            break
    assert result == MOCK_TRANSCRIPT
