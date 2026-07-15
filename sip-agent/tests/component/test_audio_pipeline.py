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
    # Per-call audio state (the pipeline itself is stateless per-call now).
    state = pipeline.new_session_state()
    # Simulate a buffered utterance, then drive silence to trigger end-of-turn.
    state.buffer.extend(_audio(300))  # > min_speech_duration_ms (200)
    state.vad.is_speaking = True

    result = None
    for _ in range(200):
        result = await pipeline.process_audio(state, _audio(20))  # silent 20ms chunks
        if result:
            break
    assert result == MOCK_TRANSCRIPT


async def test_synthesize_sanitizes_text_for_tts(pipeline):
    import mock_speaches
    await pipeline.synthesize("**bold** hi 🎉")
    assert mock_speaches.TTS_REQUESTS[-1] == "bold hi"


# --- Per-session realtime STT (04b) ------------------------------------------

async def test_session_stt_is_noop_in_batch_mode(pipeline):
    state = pipeline.new_session_state()
    await pipeline.start_session_stt(state)
    assert state.realtime is None
    # Idempotent close with nothing attached.
    await pipeline.stop_session_stt(state)
    await pipeline.stop_session_stt(None)


async def test_session_stt_cap_falls_back_to_batch(comp_config):
    """At the MAX_CONCURRENT_CALLS connection cap the session gets no
    dedicated realtime client and silently uses the shared batch path."""
    from types import SimpleNamespace
    p = LowLatencyAudioPipeline(comp_config)  # cap defaults to 1
    p._stt_manager = SimpleNamespace(is_realtime=True)
    p._session_realtime_clients = {object()}  # one connection already live

    state = p.new_session_state()
    await p.start_session_stt(state)
    assert state.realtime is None


async def test_session_stt_detach_closes_client(comp_config):
    p = LowLatencyAudioPipeline(comp_config)
    closed = []

    class FakeClient:
        async def close(self):
            closed.append(True)

    state = p.new_session_state()
    client = FakeClient()
    state.realtime = client
    p._session_realtime_clients.add(client)

    await p.stop_session_stt(state)
    assert closed == [True]
    assert state.realtime is None
    assert client not in p._session_realtime_clients
    # Second call is a no-op (both teardown paths can reach it).
    await p.stop_session_stt(state)
    assert closed == [True]


async def test_fallback_session_uses_batch_never_shared_realtime(comp_config):
    """A session WITHOUT its own realtime connection (connect failed / cap
    reached) must transcribe its locally buffered audio through the shared
    BATCH client. Streaming into — or committing/clearing — the shared
    realtime manager would interleave concurrent calls' audio in one
    server-side buffer and cross-contaminate transcripts (spec 04b §3)."""
    from types import SimpleNamespace

    p = LowLatencyAudioPipeline(comp_config)

    manager_calls = []

    async def _mgr_push(audio):
        manager_calls.append("push")

    async def _mgr_commit(timeout):
        manager_calls.append("commit")
        return "WRONG CALL'S WORDS"

    async def _mgr_clear():
        manager_calls.append("clear")

    p._stt_manager = SimpleNamespace(
        is_realtime=True, available=True, push_audio=_mgr_push,
        commit_and_wait=_mgr_commit, clear_audio=_mgr_clear)

    batch_calls = []

    async def _batch_transcribe(audio):
        batch_calls.append(audio)
        return "batch transcript"

    p._stt_batch_client = SimpleNamespace(
        available=True, transcribe=_batch_transcribe)

    # Per-session connect failed -> state.realtime stays None.
    state = p.new_session_state()
    state.buffer.extend(_audio(300))  # > min_speech_duration_ms
    state.vad.is_speaking = True

    result = None
    for _ in range(200):
        result = await p.process_audio(state, _audio(20))  # silence
        if result:
            break

    assert result == "batch transcript"
    assert len(batch_calls) == 1
    assert manager_calls == []  # the shared realtime WS was never touched

    # A sub-threshold (noise) discard must not clear the shared buffer
    # either — that could wipe ANOTHER caller's streamed utterance.
    state.buffer.extend(_audio(10))
    assert await p._transcribe_buffer(state) == ""
    assert manager_calls == []
