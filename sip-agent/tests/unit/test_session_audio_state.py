"""Unit tests: SessionAudioState isolation.

The audio pipeline is stateless per-call: all mutable per-utterance state
(VAD state machine, utterance buffer, latency metrics) lives on a
SessionAudioState created by new_session_state(). Two states fed interleaved
chunks through the SAME pipeline must never share VAD or buffer state.
"""
import numpy as np
import pytest

from audio_pipeline import LowLatencyAudioPipeline

pytestmark = pytest.mark.unit

SAMPLE_RATE = 16000
CHUNK = np.zeros(SAMPLE_RATE * 20 // 1000, dtype=np.int16).tobytes()  # 20ms


def test_new_session_state_returns_fresh_isolated_state(config):
    pipeline = LowLatencyAudioPipeline(config)
    s1 = pipeline.new_session_state()
    s2 = pipeline.new_session_state()

    assert s1 is not s2
    assert s1.vad is not s2.vad
    assert s1.buffer is not s2.buffer
    assert s1.metrics is not s2.metrics
    assert s1.buffer == bytearray() and s2.buffer == bytearray()


async def test_interleaved_states_do_not_share_vad_or_buffer(
        config, monkeypatch):
    """One 'speaking' call and one silent call interleave chunk-for-chunk on
    the same pipeline: only the speaking state's buffer/VAD may change."""
    pipeline = LowLatencyAudioPipeline(config)
    s1 = pipeline.new_session_state()
    s2 = pipeline.new_session_state()

    # s1's line carries speech; s2's line is silent (classification forced —
    # webrtcvad's judgment on synthetic audio is not under test).
    monkeypatch.setattr(s1.vad, "is_speech",
                        lambda chunk, update_noise=True: True)
    monkeypatch.setattr(s2.vad, "is_speech",
                        lambda chunk, update_noise=True: False)

    for _ in range(5):
        assert await pipeline.process_audio(s1, CHUNK) is None
        assert await pipeline.process_audio(s2, CHUNK) is None

    # Speech accumulated only on s1.
    assert len(s1.buffer) == 5 * len(CHUNK)
    assert s1.vad.is_speaking is True
    assert s1.vad.speech_ms == 5 * config.chunk_duration_ms

    # s2 saw only silence: untouched buffer, untouched VAD.
    assert len(s2.buffer) == 0
    assert s2.vad.is_speaking is False
    assert s2.vad.speech_ms == 0


async def test_has_speech_uses_the_given_state_only(config, monkeypatch):
    pipeline = LowLatencyAudioPipeline(config)
    s1 = pipeline.new_session_state()
    s2 = pipeline.new_session_state()

    # Feed s1's adaptive noise floor only; s2's must stay untouched.
    for _ in range(3):
        pipeline.has_speech(s1, CHUNK, update_noise=True)
    assert len(s1.vad.noise_samples) == 3
    assert len(s2.vad.noise_samples) == 0
