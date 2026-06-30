"""Unit tests for FastVoiceActivityDetector's end-of-utterance state machine.

We assert the deterministic parts: silence is never speech, and end-of-utterance
fires after `SILENCE_TIMEOUT_MS` of silence once speaking. Whether webrtcvad
classifies a given waveform as speech is the third-party library's job and is
exercised with real audio in the e2e tier, not pinned here.
"""
import numpy as np
import pytest

from audio_pipeline import FastVoiceActivityDetector

pytestmark = pytest.mark.unit

CHUNK_MS = 20
SAMPLE_RATE = 16000
SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_MS // 1000  # 320 samples / 640 bytes


def silence_chunk() -> bytes:
    return np.zeros(SAMPLES_PER_CHUNK, dtype=np.int16).tobytes()


def test_silence_is_never_speech(config):
    vad = FastVoiceActivityDetector(config)
    for _ in range(50):
        is_speech, end = vad.process_audio(silence_chunk())
        assert is_speech is False
        assert end is False


def test_end_of_utterance_after_silence_timeout(config):
    vad = FastVoiceActivityDetector(config)
    # Simulate that speech has been detected (webrtcvad-independent), then feed
    # silence and confirm end-of-utterance latches after the configured timeout.
    vad.is_speaking = True
    vad.silence_frames = 0

    timeout_ms = config.silence_duration_ms  # default 750ms
    fired_at = None
    for i in range(1, 200):
        is_speech, end = vad.process_audio(silence_chunk())
        if end:
            fired_at = i
            break

    assert fired_at is not None, "end_of_utterance never fired"
    # Should fire right around timeout_ms / chunk_duration_ms frames.
    expected = timeout_ms / CHUNK_MS
    assert abs(fired_at - expected) <= 2
    # Latching end-of-utterance must reset the speaking flag.
    assert vad.is_speaking is False


def test_reset_clears_state(config):
    vad = FastVoiceActivityDetector(config)
    vad.is_speaking = True
    vad.silence_frames = 10
    vad.speech_frames.append(silence_chunk())
    vad.reset()
    assert vad.is_speaking is False
    assert vad.silence_frames == 0
    assert len(vad.speech_frames) == 0
