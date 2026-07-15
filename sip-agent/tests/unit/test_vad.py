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


@pytest.mark.parametrize("chunk_ms", [20, 50, 100])
def test_end_of_utterance_measures_real_time_at_any_chunk_size(config, chunk_ms):
    """The silence timeout must be honored in real milliseconds regardless of
    how much audio a chunk carries.

    sip_handler.receive_audio() returns a variable amount — up to 100ms at a
    time — but the VAD used to credit every chunk a flat config.chunk_duration_ms
    (20ms). A 100ms read therefore counted as 20ms, stretching the configured
    hangover 5x: SILENCE_TIMEOUT_MS=750 meant ~3.8s of dead air before STT ran.
    """
    vad = FastVoiceActivityDetector(config)
    vad.is_speaking = True

    samples = SAMPLE_RATE * chunk_ms // 1000
    chunk = np.zeros(samples, dtype=np.int16).tobytes()

    elapsed_ms = 0.0
    for _ in range(500):
        _, end = vad.process_audio(chunk)
        elapsed_ms += chunk_ms
        if end:
            break

    assert vad.is_speaking is False, "end_of_utterance never fired"
    # Fires at the configured timeout, within one chunk of quantization.
    assert elapsed_ms == pytest.approx(config.silence_duration_ms, abs=chunk_ms)


@pytest.mark.parametrize("chunk_ms", [20, 50, 100])
def test_speech_ms_tracks_real_speech_duration(config, chunk_ms, monkeypatch):
    """speech_ms feeds adaptive endpointing (suggest_timeout_ms), so it must be
    real elapsed speech. Credited per-chunk, a caller speaking 3s registered as
    600ms — permanently in the "short utterance" bucket that gets the MAXIMUM
    hangover, making adaptive mode a constant worst case."""
    vad = FastVoiceActivityDetector(config)
    monkeypatch.setattr(vad, "is_speech", lambda c, update_noise=True: True)

    samples = SAMPLE_RATE * chunk_ms // 1000
    chunk = np.zeros(samples, dtype=np.int16).tobytes()
    for _ in range(int(3000 / chunk_ms)):  # 3 seconds of speech
        vad.process_audio(chunk)

    assert vad.speech_ms == pytest.approx(3000, abs=chunk_ms)


def test_reset_clears_state(config):
    vad = FastVoiceActivityDetector(config)
    vad.is_speaking = True
    vad.silence_frames = 10
    vad.speech_ms = 240
    vad.speech_frames.append(silence_chunk())
    vad.reset()
    assert vad.is_speaking is False
    assert vad.silence_frames == 0
    assert len(vad.speech_frames) == 0
    assert vad.speech_ms == 0


def test_dynamic_silence_timeout_overrides_configured_value(config):
    vad = FastVoiceActivityDetector(config)
    vad.is_speaking = True
    vad.silence_frames = 0

    override_ms = 200  # much shorter than the configured 750ms
    fired_at = None
    for i in range(1, 200):
        _, end = vad.process_audio(silence_chunk(), silence_timeout_ms=override_ms)
        if end:
            fired_at = i
            break

    assert fired_at is not None
    assert abs(fired_at - override_ms / CHUNK_MS) <= 2
    # Well before the configured timeout would have fired.
    assert fired_at < config.silence_duration_ms / CHUNK_MS


def test_dynamic_timeout_none_keeps_configured_value(config):
    vad = FastVoiceActivityDetector(config)
    vad.is_speaking = True
    fired_at = None
    for i in range(1, 200):
        _, end = vad.process_audio(silence_chunk(), silence_timeout_ms=None)
        if end:
            fired_at = i
            break
    assert fired_at is not None
    assert abs(fired_at - config.silence_duration_ms / CHUNK_MS) <= 2


def test_speech_ms_accumulates_per_utterance(config, monkeypatch):
    vad = FastVoiceActivityDetector(config)
    # Force the speech classification (webrtcvad's judgment is not under test).
    monkeypatch.setattr(vad, "is_speech",
                        lambda chunk, update_noise=True: True)
    for _ in range(5):
        vad.process_audio(silence_chunk())
    assert vad.speech_ms == 5 * config.chunk_duration_ms

    # End the utterance, then a new one starts counting from zero.
    monkeypatch.setattr(vad, "is_speech",
                        lambda chunk, update_noise=True: False)
    for _ in range(200):
        _, end = vad.process_audio(silence_chunk())
        if end:
            break
    assert vad.is_speaking is False
    monkeypatch.setattr(vad, "is_speech",
                        lambda chunk, update_noise=True: True)
    vad.process_audio(silence_chunk())
    assert vad.speech_ms == config.chunk_duration_ms


def test_update_noise_false_does_not_move_noise_floor(config):
    vad = FastVoiceActivityDetector(config)
    chunk = (np.ones(SAMPLES_PER_CHUNK, dtype=np.int16) * 500).tobytes()

    for _ in range(30):
        vad.is_speech(chunk, update_noise=False)
    assert len(vad.noise_samples) == 0
    assert vad.noise_floor == 200  # untouched initial value

    # Sanity: the default path still adapts.
    for _ in range(30):
        vad.is_speech(chunk)
    assert len(vad.noise_samples) == 30
    assert vad.noise_floor != 200


def test_double_call_per_chunk_counts_noise_once(config):
    """The audio loop runs has_speech() (barge-in check) and process_audio()
    on the SAME chunk; only process_audio may feed the noise floor."""
    from audio_pipeline import LowLatencyAudioPipeline
    pipeline = LowLatencyAudioPipeline(config)
    state = pipeline.new_session_state()  # per-call VAD state
    vad = state.vad

    chunk = silence_chunk()
    for i in range(1, 11):
        pipeline.has_speech(state, chunk)   # barge-in style check
        vad.process_audio(chunk)            # VAD/STT path
        assert len(vad.noise_samples) == i  # one sample per chunk, not two


def test_has_speech_update_noise_true_feeds_floor(config):
    """Answering-machine detection runs ONLY has_speech() per chunk (no
    process_audio during the AMD window), so it passes update_noise=True —
    otherwise the adaptive floor would freeze at a stale value and steady
    line noise would read as continuous machine speech."""
    from audio_pipeline import LowLatencyAudioPipeline
    pipeline = LowLatencyAudioPipeline(config)
    state = pipeline.new_session_state()  # per-call VAD state
    chunk = silence_chunk()

    for _ in range(10):
        pipeline.has_speech(state, chunk)  # default: side-effect-free
    assert len(state.vad.noise_samples) == 0

    for i in range(1, 11):
        pipeline.has_speech(state, chunk, update_noise=True)  # AMD-style sole feed
        assert len(state.vad.noise_samples) == i
