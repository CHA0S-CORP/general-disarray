"""Regression tests for audio_pipeline fixes: VAD frame alignment, utterance
buffering (pauses + pre-roll), STT/TTS re-probe, STT timeout/4xx handling,
TTS resample clipping + compressed decode, and realtime->batch fallback."""
import io
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

import audio_pipeline
from audio_pipeline import (
    FastVoiceActivityDetector,
    LowLatencyAudioPipeline,
    SpeachesTTSClient,
    WhisperAPIClient,
)

pytestmark = pytest.mark.unit

RATE = 16000
FRAME_BYTES = int(RATE * 0.03) * 2  # 960


def _tone(n_bytes: int, amp: int = 3000) -> bytes:
    n = n_bytes // 2
    t = np.arange(n) / RATE
    return (np.sin(2 * np.pi * 300 * t) * amp).astype(np.int16).tobytes()


def _chunk(ms: int, value: int = 0) -> bytes:
    return np.full(RATE * ms // 1000, value, dtype=np.int16).tobytes()


class _FakeWebrtcVad:
    """Records the frames webrtcvad would see; classifies by a flag."""

    def __init__(self, speech: bool = True):
        self.frames = []
        self.speech = speech

    def is_speech(self, frame, rate):
        self.frames.append(len(frame))
        return self.speech


# --- 1a: partial-frame carry -------------------------------------------------

def test_vad_carries_partial_frames_across_chunks(config):
    vad = FastVoiceActivityDetector(config)
    fake = _FakeWebrtcVad(speech=False)  # non-speech: every frame is scored
    vad.vad = fake

    # The recorder grows in 4096-byte steps: receive_audio returns 3200 + 896.
    vad.is_speech(_tone(3200))
    vad.is_speech(_tone(896))
    assert all(n == FRAME_BYTES for n in fake.frames)
    assert len(fake.frames) == 4096 // FRAME_BYTES
    assert len(vad._frame_remainder) == 4096 % FRAME_BYTES

    # An 896-byte chunk (< one 30ms frame) used to be unconditionally
    # non-speech; with the carried remainder it completes a frame.
    fake.speech = True
    assert vad.is_speech(_tone(896)) is True


def test_vad_side_effect_free_peek_does_not_consume_remainder(config):
    vad = FastVoiceActivityDetector(config)
    vad.vad = _FakeWebrtcVad(speech=True)
    vad.is_speech(_tone(1000))
    before = vad._frame_remainder
    vad.is_speech(_tone(500), update_noise=False)  # has_speech()-style peek
    assert vad._frame_remainder == before


def test_vad_aggressiveness_comes_from_config(config, monkeypatch):
    made = []
    monkeypatch.setattr(audio_pipeline, "VAD_AVAILABLE", True)
    monkeypatch.setattr(audio_pipeline, "webrtcvad",
                        SimpleNamespace(Vad=lambda mode: made.append(mode) or object()),
                        raising=False)
    config.vad_aggressiveness = 1
    FastVoiceActivityDetector(config)
    config.vad_aggressiveness = 9  # out of range -> clamped
    FastVoiceActivityDetector(config)
    assert made == [1, 3]


# --- 1b/1c: utterance buffering ------------------------------------------------

def _pipeline_with_batch(config):
    p = LowLatencyAudioPipeline(config)
    calls = []

    async def transcribe(audio):
        calls.append(audio)
        return "hello"

    p._stt_batch_client = SimpleNamespace(available=True, transcribe=transcribe)
    return p, calls


async def test_inter_word_pauses_reach_stt(config, monkeypatch):
    p, calls = _pipeline_with_batch(config)
    state = p.new_session_state()
    pattern = iter([True, False, True, False, False, True])  # words + pauses
    monkeypatch.setattr(state.vad, "is_speech",
                        lambda c, update_noise=True: next(pattern, False))

    chunks = [_chunk(100, v) for v in (1, 2, 3, 4, 5, 6)]
    for c in chunks:
        assert await p.process_audio(state, c) is None
    result = None
    for _ in range(50):
        result = await p.process_audio(state, _chunk(20))
        if result:
            break
    assert result == "hello"
    audio = calls[0]
    # Every utterance chunk, pauses included, in order.
    assert audio.startswith(b"".join(chunks))


async def test_preroll_is_prepended_on_speech_onset(config, monkeypatch):
    p, _ = _pipeline_with_batch(config)
    state = p.new_session_state()
    speech = {"on": False}
    monkeypatch.setattr(state.vad, "is_speech",
                        lambda c, update_noise=True: speech["on"])

    for v in range(1, 21):  # 20 x 20ms of non-speech, distinct content
        await p.process_audio(state, _chunk(20, v))
    pad_bytes = int(RATE * config.speech_pad_ms / 1000) * 2
    assert len(state.preroll) == pad_bytes
    assert len(state.buffer) == 0

    speech["on"] = True
    onset = _chunk(20, 99)
    await p.process_audio(state, onset)
    # Buffer = the most recent ~speech_pad_ms of pre-roll, then the onset.
    expected_preroll = b"".join(_chunk(20, v) for v in range(11, 21))
    assert bytes(state.buffer) == expected_preroll + onset
    assert len(state.preroll) == 0


async def test_lone_click_padded_with_preroll_and_silence_is_not_sent(config, monkeypatch):
    p, calls = _pipeline_with_batch(config)
    state = p.new_session_state()
    seq = iter([False] * 15 + [True])  # pre-roll, then one 20ms "click"
    monkeypatch.setattr(state.vad, "is_speech",
                        lambda c, update_noise=True: next(seq, False))
    for _ in range(200):
        result = await p.process_audio(state, _chunk(20))
        if not state.vad.is_speaking and len(state.buffer) == 0 and _ > 20:
            break
    assert result in (None, "")
    assert calls == []


# --- 11 (pipeline side): realtime -> batch fallback ----------------------------

async def test_realtime_empty_commit_falls_back_to_batch(config):
    p, calls = _pipeline_with_batch(config)
    state = p.new_session_state()

    async def commit_and_wait(timeout):
        return ""  # connection down / timed out

    state.realtime = SimpleNamespace(commit_and_wait=commit_and_wait)
    state.buffer.extend(_chunk(400, 7))
    assert await p._transcribe_buffer(state) == "hello"
    assert calls == [_chunk(400, 7)]


# --- 3/4: STT re-probe, timeout, 4xx -------------------------------------------

def _wav_bytes(n_samples=2400, rate=24000) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * n_samples)
    return buf.getvalue()


class _Speaches:
    def __init__(self, health=200, stt_status=200, tts_status=200):
        self.health = health
        self.stt_status = stt_status
        self.tts_status = tts_status
        self.paths = []
        self.timeouts = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        self.timeouts.setdefault(path, []).append(request.extensions.get("timeout"))
        if path == "/health":
            return httpx.Response(self.health)
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        if path == "/v1/audio/transcriptions":
            if self.stt_status != 200:
                return httpx.Response(self.stt_status, json={"detail": "x"})
            return httpx.Response(200, json={"text": "recovered"})
        if path == "/v1/audio/speech":
            if self.tts_status != 200:
                return httpx.Response(self.tts_status, json={"detail": "x"})
            return httpx.Response(200, content=_wav_bytes())
        return httpx.Response(404)


def _stt(config_factory, server, **env):
    cfg = config_factory(WHISPER_MODEL="m", API_RETRY_BASE_DELAY_S="0.001",
                         API_RETRY_MAX_DELAY_S="0.002", **env)
    c = WhisperAPIClient(cfg)
    c.client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    return c


async def test_stt_reprobes_after_interval_and_recovers(config_factory):
    server = _Speaches(health=503)
    c = _stt(config_factory, server, SPEECH_REPROBE_INTERVAL_S="30")
    await c._probe(warm_up=False)
    assert c.available is False

    server.health = 200
    # Within the interval: no probe, no request.
    server.paths.clear()
    assert await c.transcribe(_chunk(300)) == ""
    assert server.paths == []

    c._last_probe = time.monotonic() - 31
    assert await c.transcribe(_chunk(300)) == "recovered"
    assert c.available is True
    assert "/health" in server.paths


async def test_stt_uses_per_request_timeout(config_factory):
    server = _Speaches()
    c = _stt(config_factory, server, STT_TIMEOUT_S="4.5")
    c.available = True
    assert await c.transcribe(_chunk(300)) == "recovered"
    t = server.timeouts["/v1/audio/transcriptions"][0]
    assert t["read"] == 4.5


async def test_stt_4xx_is_not_retried(config_factory):
    server = _Speaches(stt_status=400)
    c = _stt(config_factory, server, API_RETRY_ATTEMPTS="3")
    c.available = True
    assert await c.transcribe(_chunk(300)) == ""
    assert server.paths.count("/v1/audio/transcriptions") == 1


async def test_stt_5xx_is_still_retried(config_factory):
    server = _Speaches(stt_status=503)
    c = _stt(config_factory, server, API_RETRY_ATTEMPTS="3")
    c.available = True
    assert await c.transcribe(_chunk(300)) == ""
    assert server.paths.count("/v1/audio/transcriptions") == 3


# --- 3: TTS re-probe + warm-up timeout -----------------------------------------

def _tts(config_factory, server, **env):
    c = SpeachesTTSClient(config_factory(**env))
    c.client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    c.cache_enabled = False
    return c


async def test_tts_reprobes_and_recovers(config_factory):
    server = _Speaches(tts_status=503)  # Speaches up, model still loading
    c = _tts(config_factory, server, SPEECH_REPROBE_INTERVAL_S="30")
    assert await c._probe() is False
    assert c.available is False

    server.tts_status = 200
    assert await c.synthesize("hello there") == b""  # rate-limited
    c._last_probe = time.monotonic() - 31
    audio = await c.synthesize("hello there")
    assert audio and c.available is True


async def test_tts_warmup_probe_timeout_is_generous(config_factory):
    server = _Speaches()
    c = _tts(config_factory, server, API_TIMEOUT_S="30")
    assert await c._probe() is True
    t = server.timeouts["/v1/audio/speech"][0]
    assert t["read"] >= 60


# --- 9: resample clipping + compressed formats ---------------------------------

def test_resample_clips_instead_of_wrapping(config):
    import scipy.signal
    c = SpeachesTTSClient(config)
    sq = np.where((np.arange(4800) // 120) % 2 == 0, 32767, -32768).astype(np.int16)
    out = np.frombuffer(c._resample(sq.tobytes(), 24000, 16000), dtype=np.int16)
    ref = scipy.signal.resample_poly(sq.astype(np.float64), 2, 3)
    assert ref.max() > 32767  # overshoot exists, so wraparound was possible
    assert np.array_equal(out, np.clip(np.round(ref), -32768, 32767).astype(np.int16))


async def test_compressed_tts_format_is_decoded_to_pcm(config_factory):
    import soundfile as sf
    c = SpeachesTTSClient(config_factory(TTS_RESPONSE_FORMAT="flac"))
    buf = io.BytesIO()
    sf.write(buf, np.zeros(2400, dtype=np.int16), 24000, format="FLAC")
    pcm = await c._to_call_pcm(buf.getvalue())
    assert len(pcm) == 1600 * 2  # 100ms at 16 kHz, int16


async def test_undecodable_compressed_tts_returns_empty(config_factory):
    c = SpeachesTTSClient(config_factory(TTS_RESPONSE_FORMAT="mp3"))
    assert await c._to_call_pcm(b"not audio at all") == b""
