"""Unit tests: LowLatencyAudioPipeline wires ENDPOINT_MODE into the VAD.

fixed passes None (the VAD's configured timeout — today's behavior),
adaptive computes suggest_timeout_ms from the utterance length (audio-only:
no interim transcripts exist before the realtime commit), speculative always
uses the short threshold.

process_audio() operates on a per-call SessionAudioState, so each test
creates one via new_session_state() and drives it explicitly.
"""
import pytest

from audio_pipeline import LowLatencyAudioPipeline
from endpointing import suggest_timeout_ms

pytestmark = pytest.mark.unit

CHUNK = b"\x00" * 640


def _capture_vad_timeouts(state, monkeypatch):
    seen = []

    def fake_process(chunk, silence_timeout_ms=None):
        seen.append(silence_timeout_ms)
        return False, False

    monkeypatch.setattr(state.vad, "process_audio", fake_process)
    return seen


async def test_fixed_mode_passes_none(config_factory, monkeypatch):
    cfg = config_factory(endpoint_mode="fixed")
    pipeline = LowLatencyAudioPipeline(cfg)
    state = pipeline.new_session_state()
    seen = _capture_vad_timeouts(state, monkeypatch)
    await pipeline.process_audio(state, CHUNK)
    assert seen == [None]


async def test_adaptive_mode_scales_with_utterance_length(config_factory, monkeypatch):
    cfg = config_factory(endpoint_mode="adaptive")
    pipeline = LowLatencyAudioPipeline(cfg)
    state = pipeline.new_session_state()
    seen = _capture_vad_timeouts(state, monkeypatch)

    state.vad.speech_ms = 500      # winding up -> max hangover
    await pipeline.process_audio(state, CHUNK)
    state.vad.speech_ms = 10000    # long utterance -> back to base
    await pipeline.process_audio(state, CHUNK)

    assert seen == [
        suggest_timeout_ms(None, 500, cfg.silence_duration_ms,
                           cfg.endpoint_min_silence_ms, cfg.endpoint_max_silence_ms),
        suggest_timeout_ms(None, 10000, cfg.silence_duration_ms,
                           cfg.endpoint_min_silence_ms, cfg.endpoint_max_silence_ms),
    ]
    assert seen[0] == cfg.endpoint_max_silence_ms
    assert seen[1] == cfg.silence_duration_ms


async def test_speculative_mode_uses_short_threshold(config_factory, monkeypatch):
    cfg = config_factory(endpoint_mode="speculative")
    pipeline = LowLatencyAudioPipeline(cfg)
    state = pipeline.new_session_state()
    seen = _capture_vad_timeouts(state, monkeypatch)
    state.vad.speech_ms = 10000  # utterance length must not matter here
    await pipeline.process_audio(state, CHUNK)
    assert seen == [cfg.endpoint_min_silence_ms]


async def test_per_call_site_override_beats_configured_mode(config_factory,
                                                            monkeypatch):
    """api.py's choice collection has no hold/merge machinery, so it forces
    endpoint_mode="fixed" per call even when the config says speculative —
    the VAD must then get the configured fixed timeout (None), not the short
    speculative cutoff."""
    cfg = config_factory(endpoint_mode="speculative")
    pipeline = LowLatencyAudioPipeline(cfg)
    state = pipeline.new_session_state()
    seen = _capture_vad_timeouts(state, monkeypatch)
    await pipeline.process_audio(state, CHUNK, endpoint_mode="fixed")
    await pipeline.process_audio(state, CHUNK)  # no override -> configured mode
    assert seen == [None, cfg.endpoint_min_silence_ms]
