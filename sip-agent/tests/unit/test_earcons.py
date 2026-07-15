"""Unit tests for earcon (chime) synthesis."""
import numpy as np
import pytest

from earcons import generate_chime

pytestmark = pytest.mark.unit


def test_chime_is_int16_mono():
    pcm = generate_chime()
    assert len(pcm) % 2 == 0
    samples = np.frombuffer(pcm, dtype=np.int16)
    assert samples.size == len(pcm) // 2


def test_chime_duration():
    pcm = generate_chime(sample_rate=16000)
    duration = len(pcm) / 2 / 16000
    assert 0.25 <= duration <= 0.35


@pytest.mark.parametrize("volume", [0.3, 0.1])
def test_chime_peak_matches_volume(volume):
    samples = np.frombuffer(generate_chime(volume=volume), dtype=np.int16)
    peak = np.abs(samples.astype(np.int32)).max()
    assert abs(peak - volume * 32767) <= 2


def test_chime_no_click():
    samples = np.frombuffer(generate_chime(), dtype=np.int16)
    assert abs(int(samples[0])) < 500  # 5 ms attack ramp starts near zero


def test_chime_deterministic():
    assert generate_chime() == generate_chime()


def test_chime_respects_sample_rate():
    full = generate_chime(sample_rate=16000)
    half = generate_chime(sample_rate=8000)
    assert len(half) * 2 == len(full)
