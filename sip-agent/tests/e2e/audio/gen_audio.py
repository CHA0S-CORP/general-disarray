"""Generate silence-padded WAV question fixtures via the stack's Speaches TTS.

The e2e conftest calls `ensure_question_wav()` to lazily create fixtures against
the live stack, so no binary WAVs need to be committed. You can also run this
standalone once the stack is up:

    python tests/e2e/audio/gen_audio.py --speaches http://localhost:8001

Each fixture gets ~3s of leading silence (so the spoken question lands after the
agent's greeting, inside the listening window) and ~1s of trailing silence (so
the agent's VAD cleanly detects end-of-turn). Output is 16-bit mono PCM WAV at
16 kHz, which pjsua resamples to the negotiated codec.
"""
import argparse
import io
import wave
from pathlib import Path

import httpx
import numpy as np

try:
    import scipy.signal  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

TARGET_RATE = 16000

# (filename, spoken text). Anchored on deterministic tools so assertions are
# robust to LLM phrasing variance.
FIXTURES = {
    "simon_says_eagle.wav": "Simon says, the eagle has landed.",
    "calc_17x3.wav": "What is seventeen times three?",
}


def _resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return samples
    if _HAVE_SCIPY:
        n = int(len(samples) * to_rate / from_rate)
        return scipy.signal.resample(samples.astype(np.float64), n).astype(np.int16)
    idx = np.linspace(0, len(samples) - 1, int(len(samples) * to_rate / from_rate))
    return np.interp(idx, np.arange(len(samples)), samples.astype(np.float64)).astype(np.int16)


def _tts_pcm(text: str, speaches_url: str) -> np.ndarray:
    """Synthesize `text` via Speaches and return int16 PCM resampled to 16 kHz."""
    resp = httpx.post(
        f"{speaches_url.rstrip('/')}/v1/audio/speech",
        json={
            "model": "speaches-ai/Kokoro-82M-v1.0-ONNX",
            "voice": "af_heart",
            "input": text,
            "response_format": "wav",
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    with wave.open(io.BytesIO(resp.content), "rb") as w:
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16)
    return _resample(samples, rate, TARGET_RATE)


def _write_padded(path: Path, speech: np.ndarray, lead_s: float = 3.0, trail_s: float = 90.0):
    # trail_s must exceed the longest test call duration: pjsua --auto-play
    # LOOPS the file, so a short WAV re-asks the question mid-reply, which the
    # agent (correctly) treats as a barge-in and truncates its answer.
    lead = np.zeros(int(lead_s * TARGET_RATE), dtype=np.int16)
    trail = np.zeros(int(trail_s * TARGET_RATE), dtype=np.int16)
    full = np.concatenate([lead, speech, trail])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(full.tobytes())


def ensure_question_wav(filename: str, speaches_url: str, audio_dir: Path) -> Path:
    """Create `audio_dir/filename` from FIXTURES[filename] if absent; return path."""
    out = audio_dir / filename
    if out.exists():
        return out
    if filename not in FIXTURES:
        raise KeyError(f"No fixture text registered for {filename}")
    audio_dir.mkdir(parents=True, exist_ok=True)
    speech = _tts_pcm(FIXTURES[filename], speaches_url)
    _write_padded(out, speech)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaches", default="http://localhost:8001")
    args = ap.parse_args()
    audio_dir = Path(__file__).parent
    for name in FIXTURES:
        path = ensure_question_wav(name, args.speaches, audio_dir)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
