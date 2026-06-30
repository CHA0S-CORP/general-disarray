"""In-process mock of the Speaches STT+TTS server (OpenAI-compatible).

Implements just enough of the surface that `audio_pipeline`'s WhisperAPIClient
and SpeachesTTSClient exercise:

  GET  /health                        -> 200
  GET  /v1/models                     -> empty list (STT then POSTs to "download")
  POST /v1/models/{model_id}          -> 200 (pretend download succeeded)
  POST /v1/audio/speech               -> 200, a real little-endian int16 WAV
  POST /v1/audio/transcriptions       -> {"text": MOCK_TRANSCRIPT}

The transcription text is fixed so component tests are deterministic.
"""
import io
import wave

import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response

# Deterministic transcript returned for every STT request. Tests assert on this.
MOCK_TRANSCRIPT = "the eagle has landed"


def make_wav(duration_s: float = 0.4, rate: int = 24000, freq: float = 220.0) -> bytes:
    """Return a mono 16-bit PCM WAV (RIFF) of a quiet sine tone.

    Kokoro's native rate is 24000 Hz; the TTS client reads the rate from the WAV
    header and resamples to 16000, so any valid rate works here.
    """
    n = int(duration_s * rate)
    t = np.arange(n) / rate
    samples = (0.3 * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        # Empty -> WhisperAPIClient falls through to the POST "download" path.
        return {"data": []}

    @app.post("/v1/models/{model_id:path}")
    async def download_model(model_id: str):
        return {"status": "ok", "id": model_id}

    @app.post("/v1/audio/speech")
    async def speech():
        return Response(content=make_wav(), media_type="audio/wav")

    @app.post("/v1/audio/transcriptions")
    async def transcribe():
        # The uploaded audio is intentionally ignored; transcript is fixed.
        return {"text": MOCK_TRANSCRIPT}

    return app
