"""
Earcons
=======
In-memory audio cue synthesis (confirmation chime).

Pure functions returning mono int16 little-endian PCM, ready for
``SIPHandler.send_audio()``. Generated once at startup; never touches disk
or the TTS service.
"""

import numpy as np


def generate_chime(sample_rate: int = 16000, volume: float = 0.3) -> bytes:
    """Short two-note ascending ding (C6 then E6, a major third), ~0.30 s.

    Bell-like: sine fundamental plus a quiet second harmonic, 5 ms attack
    ramp (no click), exponential decay. The second note starts 90 ms in and
    overlaps the first's tail, so the whole cue stays under a third of a
    second — audible confirmation without eating into response latency.

    volume: peak amplitude as a fraction of int16 full scale (0.0-1.0].
    """
    duration_s = 0.30
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    sig = np.zeros(n)

    for freq, onset_s in ((1046.5, 0.0), (1318.5, 0.09)):  # C6, E6
        start = int(onset_s * sample_rate)
        tt = t[: n - start]
        env = np.minimum(tt / 0.005, 1.0) * np.exp(-tt * 14.0)
        note = (np.sin(2 * np.pi * freq * tt)
                + 0.35 * np.sin(2 * np.pi * 2 * freq * tt))
        sig[start:] += note * env

    peak = np.abs(sig).max()
    if peak > 0:
        sig /= peak
    return (sig * volume * 32767.0).astype(np.int16).tobytes()
