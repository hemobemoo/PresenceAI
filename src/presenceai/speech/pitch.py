"""Pitch track with pYIN (librosa, ISC), run once per recording (ADR 0012).

The pitch range is limited (`speech.pitch.fmin_hz`..`fmax_hz`), which keeps pYIN at about 5 s per
3 minutes of audio on the laptop after the first call (numba compiles once, ~20 s). The track is
used by the acoustic filler detector (slice 3) and reused for pitch variation (slice 4).
"""

from __future__ import annotations

import numpy as np

from presenceai.schemas.speech import PitchTrack
from presenceai.settings import PitchSettings


def track(samples: np.ndarray, sr: int, cfg: PitchSettings) -> PitchTrack:
    hop_s = cfg.hop_length / sr
    if samples.size < cfg.frame_length:
        return PitchTrack(
            hop_s=hop_s, fmin_hz=cfg.fmin_hz, fmax_hz=cfg.fmax_hz, f0_hz=(), voiced=()
        )
    import librosa  # heavy import (numba), only when a recording is analysed

    f0, voiced, _ = librosa.pyin(
        samples.astype(np.float32, copy=False),
        fmin=cfg.fmin_hz,
        fmax=cfg.fmax_hz,
        sr=sr,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
    )
    voiced = np.asarray(voiced, dtype=bool) & np.isfinite(f0)
    f0 = np.where(voiced, f0, 0.0)
    return PitchTrack(
        hop_s=hop_s,
        fmin_hz=cfg.fmin_hz,
        fmax_hz=cfg.fmax_hz,
        f0_hz=tuple(float(v) for v in f0),
        voiced=tuple(bool(v) for v in voiced),
    )
