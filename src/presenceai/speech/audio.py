"""The one way audio reaches faster-whisper (ADR 0006).

faster-whisper 1.2.1 decodes file paths itself with PyAV and passes `metadata_errors`, which
PyAV 19 no longer accepts. So audio is always decoded here and handed over as an array:

- `load_audio(path)` -> float32 array, 16 kHz, mono, values in [-1, 1];
- `transcribe(model, path, **options)` is the only place in the code base that calls a
  faster-whisper model's `.transcribe`, always with `load_audio(path)`.

`tests/unit/test_audio_loader.py` fails CI if any other module calls `.transcribe(` or uses
faster-whisper's own `decode_audio`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import av
import numpy as np

SAMPLE_RATE = 16_000  # what Whisper models expect


class _WhisperModel(Protocol):
    def transcribe(self, audio: np.ndarray, /, **options: Any) -> tuple[Iterable[Any], Any]: ...


def load_audio(path: str | Path) -> np.ndarray:
    """Decode any file PyAV can read into float32 mono samples at 16 kHz."""
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError(f"{Path(path).name}: no audio stream")
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))  # flush
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def transcribe(model: _WhisperModel, path: str | Path, **options: Any) -> tuple[list[Any], Any]:
    """Transcribe a file with a faster-whisper model; segments are fully decoded on return."""
    segments, info = model.transcribe(load_audio(path), **options)
    return list(segments), info
