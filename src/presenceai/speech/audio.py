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


def load_native(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode at the file's own sample rate and channel count, as float32 (channels, samples).

    Used for clipping checks (ADR 0013): resampling to 16 kHz smooths clipped peaks away, so
    clipping is detected on these original samples. Compressed formats (MP3, AAC) can decode to
    values slightly above 1.0 where the source was clipped; they are kept as they are.
    """
    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError(f"{Path(path).name}: no audio stream")
        stream = container.streams.audio[0]
        rate = int(stream.rate or stream.codec_context.sample_rate or SAMPLE_RATE)
        layout = stream.layout.name if stream.layout else "mono"
        # float output at the ORIGINAL rate and layout: no resampling, no downmix
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
        chunks: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray() for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray() for f in resampler.resample(None))
    if not chunks:
        return np.zeros((1, 0), dtype=np.float32), rate
    return np.concatenate(chunks, axis=1).astype(np.float32, copy=False), rate


class _LanguageDetector(Protocol):
    def detect_language(self, audio: np.ndarray, /, **options: Any) -> tuple[str, float, Any]: ...


def detect_language(
    model: _LanguageDetector, samples: np.ndarray
) -> tuple[str, float, list[tuple[str, float]]]:
    """Whisper language detection on 16 kHz samples (one 30 s window, no VAD filter).

    faster-whisper 1.2.1 `detect_language` returns (language, probability, all probabilities).
    """
    language, probability, all_probs = model.detect_language(
        samples, vad_filter=False, language_detection_segments=1
    )
    return str(language), float(probability), [(str(c), float(p)) for c, p in all_probs]


def transcribe(model: _WhisperModel, path: str | Path, **options: Any) -> tuple[list[Any], Any]:
    """Transcribe a file with a faster-whisper model; segments are fully decoded on return."""
    segments, info = model.transcribe(load_audio(path), **options)
    return list(segments), info
