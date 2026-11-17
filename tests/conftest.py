"""Shared pytest setup. Tests never need models, camera, microphone or network.

Fakes for faster-whisper live here so every speech test uses the same shapes: `FakeWord`,
`FakeSegment` and `FakeInfo` mirror the attributes PresenceAI reads from faster-whisper's
`Word`, `Segment` and `TranscriptionInfo`; `FakeWhisperModel.transcribe` records the array and
options it was given and returns the canned segments.
"""

from __future__ import annotations

import wave
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@dataclass
class FakeWord:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    words: list[FakeWord] | None
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.01


@dataclass
class FakeInfo:
    language: str
    language_probability: float
    duration: float


@dataclass
class FakeWhisperModel:
    segments: Sequence[FakeSegment]
    info: FakeInfo
    calls: list[dict[str, Any]] = field(default_factory=list)

    def transcribe(self, audio: np.ndarray, /, **options: Any) -> tuple[Iterator[Any], Any]:
        self.calls.append({"audio": audio, "options": options})
        return iter(self.segments), self.info


def two_segments() -> list[FakeSegment]:
    """'Hello everyone. Today I will talk.' with word timestamps, as faster-whisper gives them
    (each word carries its leading space)."""
    return [
        FakeSegment(
            0.0,
            1.2,
            " Hello everyone.",
            [FakeWord(" Hello", 0.0, 0.4, 0.98), FakeWord(" everyone.", 0.45, 1.2, 0.90)],
        ),
        FakeSegment(
            2.0,
            3.6,
            " Today I will talk.",
            [
                FakeWord(" Today", 2.0, 2.4, 0.95),
                FakeWord(" I", 2.45, 2.55, 0.99),
                FakeWord(" will", 2.6, 2.9, 0.80),
                FakeWord(" talk.", 2.95, 3.6, 0.70),
            ],
            avg_logprob=-0.3,
            no_speech_prob=0.03,
        ),
    ]


@pytest.fixture
def fake_whisper() -> Callable[..., FakeWhisperModel]:
    def make(
        segments: Sequence[FakeSegment] | None = None,
        language: str = "en",
        duration: float = 4.0,
    ) -> FakeWhisperModel:
        segs = two_segments() if segments is None else list(segments)
        return FakeWhisperModel(segs, FakeInfo(language, 1.0, duration))

    return make


@pytest.fixture
def whisper_parts() -> Any:
    """The fake classes, for tests that build their own segments."""
    from types import SimpleNamespace

    return SimpleNamespace(Word=FakeWord, Segment=FakeSegment, Info=FakeInfo)


@pytest.fixture
def speech_cfg() -> Any:
    """`speech:` settings exactly as in configs/default.yaml (ADR 0011); no .env is read."""
    from presenceai.settings import Settings

    return Settings(_env_file=None).speech  # pyright: ignore[reportCallIssue]


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """A 0.5 s silent 16 kHz mono WAV (the fake model ignores the samples)."""
    path = tmp_path / "talk.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(np.zeros(8_000, dtype="<i2").tobytes())
    return path
