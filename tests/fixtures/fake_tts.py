"""A fake phrase TTS for C1 tests (ADR 0014): no model, deterministic, known edges.

Each word is a vowel-like tone, 0.30 s at speed 1 (so 200 wpm of pure speech), with 20 ms
silent gaps between words. Words that start with s, f or h get a quiet noise onset, and words
that end with s or f get a quiet noise ending, 35-45 dB below the vowel: the hardest edges for
the trimmer. `internal_silence` inserts a silence of its own inside phrases containing a given
word, like a pause Kokoro might add. `calls` records every request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from presenceai.schemas.c1 import WORD

VOWEL_AMP = 0.3
FRICATIVE_DB = {"s": -35.0, "f": -42.0, "h": -45.0}  # below the vowel's level
FRICATIVE_S = 0.06
WORD_S = 0.30
WORD_GAP_S = 0.02
LEAD_S, TAIL_S = 0.12, 0.18  # silence Kokoro-like output has around a phrase


@dataclass
class FakeTTS:
    sample_rate: int = 24_000
    internal_silence: dict[str, float] = field(default_factory=dict)  # word -> seconds after it
    calls: list[tuple[str, str, float]] = field(default_factory=list)
    seed: int = 0

    def _noise(self, seconds: float, db: float, rng: np.random.Generator) -> np.ndarray:
        n = int(seconds * self.sample_rate)
        amp = VOWEL_AMP / np.sqrt(2) * 10 ** (db / 20)  # RMS relative to the vowel's RMS
        return (amp * rng.standard_normal(n)).astype(np.float32)

    def _tone(self, seconds: float) -> np.ndarray:
        n = int(seconds * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        ramp = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
        return (VOWEL_AMP * ramp * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    def synth(self, text: str, voice: str, speed: float) -> np.ndarray:
        self.calls.append((text, voice, speed))
        rng = np.random.default_rng([self.seed, len(self.calls)])
        sr = self.sample_rate
        parts = [np.zeros(int(LEAD_S * sr), np.float32)]
        words = [w.lower() for w in WORD.findall(text)]
        for i, w in enumerate(words):
            if w[0] in FRICATIVE_DB:
                parts.append(self._noise(FRICATIVE_S / speed, FRICATIVE_DB[w[0]], rng))
            parts.append(self._tone(WORD_S / speed - (FRICATIVE_S / speed if w[0] in "sfh" else 0)))
            if w[-1] in "sf":
                parts.append(self._noise(FRICATIVE_S / speed, FRICATIVE_DB[w[-1]], rng))
            if i < len(words) - 1:
                gap = self.internal_silence.get(w, WORD_GAP_S / speed)
                parts.append(np.zeros(int(gap * sr), np.float32))
        parts.append(np.zeros(int(TAIL_S * sr), np.float32))
        return np.concatenate(parts)


def edges(text: str, speed: float = 1.0, sample_rate: int = 24_000) -> tuple[float, float]:
    """Where the fake's sound really starts and ends inside a phrase (seconds)."""
    tts = FakeTTS(sample_rate=sample_rate)
    x = tts.synth(text, "v", speed)
    nz = np.flatnonzero(np.abs(x) > 0)
    return nz[0] / sample_rate, (nz[-1] + 1) / sample_rate
