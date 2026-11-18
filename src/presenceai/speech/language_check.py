"""Language check on the first speech (FINAL_04 §5, ADR 0016).

Whisper detects the language of the first `window_s` seconds counted from the first VAD speech
onset, so leading silence does not dilute the check. When another language than the chosen one
is detected with at least `mismatch_min_probability`, the report gets a notice in the report
language (the recording's chosen language). It is a notice only: the analysis still runs in the
chosen language, nothing stops, and no score changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from presenceai.lang import language_name, text
from presenceai.schemas.assessment import TimeSpan
from presenceai.schemas.speech import LanguageCheck, SpeechLanguage, SpeechRegion
from presenceai.settings import LanguageCheckSettings
from presenceai.speech import audio


def check(
    model: Any,
    samples: np.ndarray,
    regions: tuple[SpeechRegion, ...],
    language: SpeechLanguage,
    cfg: LanguageCheckSettings,
    sr: int = audio.SAMPLE_RATE,
) -> LanguageCheck | None:
    """None when there is no speech to check."""
    if not regions:
        return None
    start = regions[0].start_s
    end = min(start + cfg.window_s, samples.size / sr)
    window = samples[int(start * sr) : int(end * sr)]
    if window.size == 0:
        return None
    detected, probability, all_probs = audio.detect_language(model, window)
    chosen_p = next((p for code, p in all_probs if code == language), None)
    mismatch = detected != language and probability >= cfg.mismatch_min_probability
    notice = None
    if mismatch:
        notice = text(
            language,
            "language_check.notice",
            window_s=cfg.window_s,
            detected=language_name(language, detected),
            probability=probability,
            chosen=language_name(language, language),
        )
    return LanguageCheck(
        chosen=language,
        detected=detected,
        probability=min(1.0, max(0.0, probability)),
        chosen_probability=None if chosen_p is None else min(1.0, max(0.0, chosen_p)),
        window=TimeSpan(start_s=start, end_s=max(start, end)),
        mismatch=mismatch,
        notice=notice,
    )
