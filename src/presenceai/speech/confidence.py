"""Confidence bands for speech measures (FINAL_04 §6, ADR 0011; values in configs/default.yaml).

high:   mean word probability >= high.min_word_probability
        and speaking time >= high.min_speaking_time_s
medium: the same with the `medium` values
low:    otherwise

`None` for either input means "not used for this measure" (for example the syllable pace, which
does not depend on the transcript).
"""

from __future__ import annotations

from presenceai.schemas.assessment import Confidence
from presenceai.settings import ConfidenceBand, ConfidenceSettings


def _meets(band: ConfidenceBand, probability: float | None, speaking_time_s: float | None) -> bool:
    ok_p = probability is None or probability >= band.min_word_probability
    ok_t = speaking_time_s is None or speaking_time_s >= band.min_speaking_time_s
    return ok_p and ok_t


def band(
    cfg: ConfidenceSettings, *, probability: float | None, speaking_time_s: float | None
) -> Confidence:
    if _meets(cfg.high, probability, speaking_time_s):
        return Confidence.HIGH
    if _meets(cfg.medium, probability, speaking_time_s):
        return Confidence.MEDIUM
    return Confidence.LOW
