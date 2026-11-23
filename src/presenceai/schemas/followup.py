"""Follow-up answers in S2 (ADR 0022): per-answer measures and the weaker-answer summary.

Each answer is recorded on its own, starting automatically when its question ends, so an
answer's times are seconds from the end of its question.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# How an answer recording ended (owner, 2026-11-23, ADR 0022):
# - answered: speech found and transcribed;
# - no_answer: the recording passed its checks, the full answer window elapsed and the voice
#   detector found no speech: a measured non-answer (relevance 0.0, latency = the window);
# - speech_untranscribed: speech found but the transcript is empty: relevance missing;
# - incomplete: no speech found, but the recording stopped before the window ended: missing;
# - recording_failed: the recording failed its checks: relevance and latency missing.
AnswerStatus = Literal[
    "answered", "no_answer", "speech_untranscribed", "incomplete", "recording_failed"
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AnswerMeasures(_Frozen):
    index: int = Field(ge=0)  # 0 for the first follow-up question
    status: AnswerStatus = "answered"
    window_s: float | None = Field(default=None, gt=0)  # the answer window (time evidence)
    # None: missing. For a no_answer it is the whole window: longer than any latency limit.
    latency_s: float | None = Field(default=None, ge=0)
    relevance: float | None = Field(default=None, ge=-1, le=1)  # None: not computed
    limitation: str | None = None  # the first limitation cue found, if any


class FollowupSummary(_Frozen):
    """The weaker answer decides (owner, 2026-11-23): lowest relevance, longest latency."""

    answers: tuple[AnswerMeasures, ...] = Field(min_length=1)
    relevance_min: float | None = None
    relevance_min_index: int | None = None
    latency_max_s: float | None = None
    latency_max_index: int | None = None
    limitation_acknowledged: bool = False
    limitation_index: int | None = None
