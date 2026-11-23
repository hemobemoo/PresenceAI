"""Follow-up answers in S2 (ADR 0022): per-answer measures and the weaker-answer summary.

Each answer is recorded on its own, starting automatically when its question ends, so an
answer's times are seconds from the end of its question.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AnswerMeasures(_Frozen):
    index: int = Field(ge=0)  # 0 for the first follow-up question
    latency_s: float | None = Field(default=None, ge=0)  # None: no speech onset found
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
