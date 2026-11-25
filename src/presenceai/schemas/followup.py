"""Follow-up answers in S2 (ADR 0022): per-answer measures and the weaker-answer summary.

Each answer is recorded on its own, starting automatically when its question ends, so an
answer's times are seconds from the end of its question.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from presenceai.schemas.assessment import Measurement
from presenceai.schemas.speech import Transcript

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
    # Reading allowance (owner, 2026-11-25): question words / fu_reading_wps. The latency
    # conditions compare the thinking time, latency minus this, with the 2 s and 4 s limits.
    reading_allowance_s: float = Field(default=0.0, ge=0)
    # False for a question from the generic bank: its similarity is reported, not scored, until
    # relevance is recalibrated on generic questions (owner, 2026-11-25). A measured non-answer
    # (no_answer) is decided from the audio and always scores 0.
    relevance_scored: bool = True


class FollowupSummary(_Frozen):
    """The weaker answer decides (owner, 2026-11-23): lowest relevance, longest latency."""

    answers: tuple[AnswerMeasures, ...] = Field(min_length=1)
    relevance_min: float | None = None
    relevance_min_index: int | None = None
    latency_max_s: float | None = None  # the longest thinking time (latency - reading allowance)
    latency_max_index: int | None = None
    limitation_acknowledged: bool = False
    limitation_index: int | None = None


class FollowupAnswerInput(_Frozen):
    """One follow-up in the `presenceai assess --followups` file. Until P7 generates the
    questions, they are given as input (ADR 0022 item 9)."""

    question: str = Field(min_length=1)
    file: str = Field(min_length=1)  # the answer recording; relative to the JSON file's folder
    window_s: float = Field(gt=0)  # the answer window the recording was given


class FollowupInput(_Frozen):
    answers: tuple[FollowupAnswerInput, ...] = Field(min_length=1)


class FollowupReport(_Frozen):
    """The follow-up part of an S2 assessment: per-answer measures, the weaker-answer summary,
    and the rubric measures with evidence that names the answer it comes from."""

    questions: tuple[str, ...] = Field(min_length=1)
    sources: tuple[str, ...]  # answer recording file names, no folders
    transcripts: tuple[Transcript | None, ...]  # None: the recording could not be read
    summary: FollowupSummary
    measurements: dict[str, Measurement] = Field(default_factory=dict)
    not_measurable: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_entry_per_answer(self) -> Self:
        n = len(self.summary.answers)
        if not (len(self.questions) == len(self.sources) == len(self.transcripts) == n):
            raise ValueError("questions, sources, transcripts and answers must match in number")
        if [a.index for a in self.summary.answers] != list(range(n)):
            raise ValueError("answers must be indexed 0..n-1 in order")
        return self
