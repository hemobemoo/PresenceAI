"""C1 synthetic speech and E2 scoring types (ADR 0014).

A C1 clip is built from separately generated phrases joined by silences of exact length, so its
ground truth (pause times and positions, word count, speaking time) comes from the construction
and is then checked on the finished audio. Imports only from `presenceai.schemas`.
"""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from presenceai.schemas.speech import PausePosition

Boundary = Literal["sentence_boundary", "clause_boundary", "mid_sentence", "end"]
Part = Literal["dev", "test"]
PauseSource = Literal["inserted", "kokoro"]
WORD = re.compile(r"[A-Za-z]+")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def count_words(text: str) -> int:
    """Words as C1 counts them: runs of letters (scripts have no digits, apostrophes or hyphens)."""
    return len(WORD.findall(text))


class Phrase(_Frozen):
    """One TTS call. `boundary_after` says what follows it in the script."""

    text: str = Field(min_length=1)
    boundary_after: Boundary

    @property
    def words(self) -> int:
        return count_words(self.text)


class Script(_Frozen):
    id: str = Field(pattern=r"^s\d{2}$")
    voice: str
    title: str
    phrases: tuple[Phrase, ...] = Field(min_length=1)
    source_text: str = ""  # as written, with the '|' marks

    @model_validator(mode="after")
    def _last_is_end(self) -> Self:
        if self.phrases[-1].boundary_after != "end":
            raise ValueError("the last phrase must end the script")
        if any(p.boundary_after == "end" for p in self.phrases[:-1]):
            raise ValueError("only the last phrase may end the script")
        return self

    @property
    def text(self) -> str:
        return " ".join(p.text for p in self.phrases)

    @property
    def words(self) -> int:
        return sum(p.words for p in self.phrases)


class GapPlan(_Frozen):
    """What goes after phrase `after_phrase`: a silence, optionally a filler and a short gap."""

    after_phrase: int = Field(ge=0)
    boundary: Literal["sentence_boundary", "clause_boundary", "mid_sentence"]
    seconds: float = Field(gt=0)
    filler: str | None = None
    after_filler_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _filler_gap(self) -> Self:
        if (self.filler is None) != (self.after_filler_s is None):
            raise ValueError("a filler needs after_filler_s, and only a filler has it")
        return self


class ClipPlan(_Frozen):
    clip_id: str
    script_id: str
    voice: str
    render: int = Field(ge=0)
    target_wpm: float = Field(gt=0)
    gaps: tuple[GapPlan, ...]


class TruePause(_Frozen):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    position: PausePosition
    source: PauseSource

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class InternalSilence(_Frozen):
    """A silence Kokoro made inside one phrase (ADR 0014 §1), in clip time."""

    phrase: int = Field(ge=0)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class ClipTruth(_Frozen):
    """Ground truth of one clip, measured on the finished clean audio (before the noise floor)."""

    clip_id: str
    script_id: str
    voice: str
    render: int = Field(ge=0)
    sample_rate: int = Field(gt=0)
    duration_s: float = Field(gt=0)
    words: int = Field(gt=0)  # script words + fillers: every spoken word
    fillers: int = Field(ge=0)
    speech_onset_s: float = Field(ge=0)
    speech_offset_s: float = Field(gt=0)
    speech_spans: tuple[tuple[float, float], ...]  # every phrase / filler, trimmed
    wpm: float = Field(gt=0)  # words / (offset - onset), as rate.wpm (ADR 0011)
    articulation_wpm: float = Field(gt=0)  # words / time inside speech spans
    target_wpm: float = Field(gt=0)
    speed: float = Field(gt=0)
    attempts: int = Field(ge=1)
    pauses: tuple[TruePause, ...]
    internal_silences: tuple[InternalSilence, ...]
    text: str


class PauseMatch(_Frozen):
    true_index: int = Field(ge=0)
    detected_index: int = Field(ge=0)
    onset_error_s: float
    duration_error_s: float


class ThresholdScore(_Frozen):
    """Pause detection at one threshold. `excluded` true pauses lie within the exclusion zone."""

    threshold_s: float
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    excluded: int = Field(ge=0)
    tp_all: int = Field(ge=0)  # the same without the exclusion zone
    fp_all: int = Field(ge=0)
    fn_all: int = Field(ge=0)


def f1(tp: int, fp: int, fn: int) -> float | None:
    if tp + fp + fn == 0:
        return None
    return 2 * tp / (2 * tp + fp + fn)


class E2ClipScore(_Frozen):
    clip_id: str
    script_id: str
    voice: str
    part: Part
    condition: str
    has_fillers: bool
    rate_band: str | None
    wpm_true: float
    wpm_measured: float | None
    wpm_error_pct: float | None
    true_pauses: int = Field(ge=0)
    detected_pauses: int = Field(ge=0)
    thresholds: tuple[ThresholdScore, ...]
    duration_abs_errors_s: tuple[float, ...]
    positions: tuple[tuple[str, str], ...]  # (true, detected label) per matched pause
    true_positions: tuple[str, ...]
    detected_positions: tuple[str, ...]
    kokoro_pauses: int = Field(ge=0)


class E2Summary(_Frozen):
    """One part and condition. `passes` is None on the development part (tuning only)."""

    part: Part
    condition: str
    clips: int = Field(ge=0)
    wpm_mape_pct: float | None
    wpm_mape_fillers_pct: float | None
    wpm_mape_no_fillers_pct: float | None
    f1: dict[float, float | None]  # target F1, exclusion zone applied
    f1_all: dict[float, float | None]  # without the exclusion zone
    excluded: dict[float, int]  # true pauses in each threshold's exclusion zone
    true_pauses: int = Field(ge=0)
    detected_pauses: int = Field(ge=0)
    matched: int = Field(ge=0)
    duration_mae_s: float | None
    true_mid_share: float | None
    detected_mid_share: float | None
    position_error_rate: float | None
    position_unknown_share: float | None
    kokoro_pauses: int = Field(ge=0)
    passes: dict[str, bool | None]
    passed: bool | None
