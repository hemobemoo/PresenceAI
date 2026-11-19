"""B1 pilot set: the owner's own annotated recordings (ADR 0010, ADR 0016).

One clip has two annotation files, both pre-filled by the app and then corrected by the owner:

- `<clip>.yaml` -> `B1Annotation`: the verbatim transcript (Whisper's text to correct), the
  speech start and end, and the words ADR 0015 removed at the end (marked real or not);
- `<clip>.labels.txt`: the pauses the app detected, one `start<TAB>end<TAB>label` line each.
  The times are never edited. The labels (`sentence`, `clause`, `mid`; `?` = still to decide)
  are filled in by `run.py label` from the punctuation of the owner's corrected transcript.

Results hold numbers only, never transcript text (`B1ClipScore`). This module imports only from
`presenceai.schemas`.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from presenceai.schemas.speech import PausePosition, SpeechLanguage

ClipType = Literal["normal", "defense", "slow", "fast", "fillers", "noise"]
AnnotationStatus = Literal["prefilled", "corrected"]
# "prefill": labels file as prefill wrote it (all "?"); "transcript_punctuation": written by
# `run.py label` from the owner's corrected transcript (ADR 0016 amendment)
LabelsMethod = Literal["prefill", "transcript_punctuation"]
LABELS: dict[str, PausePosition] = {
    "sentence": PausePosition.SENTENCE_BOUNDARY,
    "clause": PausePosition.CLAUSE_BOUNDARY,
    "mid": PausePosition.MID_SENTENCE,
}
UNDECIDED = "?"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RemovedWord(_Frozen):
    """A word Whisper wrote after the last VAD speech, removed by ADR 0015."""

    text: str = Field(min_length=1)
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    real: bool | None = None  # owner: was this word really spoken? (None = not checked yet)


class Prefill(_Frozen):
    model: str
    config_hash: str
    commit: str | None = None


class B1Annotation(_Frozen):
    format_version: Literal[1] = 1
    clip: str = Field(min_length=1)  # file name in data/b1/audio/
    language: SpeechLanguage
    number: int = Field(ge=1, le=10)
    clip_type: ClipType
    code_switching: bool = False  # Arabic clips with English terms mid-sentence
    long_sentences: bool = False  # Arabic clips with long sentences (ADR 0011)
    status: AnnotationStatus = "prefilled"
    labels_method: LabelsMethod = "prefill"
    pause_edges_checked: bool = False  # pause times are the app's; nobody moved them by ear
    prefill: Prefill
    speech_start_s: float = Field(ge=0)  # first and last speech you hear (VAD pre-filled)
    speech_end_s: float = Field(gt=0)
    whisper_text: str  # what Whisper wrote (after ADR 0015); do not edit
    transcript: str  # correct this to exactly what was said, fillers and repeats included
    removed_at_end: tuple[RemovedWord, ...] = ()
    notes: str = ""

    @model_validator(mode="after")
    def _checked(self) -> Self:
        if self.speech_end_s <= self.speech_start_s:
            raise ValueError("speech_end_s must be after speech_start_s")
        if self.status == "corrected" and any(w.real is None for w in self.removed_at_end):
            raise ValueError("mark every removed word as real: true or real: false")
        return self


class LabelledPause(_Frozen):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    label: str  # a key of LABELS, or UNDECIDED

    @model_validator(mode="after")
    def _valid(self) -> Self:
        if self.end_s <= self.start_s:
            raise ValueError("a pause must end after it starts")
        if self.label not in LABELS and self.label != UNDECIDED:
            raise ValueError(f"unknown pause label {self.label!r}; use sentence, clause or mid")
        return self


class B1ClipScore(_Frozen):
    """Numbers only: no transcript text leaves data/b1/."""

    clip: str
    language: SpeechLanguage
    clip_type: ClipType
    code_switching: bool
    duration_s: float
    words_true: int
    words_true_no_hesitations: int
    words_measured: int
    wpm_true: float
    wpm_measured: float | None
    wpm_error_pct: float
    wer: float | None  # word error rate vs the corrected transcript
    wer_no_hesitations: float | None  # hesitations left out on both sides
    true_pauses: int
    detected_pauses: int
    pause_edges_checked: bool
    # Pause detection and duration need hand-checked pause edges; None otherwise (ADR 0016).
    tp: dict[str, int] | None  # per F1 threshold ("0.5", "1.0"), with the E2 exclusion zone
    fp: dict[str, int] | None
    fn: dict[str, int] | None
    duration_abs_errors_s: tuple[float, ...] | None
    positions: tuple[tuple[str, str], ...]  # (true, detected) for matched pauses
    detected_positions: tuple[str, ...]
    words_per_sentence_mark: float | None  # ADR 0011: positions unknown above the config value
    removed_at_end: int
    removed_real: int  # ADR 0015 check: real words the end margin removed
    clipping_share: float | None
    possible_clipping_share: float | None
    pileup_ratio_max: float | None  # ADR 0013: unclipped real speech should stay low
    snr_db: float | None
    detected_language: str | None
    language_probability: float | None
    language_mismatch: bool | None
