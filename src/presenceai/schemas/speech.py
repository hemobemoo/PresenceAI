"""Speech types: the transcript with word timestamps, and the `analyze` report.

Every `Word` records the character span it occupies in `Transcript.text`, so later stages
(pauses, fillers, repetitions, rubric) can quote the transcript exactly as evidence
(`Transcript.span` -> `TranscriptSpan`, AGENTS.md rule 4). Defined in
docs/plan/FINAL_04_MULTIMODAL_MEASUREMENT.md §2 (ASR confidence row) and FINAL_12 P1.
This module imports only from `presenceai.schemas`.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from presenceai.schemas.assessment import Confidence, Measurement, TranscriptSpan

SpeechLanguage = Literal["en", "ar"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Word(_Frozen):
    text: str = Field(min_length=1)
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    probability: float = Field(ge=0, le=1)
    start_char: int = Field(ge=0)  # offsets into Transcript.text
    end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.end_s < self.start_s:
            raise ValueError("end_s must be >= start_s")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("end_char - start_char must equal len(text)")
        return self


class Segment(_Frozen):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    words: tuple[Word, ...]
    avg_logprob: float
    no_speech_prob: float = Field(ge=0, le=1)


class Transcript(_Frozen):
    language: SpeechLanguage  # the language the user chose and Whisper was told to use
    duration_s: float = Field(ge=0)
    text: str
    segments: tuple[Segment, ...]

    @model_validator(mode="after")
    def _offsets_match_text(self) -> Self:
        for w in self.words:
            if self.text[w.start_char : w.end_char] != w.text:
                raise ValueError(f"word offsets do not match the text: {w.text!r}")
        return self

    @property
    def words(self) -> tuple[Word, ...]:
        return tuple(w for s in self.segments for w in s.words)

    def span(self, first: int, stop: int) -> TranscriptSpan:
        """Evidence span covering words[first:stop], quoted exactly from `text`."""
        words = self.words[first:stop]
        if not words:
            raise ValueError(f"empty word span [{first}:{stop}]")
        start, end = words[0].start_char, words[-1].end_char
        return TranscriptSpan(start_char=start, end_char=end, text=self.text[start:end])


class SpeechRegion(_Frozen):
    """A stretch the VAD marks as speech, in seconds from the start of the recording."""

    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_s < self.start_s:
            raise ValueError("end_s must be >= start_s")
        return self

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class PausePosition(StrEnum):
    """Where a pause falls in the transcript (ADR 0011). Only `mid_sentence` may be penalised."""

    SENTENCE_BOUNDARY = "sentence_boundary"  # after . ! ? ؟ …
    CLAUSE_BOUNDARY = "clause_boundary"  # after , ، ; ؛ (reported separately, not penalised)
    MID_SENTENCE = "mid_sentence"
    UNKNOWN = "unknown"  # too little punctuation, or no recognised word next to the pause


class PauseEvent(_Frozen):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    position: PausePosition
    confidence: Confidence
    word_before: int | None = Field(default=None, ge=0)  # index into Transcript.words
    word_after: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_s <= self.start_s:
            raise ValueError("a pause must have end_s > start_s")
        return self

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class RateWindow(_Frozen):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    words: int = Field(ge=0)
    wpm: float = Field(ge=0)


class PitchTrack(_Frozen):
    """pYIN output for one recording (ADR 0012). Frame i is at i * hop_s; f0 is 0.0 when unvoiced.
    Computed once per recording and reused (slice 4 pitch variation)."""

    hop_s: float = Field(gt=0)
    fmin_hz: float = Field(gt=0)
    fmax_hz: float = Field(gt=0)
    f0_hz: tuple[float, ...]
    voiced: tuple[bool, ...]

    @model_validator(mode="after")
    def _same_length(self) -> Self:
        if len(self.f0_hz) != len(self.voiced):
            raise ValueError("f0_hz and voiced must have the same length")
        return self


class FillerKind(StrEnum):
    HESITATION = "hesitation"  # counted as a filler
    DISCOURSE_MARKER = "discourse_marker"  # information only, never penalised (ADR 0012)


class FillerEvent(_Frozen):
    kind: FillerKind
    first_word: int = Field(ge=0)  # inclusive indices into Transcript.words
    last_word: int = Field(ge=0)
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)


class AcousticCandidate(_Frozen):
    """Voiced, steady-pitch sound inside speech with no recognised word on it (FINAL_04 §2)."""

    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    voiced_fraction: float = Field(ge=0, le=1)
    pitch_sd_semitones: float = Field(ge=0)
    agrees_with_lexicon: bool  # next to or overlapping a lexicon hesitation


class RepetitionEvent(_Frozen):
    first_word: int = Field(ge=0)  # inclusive span covering every copy
    last_word: int = Field(ge=0)
    ngram: int = Field(ge=1)  # words in the repeated phrase
    copies: int = Field(ge=2)  # copies in a row ("I I I" = 3)
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)


ProcessingState = Literal["on", "off", "unknown"]


class CaptureProcessing(_Frozen):
    """Browser audio processing during capture (ADR 0013). Auto-gain flattens loudness variation
    and makes baseline comparisons unreliable, so live capture (P6) turns all three off.
    Uploaded files are "unknown"."""

    auto_gain_control: ProcessingState = "unknown"
    noise_suppression: ProcessingState = "unknown"
    echo_cancellation: ProcessingState = "unknown"

    @property
    def all_off(self) -> bool:
        return (self.auto_gain_control, self.noise_suppression, self.echo_cancellation) == (
            "off",
            "off",
            "off",
        )


class LoudnessProfile(_Frozen):
    """Recording conditions (ADR 0013). The absolute level is kept ONLY here, for baseline
    comparisons; it is never a measurement and never feeds a score or the engine."""

    median_speech_dbfs: float | None = None
    noise_floor_dbfs: float | None = None
    snr_db: float | None = None
    native_sample_rate: int | None = Field(default=None, gt=0)
    clipped_runs: int = Field(default=0, ge=0)


class PitchDiagnostics(_Frozen):
    """How the pitch track was cleaned (ADR 0013). Large counts show where the octave rule may
    also have removed real large jumps."""

    usable_frames: int = Field(ge=0)
    usable_voiced_s: float = Field(ge=0)
    octave_folded: int = Field(ge=0)
    dropped: int = Field(ge=0)
    at_limit_share: float = Field(ge=0, le=1)


class AsrQuality(_Frozen):
    """FINAL_04 §2 'ASR confidence' inputs. None when there are no words / segments."""

    word_count: int = Field(ge=0)
    mean_word_probability: float | None = Field(default=None, ge=0, le=1)
    mean_no_speech_prob: float | None = Field(default=None, ge=0, le=1)


class SpeechReport(_Frozen):
    """What `presenceai analyze` writes. Holds derived data only, never audio samples."""

    schema_version: Literal[4] = 4
    source_name: str = Field(min_length=1)  # file name only: no folders, no user names
    model: str = Field(min_length=1)  # key under `asr:` in configs/models.yaml
    transcript: Transcript
    asr_quality: AsrQuality
    speech_regions: tuple[SpeechRegion, ...] = ()
    pauses: tuple[PauseEvent, ...] = ()
    rate_windows: tuple[RateWindow, ...] = ()
    fillers: tuple[FillerEvent, ...] = ()
    acoustic_candidates: tuple[AcousticCandidate, ...] = ()
    repetitions: tuple[RepetitionEvent, ...] = ()
    # Kept in memory for later stages (slice 4); left out of JSON to keep reports small.
    pitch: PitchTrack | None = Field(default=None, exclude=True)
    capture: CaptureProcessing = CaptureProcessing()
    loudness_profile: LoudnessProfile | None = None
    pitch_diagnostics: PitchDiagnostics | None = None
    baseline_source: str | None = None  # file name of the baseline report, if one was given
    measurements: dict[str, Measurement] = Field(default_factory=dict)
    # measure name -> reason; a measure that cannot be computed reliably is never a low score
    not_measurable: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _file_name_only(self) -> Self:
        name = self.source_name
        if PureWindowsPath(name).name != name or PurePosixPath(name).name != name:
            raise ValueError("source_name must be a file name, not a path")
        return self
