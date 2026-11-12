"""Assessment types: every measure is a `Measurement` with evidence and a confidence.

Defined in docs/plan/FINAL_04_MULTIMODAL_MEASUREMENT.md (intro and section 6):
``Measurement(value, unit, confidence, method, evidence[], language, live)``.
Evidence is a transcript span, a time span, or both. This module imports nothing
from the project.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TimeSpan(_Frozen):
    """Seconds from the start of the recording."""

    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_s < self.start_s:
            raise ValueError("end_s must be >= start_s")
        return self


class TranscriptSpan(_Frozen):
    """Character offsets into the transcript text, plus the quoted text."""

    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be > start_char")
        return self


class Evidence(_Frozen):
    transcript: TranscriptSpan | None = None
    time: TimeSpan | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> Self:
        if self.transcript is None and self.time is None:
            raise ValueError("evidence needs a transcript span, a time span, or both")
        return self


class Measurement(_Frozen):
    value: float
    unit: str = Field(min_length=1)
    confidence: Confidence
    method: str = Field(min_length=1)
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    language: Literal["en", "ar"]
    live: bool
