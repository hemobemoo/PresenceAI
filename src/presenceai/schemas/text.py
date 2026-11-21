"""Text analysis report (P2-b, ADR 0018): the content measures the rubric uses (ADR 0017)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from presenceai.schemas.assessment import Measurement


class TextReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    language: Literal["en", "ar"]
    scenario: str = Field(min_length=1)
    punctuation_reliable: bool
    sentences: int = Field(ge=0)
    measurements: dict[str, Measurement]
    not_measurable: dict[str, str]  # measure -> reason (missing, never a low value)

    @model_validator(mode="after")
    def _each_measure_once(self) -> TextReport:
        both = set(self.measurements) & set(self.not_measurable)
        if both:
            raise ValueError(f"measured and not measurable at once: {sorted(both)}")
        return self
