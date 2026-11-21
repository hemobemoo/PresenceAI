"""The assessment report written by `presenceai assess` (P2-c, ADR 0019): derived data only."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from presenceai.schemas.rubric import Assessment
from presenceai.schemas.speech import SpeechReport
from presenceai.schemas.text import TextReport


class AssessmentReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    speech: SpeechReport
    text: TextReport
    assessment: Assessment
