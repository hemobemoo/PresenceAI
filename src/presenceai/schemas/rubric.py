"""Rubric definitions (P2-a, ADR 0017): an adaptation of the AAC&U VALUE Oral Communication rubric.

A rubric has dimensions; each dimension has four levels (4 Capstone … 1 Benchmark). A level is
reached when every one of its conditions holds; the dimension's level is the highest level
reached (level 1 has no conditions and is the default). A condition compares one measure with
one named threshold; threshold values live in `configs/rubric.yaml` (AGENTS.md rule 7).

The rubric files are CC BY-NC-SA 4.0 (they adapt the VALUE rubric) and live in
`resources/rubrics/`, apart from the Apache-2.0 code (NOTICE).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from presenceai.schemas.assessment import Confidence, Evidence

LEVELS = (4, 3, 2, 1)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Condition(_Frozen):
    measure: str = Field(min_length=1)  # a Measurement key, e.g. "fillers.per_100_words"
    op: Literal[">=", "<=", "present"]  # "present": a yes/no measure is yes (value >= 1)
    threshold: str | None = None  # a key in configs/rubric.yaml <rubric>.values
    # required: when the measure is missing the level is NOT awarded (instead of dropping the
    # condition). For a condition that defines the level, such as the limitation cue of
    # follow-up level 4 (owner, 2026-11-25, ADR 0022): without it the level would reduce to
    # its other conditions.
    required: bool = False

    @model_validator(mode="after")
    def _threshold_unless_present(self) -> Condition:
        if (self.op == "present") != (self.threshold is None):
            raise ValueError("comparisons name a threshold; 'present' has none")
        return self


class LevelAnchor(_Frozen):
    level: int = Field(ge=1, le=4)
    text: str = Field(min_length=1)  # observable wording; {threshold} placeholders
    text_ar: str = Field(min_length=1)  # Arabic draft; the same placeholders (native review)
    conditions: tuple[Condition, ...] = ()


class MeasureSpec(_Frozen):
    status: Literal["available", "planned"]
    planned_in: str | None = None  # slice that builds it, e.g. "P2-b"
    # a measure that cannot be computed is dropped from the conditions, never penalised
    requires: Literal["none", "camera", "session_limit", "followups"]
    languages: tuple[Literal["en", "ar"], ...] = Field(min_length=1)
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def _planned_names_a_slice(self) -> MeasureSpec:
        if (self.status == "planned") != (self.planned_in is not None):
            raise ValueError("planned measures name the slice that builds them, and only they")
        return self


class Dimension(_Frozen):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    name_ar: str = Field(min_length=1)
    value_dimension: str = Field(min_length=1)  # the VALUE dimension it adapts
    skills: tuple[str, ...] = Field(min_length=1)  # engine skills it feeds (FINAL_05 §1)
    levels: tuple[LevelAnchor, ...]
    left_out: tuple[str, ...] = ()  # parts of the VALUE wording no rule can observe

    @model_validator(mode="after")
    def _four_ordered_levels(self) -> Dimension:
        if tuple(a.level for a in self.levels) != LEVELS:
            raise ValueError(f"{self.id}: levels must be 4, 3, 2, 1 in that order")
        if self.levels[-1].conditions:
            raise ValueError(f"{self.id}: level 1 is the default and has no conditions")
        if not all(a.conditions for a in self.levels[:-1]):
            raise ValueError(f"{self.id}: levels 4, 3 and 2 need at least one condition")
        return self


class Source(_Frozen):
    title: str = Field(min_length=1)
    copy_url: str = Field(min_length=1)
    copy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    note: str | None = None


class Rubric(_Frozen):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: Literal["draft", "reviewed", "accepted"]
    licence: Literal["CC BY-NC-SA 4.0"]
    attribution: str = Field(min_length=1)
    source: Source
    definitions: dict[str, str] = Field(min_length=1)  # operational definitions; {threshold}s
    scale: dict[int, str]  # level -> name (Capstone, Milestone, Benchmark)
    scale_ar: dict[int, str]
    measures: dict[str, MeasureSpec]
    dimensions: tuple[Dimension, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Rubric:
        if tuple(sorted(self.scale, reverse=True)) != LEVELS or set(self.scale_ar) != set(LEVELS):
            raise ValueError("scale and scale_ar must name levels 4, 3, 2, 1")
        for d in self.dimensions:
            for a in d.levels:
                for c in a.conditions:
                    if c.measure not in self.measures:
                        raise ValueError(f"{d.id}: unknown measure {c.measure!r}")
        return self


# --- assessment results (P2-c, ADR 0019) ------------------------------------------------------


class ConditionResult(_Frozen):
    level: int = Field(ge=2, le=4)
    measure: str = Field(min_length=1)
    op: Literal[">=", "<=", "present"]
    threshold: str | None = None
    limit: float | None = None  # the threshold's value; None for "present"
    value: float | None = None  # None: the measure is missing (never read as low)
    holds: bool | None = None  # None when the measure is missing
    required: bool = False  # missing -> this level is not awarded (Condition.required)


class DimensionResult(_Frozen):
    dimension: str = Field(min_length=1)
    skills: tuple[str, ...]
    level: int | None = Field(default=None, ge=1, le=4)  # None: not measurable
    # the level the conditions reached before the partial cap (ADR 0019); None: not capped
    capped_from: int | None = Field(default=None, ge=2, le=4)
    score: float | None = Field(default=None, ge=0, le=1)  # y = (level - 1) / 3, after the cap
    confidence: Confidence | None = None  # lowest among the measured (deciding) measures
    measured: tuple[str, ...] = ()  # measures that could be computed
    missing: tuple[str, ...] = ()  # measures that could not, never counted as low
    # absent by the user's choice (camera off, opted out): left out of M ("N of M") and the cap
    excluded: tuple[str, ...] = ()
    # why each `missing` measure could not be computed (report cards, P3-d); system wording
    # from lang/, never the user's words. Not copied to the clear summary (ADR 0023 item 12).
    missing_reasons: dict[str, str] = Field(default_factory=dict)
    conditions: tuple[ConditionResult, ...] = ()
    next_level: tuple[ConditionResult, ...] = ()  # unmet measured conditions of level + 1
    not_measurable: str | None = None

    @model_validator(mode="after")
    def _level_or_reason(self) -> DimensionResult:
        if (self.level is None) != (self.not_measurable is not None):
            raise ValueError("a dimension has either a level or a not-measurable reason")
        if self.level is not None and (self.score is None or self.confidence is None):
            raise ValueError("a level needs a score and a confidence")
        if self.capped_from is not None and (self.level is None or self.capped_from <= self.level):
            raise ValueError("a capped level is lower than the level it was capped from")
        return self


class Finding(_Frozen):
    """One feedback statement per dimension: observable wording, evidence, confidence."""

    dimension: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    level: int | None = Field(default=None, ge=1, le=4)
    capped_from: int | None = Field(default=None, ge=2, le=4)  # as DimensionResult
    confidence: Confidence | None = None
    measured: int = Field(ge=0)  # "based on N of M measures", always shown
    total: int = Field(ge=1)  # M: the dimension's measures minus those absent by user choice
    evidence: tuple[Evidence, ...] = ()

    @model_validator(mode="after")
    def _evidence_for_a_level(self) -> Finding:
        if self.level is not None and not self.evidence:
            raise ValueError("a finding with a level needs evidence")
        return self


class Assessment(_Frozen):
    schema_version: Literal[1] = 1
    scenario: str = Field(min_length=1)
    rubric_version: str = Field(min_length=1)
    thresholds_status: str = Field(min_length=1)
    language: Literal["en", "ar"]
    dimensions: tuple[DimensionResult, ...]
    findings: tuple[Finding, ...]
    # App mode (P3): dimensions whose scoring or evidence check failed; shown as not measurable
    failed_dimensions: tuple[str, ...] = ()
    evidence_verified: Literal[True] = (
        True  # every finding's evidence was checked (rubric/evidence)
    )
