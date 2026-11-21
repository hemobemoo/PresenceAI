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

LEVELS = (4, 3, 2, 1)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Condition(_Frozen):
    measure: str = Field(min_length=1)  # a Measurement key, e.g. "fillers.per_100_words"
    op: Literal[">=", "<=", "present"]  # "present": a yes/no measure is yes (value >= 1)
    threshold: str | None = None  # a key in configs/rubric.yaml <rubric>.values

    @model_validator(mode="after")
    def _threshold_unless_present(self) -> Condition:
        if (self.op == "present") != (self.threshold is None):
            raise ValueError("comparisons name a threshold; 'present' has none")
        return self


class LevelAnchor(_Frozen):
    level: int = Field(ge=1, le=4)
    text: str = Field(min_length=1)  # observable wording; {threshold} placeholders
    conditions: tuple[Condition, ...] = ()


class MeasureSpec(_Frozen):
    status: Literal["available", "planned"]
    planned_in: str | None = None  # slice that builds it, e.g. "P2-b"
    # a measure that cannot be computed is dropped from the conditions, never penalised
    requires: Literal["none", "camera", "session_limit"]
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
    scale: dict[int, str]  # level -> name (Capstone, Milestone, Benchmark)
    measures: dict[str, MeasureSpec]
    dimensions: tuple[Dimension, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Rubric:
        if tuple(sorted(self.scale, reverse=True)) != LEVELS:
            raise ValueError("scale must name levels 4, 3, 2, 1")
        for d in self.dimensions:
            for a in d.levels:
                for c in a.conditions:
                    if c.measure not in self.measures:
                        raise ValueError(f"{d.id}: unknown measure {c.measure!r}")
        return self
