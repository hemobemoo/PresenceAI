"""Load a rubric (`resources/rubrics/<id>.yaml`) and its thresholds (`configs/rubric.yaml`).

The loader checks that the two files agree: every condition names a threshold that exists, and
every threshold is used by a condition or an anchor text, so no value is silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from presenceai.evalconfig import REPO_ROOT
from presenceai.lang import placeholders
from presenceai.schemas.rubric import LevelAnchor, Rubric

RESOURCES = REPO_ROOT / "resources" / "rubrics"
CONFIG = REPO_ROOT / "configs" / "rubric.yaml"

__all__ = [
    "PartialCap",
    "RubricThresholds",
    "anchor_text",
    "fill",
    "level_score",
    "load_partial_cap",
    "load_rubric",
    "load_thresholds",
    "placeholders",
]


class RubricThresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["proposed", "provisional", "decided"]
    decided_in: str = Field(min_length=1)  # the ADR that records the values
    values: dict[str, float]
    # Threshold keys whose values are placeholders: a measure with a condition on one of them
    # is reported with its value but not scored (its conditions are dropped; ADR 0022).
    uncalibrated: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _uncalibrated_exist(self) -> Self:
        unknown = set(self.uncalibrated) - set(self.values)
        if unknown:
            raise ValueError(f"uncalibrated names unknown thresholds: {sorted(unknown)}")
        return self


class PartialCap(BaseModel):
    """A level based on at most `max_share` of its measures is capped at `max_level` (ADR 0019)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["proposed", "provisional", "decided"]
    decided_in: str = Field(min_length=1)
    max_share: float = Field(gt=0, lt=1)
    max_level: int = Field(ge=1, le=4)


def _yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    return cast("dict[str, Any]", data)


def load_thresholds(rubric_id: str, *, config: Path = CONFIG) -> RubricThresholds:
    data = _yaml(config)
    if rubric_id not in data:
        raise ValueError(f"{config.name} has no thresholds for {rubric_id!r}")
    return RubricThresholds.model_validate(data[rubric_id])


def load_partial_cap(*, config: Path = CONFIG) -> PartialCap:
    return PartialCap.model_validate(_yaml(config)["partial_cap"])


def load_rubric(rubric_id: str, *, resources: Path = RESOURCES, config: Path = CONFIG) -> Rubric:
    """The rubric, validated against its thresholds."""
    try:
        rubric = Rubric.model_validate(_yaml(resources / f"{rubric_id}.yaml"))
    except ValueError as e:  # pydantic's ValidationError is a ValueError
        raise ValueError(f"{rubric_id}: {e}") from e
    values = load_thresholds(rubric_id, config=config).values
    used: set[str] = set()
    for name, definition in rubric.definitions.items():
        missing = placeholders(definition) - set(values)
        if missing:
            raise ValueError(f"definition {name!r}: text needs {sorted(missing)}")
        used |= placeholders(definition)
    for d in rubric.dimensions:
        for a in d.levels:
            for c in a.conditions:
                if c.threshold is None:
                    continue
                if c.threshold not in values:
                    raise ValueError(f"{d.id} level {a.level}: threshold {c.threshold!r} missing")
                used.add(c.threshold)
            missing = placeholders(a.text) - set(values)
            if missing:
                raise ValueError(f"{d.id} level {a.level}: text needs {sorted(missing)}")
            if placeholders(a.text_ar) != placeholders(a.text):
                raise ValueError(f"{d.id} level {a.level}: text_ar placeholders differ from text")
            used |= placeholders(a.text)
    orphans = set(values) - used
    if orphans:
        raise ValueError(f"thresholds not used by {rubric_id}: {sorted(orphans)}")
    return rubric


def fill(template: str, thresholds: RubricThresholds) -> str:
    """A rubric text with its threshold values filled in."""
    names = placeholders(template)
    return template.format(**{k: v for k, v in thresholds.values.items() if k in names})


def anchor_text(
    anchor: LevelAnchor, thresholds: RubricThresholds, lang: Literal["en", "ar"] = "en"
) -> str:
    """The anchor's wording in `lang` with its threshold values filled in."""
    return fill(anchor.text if lang == "en" else anchor.text_ar, thresholds)


def level_score(level: int) -> float:
    """Skill observation y for a rubric level: (level - 1) / 3 (owner, 2026-11-21, ADR 0017)."""
    if level not in (1, 2, 3, 4):
        raise ValueError(f"rubric levels are 1-4, got {level}")
    return (level - 1) / 3
