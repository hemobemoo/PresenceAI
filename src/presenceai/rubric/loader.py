"""Load a rubric (`resources/rubrics/<id>.yaml`) and its thresholds (`configs/rubric.yaml`).

The loader checks that the two files agree: every condition names a threshold that exists, and
every threshold is used by a condition or an anchor text, so no value is silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from presenceai.evalconfig import REPO_ROOT
from presenceai.lang import placeholders
from presenceai.schemas.rubric import LevelAnchor, Rubric

RESOURCES = REPO_ROOT / "resources" / "rubrics"
CONFIG = REPO_ROOT / "configs" / "rubric.yaml"

__all__ = [
    "RubricThresholds",
    "anchor_text",
    "level_score",
    "load_rubric",
    "load_thresholds",
    "placeholders",
]


class RubricThresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["proposed", "decided"]
    decided_in: str = Field(min_length=1)  # the ADR that records the values
    values: dict[str, float]


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


def load_rubric(rubric_id: str, *, resources: Path = RESOURCES, config: Path = CONFIG) -> Rubric:
    """The rubric, validated against its thresholds."""
    try:
        rubric = Rubric.model_validate(_yaml(resources / f"{rubric_id}.yaml"))
    except ValueError as e:  # pydantic's ValidationError is a ValueError
        raise ValueError(f"{rubric_id}: {e}") from e
    values = load_thresholds(rubric_id, config=config).values
    used: set[str] = set()
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
            used |= placeholders(a.text)
    orphans = set(values) - used
    if orphans:
        raise ValueError(f"thresholds not used by {rubric_id}: {sorted(orphans)}")
    return rubric


def anchor_text(anchor: LevelAnchor, thresholds: RubricThresholds) -> str:
    """The anchor's wording with its threshold values filled in."""
    names = placeholders(anchor.text)
    return anchor.text.format(**{k: v for k, v in thresholds.values.items() if k in names})


def level_score(level: int) -> float:
    """Skill observation y for a rubric level: (level - 1) / 3 (owner, 2026-11-21, ADR 0017)."""
    if level not in (1, 2, 3, 4):
        raise ValueError(f"rubric levels are 1-4, got {level}")
    return (level - 1) / 3
