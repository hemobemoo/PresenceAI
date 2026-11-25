"""A dimension's level from measure values (ADR 0017, decisions 3 and 4; ADR 0019).

A level is reached when every one of its **computable** conditions holds and at least one is
computable; the dimension gets the highest level reached. A measure that is missing (None: camera
off, no time limit, Arabic content, unreliable punctuation) is dropped from the conditions; it is
never read as a low value. A dimension with no computable measure at all is not measurable
(None), never level 1.

A **required** condition is the exception: when its measure is missing, its level is not
awarded (the level is unavailable, not failed); lower levels are still reachable.
"""

from __future__ import annotations

from collections.abc import Mapping

from presenceai.rubric.loader import RubricThresholds
from presenceai.schemas.rubric import Condition, ConditionResult, Dimension


def _holds(c: Condition, value: float, t: RubricThresholds) -> bool:
    if c.op == "present":
        return value >= 1
    limit = t.values[c.threshold or ""]
    return value >= limit if c.op == ">=" else value <= limit


def measures_of(dimension: Dimension) -> tuple[str, ...]:
    """The dimension's measures, in the order they first appear."""
    return tuple(dict.fromkeys(c.measure for a in dimension.levels for c in a.conditions))


def condition_results(
    dimension: Dimension, values: Mapping[str, float | None], t: RubricThresholds
) -> tuple[ConditionResult, ...]:
    out: list[ConditionResult] = []
    for anchor in dimension.levels[:-1]:
        for c in anchor.conditions:
            value = values.get(c.measure)
            out.append(
                ConditionResult(
                    level=anchor.level,
                    measure=c.measure,
                    op=c.op,
                    threshold=c.threshold,
                    limit=t.values[c.threshold] if c.threshold else None,
                    value=value,
                    holds=None if value is None else _holds(c, value, t),
                    required=c.required,
                )
            )
    return tuple(out)


def unavailable_levels(results: tuple[ConditionResult, ...]) -> tuple[int, ...]:
    """Levels that cannot be awarded because a required measure is missing."""
    return tuple(sorted({r.level for r in results if r.required and r.value is None}))


def level_from(results: tuple[ConditionResult, ...]) -> int | None:
    """The level from condition results; None when no measure was computed."""
    if all(r.value is None for r in results):
        return None
    for level in (4, 3, 2):
        if level in unavailable_levels(results):
            continue
        computed = [r for r in results if r.level == level and r.holds is not None]
        if computed and all(r.holds for r in computed):
            return level
    return 1


def dimension_level(
    dimension: Dimension, values: Mapping[str, float | None], t: RubricThresholds
) -> int | None:
    """The level 1-4, or None when none of the dimension's measures can be computed."""
    return level_from(condition_results(dimension, values, t))
