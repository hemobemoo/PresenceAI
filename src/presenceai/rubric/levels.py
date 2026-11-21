"""A dimension's level from measure values (ADR 0017, decision 3 and 4).

A level is reached when every one of its **computable** conditions holds and at least one is
computable; the dimension gets the highest level reached. A measure that is missing (None: camera
off, no time limit, Arabic content, unreliable punctuation) is dropped from the conditions; it is
never read as a low value. A dimension with no computable measure at all is not measurable
(None), never level 1.
"""

from __future__ import annotations

from collections.abc import Mapping

from presenceai.rubric.loader import RubricThresholds
from presenceai.schemas.rubric import Condition, Dimension


def _holds(c: Condition, value: float, t: RubricThresholds) -> bool:
    if c.op == "present":
        return value >= 1
    limit = t.values[c.threshold or ""]
    return value >= limit if c.op == ">=" else value <= limit


def dimension_level(
    dimension: Dimension, values: Mapping[str, float | None], t: RubricThresholds
) -> int | None:
    """The level 1-4, or None when none of the dimension's measures can be computed."""
    measures = {c.measure for a in dimension.levels for c in a.conditions}
    if all(values.get(m) is None for m in measures):
        return None
    for anchor in dimension.levels[:-1]:
        computable = [c for c in anchor.conditions if values.get(c.measure) is not None]
        if computable and all(_holds(c, values[c.measure] or 0.0, t) for c in computable):
            return anchor.level
    return 1
