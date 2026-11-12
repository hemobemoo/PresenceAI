"""Only five expression labels may exist (AGENTS.md rule 3, FINAL_11 §2)."""

import pytest

from presenceai.coach.guardrails import contains_banned_term
from presenceai.schemas.features import ExpressionLabel
from presenceai.vision.expressions import ALLOWED_LABELS

EXPECTED = {
    "smile_visible",
    "brows_raised",
    "brows_lowered",
    "eyes_closed",
    "mouth_moving_silent",
}


def test_allowed_labels_are_exactly_the_five():
    assert set(ALLOWED_LABELS) == EXPECTED
    assert {label.value for label in ExpressionLabel} == EXPECTED


def test_allowed_labels_is_immutable():
    assert isinstance(ALLOWED_LABELS, frozenset)


@pytest.mark.parametrize("bad", ["happy", "nervous", "confident", "smile", "angry", ""])
def test_other_labels_are_rejected(bad: str):
    with pytest.raises(ValueError):
        ExpressionLabel(bad)


@pytest.mark.parametrize("label", sorted(EXPECTED))
def test_labels_contain_no_banned_term(label: str):
    assert not contains_banned_term(label.replace("_", " "), "en")
