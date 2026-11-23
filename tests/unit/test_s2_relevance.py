"""The S2 relevance calibration script (ADR 0022 §5): parsing only. The owner's pairs are
calibration data and never appear in a test; these blocks are made up."""

import importlib.util
import sys

import pytest

from presenceai.evalconfig import REPO_ROOT
from presenceai.rubric import loader

_spec = importlib.util.spec_from_file_location(
    "s2_score_pairs", REPO_ROOT / "experiments" / "s2_relevance" / "score_pairs.py"
)
assert _spec is not None and _spec.loader is not None
score_pairs = importlib.util.module_from_spec(_spec)
sys.modules["s2_score_pairs"] = score_pairs
_spec.loader.exec_module(score_pairs)

TEXT = """id: 1
question: Why this design?
answer: Because it is simple.
label: good

id: 2
question: Why this design?
answer: Lunch is at noon.
label: off-topic
"""


def test_parse_reads_blocks():
    pairs = score_pairs.parse(TEXT.replace("\n", "\r\n"))  # Windows line ends too
    assert [(p.id, p.label) for p in pairs] == [(1, "good"), (2, "off-topic")]
    assert pairs[1].answer == "Lunch is at noon."


@pytest.mark.parametrize(
    "bad",
    [
        TEXT.replace("label: good", "label: great"),
        TEXT.replace("answer: Because it is simple.\n", ""),
        TEXT.replace("id: 2", "id: 1"),
        TEXT.replace("answer: Lunch is at noon.", "answer:"),
    ],
)
def test_parse_refuses_bad_blocks(bad: str):
    with pytest.raises(ValueError):
        score_pairs.parse(bad)


def test_summary():
    s = score_pairs.summary([0.8, 0.9, 1.0])
    assert s["n"] == 3 and s["median"] == 0.9 and s["min"] == 0.8


def test_relevance_thresholds_are_not_scored_until_calibrated():
    t = loader.load_thresholds("s2_project_defense")
    assert set(t.uncalibrated) == {"fu_relevance_high", "fu_relevance_mid", "fu_relevance_low"}
