"""S2 rubric v0.1 (ADR 0022): S1's dimensions scaled to 90 s, plus follow-up handling."""

import re
from typing import Any

import pytest
import yaml

from presenceai.coach.guardrails import contains_banned_term, contains_verdict_term
from presenceai.evalconfig import REPO_ROOT
from presenceai.rubric import levels, loader, review
from presenceai.schemas.rubric import Rubric

S1, S2 = "s1_presentation", "s2_project_defense"
ENGINE_SKILLS = set(
    yaml.safe_load((REPO_ROOT / "configs" / "engine.yaml").read_text("utf-8"))["targets"]
)
REVIEW_DOC = REPO_ROOT / "docs" / "rubric" / "S2_ANCHORS.md"
# ADR 0022 §3: counts scaled from S1 by ratio (not from data); every other value is S1's.
SCALED = {
    "sup_items_high": 3, "sup_items_mid": 2, "sup_types_high": 2, "cm_recurrences_min": 1,
    "org_transitions_high": 2, "org_transitions_mid": 1, "lang_undefined_max_2": 2,
    "del_overrun_3": 10, "del_overrun_2": 20,
}  # fmt: skip
FOLLOWUP = {"fu_relevance_high", "fu_relevance_mid", "fu_relevance_low", "fu_latency_4",
            "fu_latency_3"}  # fmt: skip


@pytest.fixture(scope="module")
def rubric() -> Rubric:
    return loader.load_rubric(S2)


@pytest.fixture(scope="module")
def thresholds() -> loader.RubricThresholds:
    return loader.load_thresholds(S2)


def _dim(rubric: Rubric, dim_id: str) -> Any:
    return next(d for d in rubric.dimensions if d.id == dim_id)


def test_s1_dimensions_plus_followup_handling(rubric: Rubric):
    s1 = loader.load_rubric(S1)
    want = [*(d.id for d in s1.dimensions), "followup_handling"]
    assert [d.id for d in rubric.dimensions] == want
    for d in rubric.dimensions:
        assert set(d.skills) <= ENGINE_SKILLS, d.id
    assert _dim(rubric, "followup_handling").skills == ("followup_handling",)
    assert rubric.status == "draft" and rubric.licence == "CC BY-NC-SA 4.0"


def test_counts_scaled_by_ratio_and_everything_else_as_s1(thresholds: Any):
    s1 = loader.load_thresholds(S1).values
    assert thresholds.status == "provisional" and thresholds.decided_in == "ADR 0022"
    assert set(thresholds.values) == set(s1) | FOLLOWUP
    for key, value in s1.items():
        assert thresholds.values[key] == SCALED.get(key, value), key


def test_s1_wording_reused_for_the_five_value_dimensions(rubric: Rubric):
    s1 = {d.id: d for d in loader.load_rubric(S1).dimensions}
    for d in rubric.dimensions[:-1]:
        assert [a.text for a in d.levels] == [a.text for a in s1[d.id].levels], d.id
        assert [a.conditions for a in d.levels] == [a.conditions for a in s1[d.id].levels], d.id


def test_followup_levels_weaker_answer_decides(rubric: Rubric):
    conds = {a.level: {(c.measure, c.op, c.threshold) for c in a.conditions}
             for a in _dim(rubric, "followup_handling").levels}  # fmt: skip
    assert conds[4] == {("followup.relevance_min", ">=", "fu_relevance_high"),
                        ("followup.latency_max_s", "<=", "fu_latency_4"),
                        ("followup.limitation_acknowledged", "present", None)}  # fmt: skip
    assert conds[3] == {("followup.relevance_min", ">=", "fu_relevance_mid"),
                        ("followup.latency_max_s", "<=", "fu_latency_3")}  # fmt: skip
    # level 2 needs BOTH answers >= low: the minimum over the answers (owner, 2026-11-23)
    assert conds[2] == {("followup.relevance_min", ">=", "fu_relevance_low")}
    assert conds[1] == set()


def test_followup_measures_need_followups_and_relevance_is_not_measured_in_arabic(rubric: Rubric):
    fu = {k: m for k, m in rubric.measures.items() if k.startswith("followup.")}
    assert set(fu) == {"followup.relevance_min", "followup.latency_max_s",
                       "followup.limitation_acknowledged"}  # fmt: skip
    assert all(m.requires == "followups" and m.planned_in == "S2-b" for m in fu.values())
    assert fu["followup.relevance_min"].languages == ("en",)
    assert fu["followup.limitation_acknowledged"].languages == ("en",)
    assert set(fu["followup.latency_max_s"].languages) == {"en", "ar"}


def test_no_followups_recorded_is_not_measurable_never_low(rubric: Rubric, thresholds: Any):
    d = _dim(rubric, "followup_handling")
    assert levels.dimension_level(d, {}, thresholds) is None
    v = thresholds.values  # relative to the thresholds, so calibration (S2-b) keeps this valid
    values = {"followup.relevance_min": v["fu_relevance_high"], "followup.latency_max_s": None,
              "followup.limitation_acknowledged": 1.0}  # fmt: skip
    assert levels.dimension_level(d, values, thresholds) == 4  # latency missing: dropped, not low
    values["followup.latency_max_s"] = v["fu_latency_3"] + 1  # one answer late: 4 and 3 fail
    assert levels.dimension_level(d, values, thresholds) == 2
    values["followup.relevance_min"] = v["fu_relevance_low"] - 0.01  # weaker answer off-topic
    assert levels.dimension_level(d, values, thresholds) == 1


def test_anchors_are_observable_numbers_without_verdict_words(rubric: Rubric, thresholds: Any):
    for d in rubric.dimensions:
        for a in d.levels:
            for lang in ("en", "ar"):
                text = loader.anchor_text(a, thresholds, lang)
                assert "{" not in text and not contains_verdict_term(text, lang), (d.id, text)
                assert not contains_banned_term(text, lang), (d.id, text)
        for a in d.levels[:-1]:
            assert re.search(r"\d", loader.anchor_text(a, thresholds)), (d.id, a.level)


def test_limitation_cue_definition_excludes_hedging(rubric: Rubric):
    text = rubric.definitions["limitation cue"]
    assert "not limitation cues" in text and "I'm not sure" in text


def test_scenario_points_to_the_rubric(rubric: Rubric):
    scenario = yaml.safe_load((REPO_ROOT / "scenarios" / f"{S2}.yaml").read_text("utf-8"))
    assert scenario["rubric"] == S2 and scenario["followups"] == 2
    assert scenario["answer_recording"] == "auto_start"  # ADR 0022 §8
    assert set(scenario["skills"]) == {s for d in rubric.dimensions for s in d.skills}


def test_review_document_is_generated_and_marked_unreviewed(rubric: Rubric, thresholds: Any):
    doc = review.review_markdown(rubric, thresholds)
    assert REVIEW_DOC.read_text("utf-8") == doc, (
        "run: uv run python scripts/render_rubric_review.py --rubric s2_project_defense"
    )
    assert "not yet reviewed" in doc and "no follow-up answer recorded" in doc
    assert "`followup_handling` is added" in doc
