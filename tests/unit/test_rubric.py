"""S1 rubric draft (P2-a, ADR 0017): structure, thresholds, wording and the review document.

The rubric is an adaptation of the AAC&U VALUE Oral Communication rubric (CC BY-NC-SA 4.0) in
`resources/rubrics/`; its proposed thresholds are in `configs/rubric.yaml` (AGENTS.md rule 7).
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from presenceai.coach.guardrails import contains_banned_term, contains_verdict_term
from presenceai.evalconfig import REPO_ROOT
from presenceai.rubric import loader, review
from presenceai.schemas.rubric import Rubric

S1 = "s1_presentation"
DIMENSIONS = {"central_message", "organization", "language", "supporting_material", "delivery"}
ENGINE_SKILLS = set(
    yaml.safe_load((REPO_ROOT / "configs" / "engine.yaml").read_text("utf-8"))["targets"]
)
REVIEW_DOC = REPO_ROOT / "docs" / "rubric" / "S1_ANCHORS_FOR_REVIEW.md"


@pytest.fixture(scope="module")
def rubric() -> Rubric:
    return loader.load_rubric(S1)


@pytest.fixture(scope="module")
def thresholds() -> loader.RubricThresholds:
    return loader.load_thresholds(S1)


def test_five_value_dimensions_with_four_ordered_levels(rubric: Rubric):
    assert {d.id for d in rubric.dimensions} == DIMENSIONS
    for d in rubric.dimensions:
        assert [a.level for a in d.levels] == [4, 3, 2, 1], d.id
        assert d.levels[-1].conditions == (), f"{d.id}: level 1 is the default, no conditions"
        assert all(a.conditions for a in d.levels[:-1]), d.id


def test_level_scores_are_level_minus_one_over_three():
    assert [loader.level_score(n) for n in (1, 2, 3, 4)] == [0.0, 1 / 3, 2 / 3, 1.0]
    with pytest.raises(ValueError):
        loader.level_score(0)


def test_dimensions_feed_known_engine_skills(rubric: Rubric):
    for d in rubric.dimensions:
        assert d.skills and set(d.skills) <= ENGINE_SKILLS, d.id


def test_every_condition_names_a_declared_measure_and_a_threshold(rubric: Rubric, thresholds: Any):
    used: set[str] = set()
    for d in rubric.dimensions:
        for a in d.levels:
            for c in a.conditions:
                assert c.measure in rubric.measures, (d.id, c.measure)
                if c.threshold is not None:
                    assert c.threshold in thresholds.values, (d.id, c.threshold)
                    used.add(c.threshold)
    texts = {p for d in rubric.dimensions for a in d.levels for p in loader.placeholders(a.text)}
    assert set(thresholds.values) == used | texts, "every threshold is used by a condition or text"


def test_every_threshold_is_marked_proposed(thresholds: Any):
    assert thresholds.status == "proposed"
    assert thresholds.decided_in == "ADR 0017"


def test_measures_are_available_or_planned_in_a_named_slice(rubric: Rubric):
    speech_src = "\n".join(
        p.read_text("utf-8") for p in (REPO_ROOT / "src/presenceai/speech").glob("*.py")
    )
    for key, m in rubric.measures.items():
        if m.status == "available":
            assert f'"{key}"' in speech_src, f"{key} is marked available but no module emits it"
        else:
            assert m.planned_in, f"{key}: planned measures name the slice that builds them"


def test_anchor_texts_fill_from_thresholds_without_verdict_words(rubric: Rubric, thresholds: Any):
    for d in rubric.dimensions:
        names = [d.name, *(loader.anchor_text(a, thresholds) for a in d.levels)]
        for value in names:
            assert "{" not in value and value.strip(), (d.id, value)
            assert not contains_verdict_term(value, "en"), (d.id, value)
            assert not contains_banned_term(value, "en"), (d.id, value)


def test_anchors_quote_numbers_not_judgements(rubric: Rubric, thresholds: Any):
    # AGENTS.md wording: observable behaviour + number; every non-default level has a number
    for d in rubric.dimensions:
        for a in d.levels[:-1]:
            assert re.search(r"\d", loader.anchor_text(a, thresholds)), (d.id, a.level)


def test_licence_and_attribution(rubric: Rubric):
    assert rubric.licence == "CC BY-NC-SA 4.0"
    assert "Oral Communication VALUE Rubric" in rubric.attribution
    assert "AAC&U" in rubric.attribution and "CC BY-NC-SA 4.0" in rubric.attribution
    assert rubric.status == "draft"


def test_scenario_points_to_the_rubric(rubric: Rubric):
    scenario = yaml.safe_load((REPO_ROOT / "scenarios" / f"{S1}.yaml").read_text("utf-8"))
    assert scenario["rubric"] == S1
    assert set(scenario["skills"]) == {s for d in rubric.dimensions for s in d.skills}


def test_review_document_is_generated_from_the_rubric(rubric: Rubric, thresholds: Any):
    assert REVIEW_DOC.read_text("utf-8") == review.review_markdown(rubric, thresholds), (
        "run: uv run python scripts/render_rubric_review.py"
    )


# --- the loader rejects inconsistent files ----------------------------------------------------


def _write(tmp_path: Path, rubric_data: dict[str, Any], values: dict[str, float]) -> Path:
    (tmp_path / "rubrics").mkdir()
    (tmp_path / "rubrics" / "x.yaml").write_text(yaml.safe_dump(rubric_data), "utf-8")
    cfg = {"x": {"status": "proposed", "decided_in": "ADR 0017", "values": values}}
    (tmp_path / "rubric.yaml").write_text(yaml.safe_dump(cfg), "utf-8")
    return tmp_path


def _minimal(threshold: str = "t", levels: tuple[int, ...] = (4, 3, 2, 1)) -> dict[str, Any]:
    cond = [{"measure": "m.x", "op": ">=", "threshold": threshold}]
    return {
        "id": "x", "version": "0.1", "status": "draft", "licence": "CC BY-NC-SA 4.0",
        "attribution": "a", "source": {"title": "t", "copy_url": "u", "copy_sha256": "0" * 64},
        "scale": {str(n): f"L{n}" for n in (4, 3, 2, 1)},
        "measures": {"m.x": {"status": "planned", "planned_in": "P2-b", "requires": "none",
                             "languages": ["en"], "description": "d"}},
        "dimensions": [{"id": "d", "name": "D", "value_dimension": "D", "skills": ["fluency"],
                        "levels": [{"level": n, "text": f"level {n} {{t}}",
                                    "conditions": cond if n > 1 else []} for n in levels]}],
    }  # fmt: skip


def test_loader_accepts_a_minimal_rubric(tmp_path: Path):
    root = _write(tmp_path, _minimal(), {"t": 1.0})
    assert loader.load_rubric("x", resources=root / "rubrics", config=root / "rubric.yaml")


@pytest.mark.parametrize(
    ("data", "values", "error"),
    [
        (_minimal(threshold="missing"), {"t": 1.0}, "missing"),
        (_minimal(), {"t": 1.0, "orphan": 2.0}, "orphan"),
        (_minimal(levels=(4, 3, 1)), {"t": 1.0}, "levels"),
    ],
)
def test_loader_rejects_inconsistent_files(tmp_path: Path, data: Any, values: Any, error: str):
    root = _write(tmp_path, data, values)
    with pytest.raises(ValueError, match=error):
        loader.load_rubric("x", resources=root / "rubrics", config=root / "rubric.yaml")
