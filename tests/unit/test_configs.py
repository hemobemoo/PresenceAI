from typing import Any

import pytest
import yaml

from presenceai.settings import CONFIG_DIR

SKILLS = {
    "central_message",
    "organization",
    "opening_conclusion",
    "supporting_material",
    "language_clarity",
    "concision_timing",
    "fluency",
    "pacing",
    "vocal_variety",
    "audience_gaze",
    "posture_gesture",
    "followup_handling",
}
EXPECTED = [
    "default.yaml",
    "engine.yaml",
    "live.yaml",
    "models.yaml",
    "profiles/lite.yaml",
    "profiles/standard.yaml",
    "profiles/space.yaml",
    "runtime/local.yaml",
    "runtime/space.yaml",
    "runtime/offline.yaml",
]


def _load(rel: str) -> dict[str, Any]:
    data = yaml.safe_load((CONFIG_DIR / rel).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data  # pyright: ignore[reportUnknownVariableType]


@pytest.mark.parametrize("rel", EXPECTED)
def test_config_file_parses(rel: str):
    assert _load(rel)


def test_engine_targets_cover_every_skill():
    assert set(_load("engine.yaml")["targets"]) == SKILLS


def test_live_prompt_skills_are_known():
    cats = _load("live.yaml")["prompts"]["categories_by_skill"]
    assert set(cats) - {"default"} <= SKILLS


def test_runtime_and_profile_files_match_settings_literals():
    assert {p.stem for p in (CONFIG_DIR / "runtime").glob("*.yaml")} == {
        "local",
        "space",
        "offline",
    }
    assert {p.stem for p in (CONFIG_DIR / "profiles").glob("*.yaml")} == {
        "lite",
        "standard",
        "space",
    }
