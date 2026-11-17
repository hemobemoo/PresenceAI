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
    "evaluation.yaml",
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


def test_e2_targets_match_adr_0009():
    e2 = _load("evaluation.yaml")["e2"]
    assert e2["wpm_error_pct_max"] == 5.0
    assert e2["pause_f1_min"] == {"0.5": 0.90, "1.0": 0.90}
    assert e2["duration_error_s_max"] == 0.10
    assert e2["report_per_clip"] is True


def test_speech_values_match_adr_0011():
    sp = _load("default.yaml")["speech"]
    assert sp["vad"] == {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
        "speech_pad_ms": 0,
    }
    assert sp["pauses"]["thresholds_s"] == [0.25, 1.0, 2.0]
    assert sp["pauses"]["sentence_marks"] == [".", "!", "?", "؟", "…"]
    assert sp["pauses"]["clause_marks"] == [",", "،", ";", "؛"]
    assert sp["pauses"]["max_words_per_sentence_mark"] == 30
    assert sp["rate"] == {"window_s": 10.0, "hop_s": 5.0, "min_speaking_time_s": 10.0}
    assert sp["syllables"]["min_dip_db"] == 2.0
    assert sp["syllables"]["silence_threshold_db"] == -25.0
    assert sp["confidence"]["high"] == {"min_word_probability": 0.85, "min_speaking_time_s": 30.0}
    assert sp["confidence"]["medium"] == {"min_word_probability": 0.70, "min_speaking_time_s": 10.0}


def test_e2_dev_test_split_is_fixed():
    split = _load("evaluation.yaml")["e2"]["split"]
    assert split["unit"] == "script"
    assert split["tune_on"] == "dev" and split["score_on"] == "test"
    assert split["dev_fraction"] == 0.3  # owner-confirmed 2026-11-17 (ADR 0011)
    assert split["seed"] == 20261117
    assert split["stratify_by"] == ["rate_band", "voice"]
    assert split["print_counts"] is True
    assert _load("evaluation.yaml")["e2"]["report_pause_position"] is True


def test_filler_repetition_pitch_values_match_adr_0012():
    sp = _load("default.yaml")["speech"]
    assert sp["pitch"] == {
        "fmin_hz": 65.0,
        "fmax_hz": 400.0,
        "frame_length": 1024,
        "hop_length": 160,
    }
    assert sp["fillers"]["confidence_cap_without_verbatim"] == {"en": "medium", "ar": "low"}
    assert sp["fillers"]["acoustic"]["min_duration_s"] == 0.15
    assert sp["fillers"]["acoustic"]["min_voiced_fraction"] == 0.7
    assert sp["fillers"]["acoustic"]["max_pitch_sd_semitones"] == 1.0
    assert sp["repetitions"] == {"max_ngram": 3, "window_s": 2.0}
    asr = _load("models.yaml")["asr"]
    assert asr["whisper_small"]["verbatim"] is False and asr["whisper_turbo"]["verbatim"] is False


def test_lexicon_files_exist_for_both_languages():
    from presenceai.speech.fillers import load_lexicon

    for lang in ("en", "ar"):
        lex = load_lexicon(lang)
        assert lex.hesitations and lex.discourse_markers
    ar = load_lexicon("ar")
    assert ("يعني",) in ar.discourse_markers and ("يعني",) not in ar.hesitations


def test_e3_tuning_and_scoring_data_never_overlap():
    e3 = _load("evaluation.yaml")["e3"]
    dev, test = set(e3["tuning"]["ami_dev_sessions"]), set(e3["scoring"]["ami_test_sessions"])
    assert not dev & test and dev | test == set(e3["ami_split"]["pool"])
    assert {s[:2] for s in dev} == {"ES", "IS", "TS"}  # one per site
    assert "test" not in e3["tuning"]["disfluencyspeech"]
    assert "podcastfillers" not in e3["tuning"]  # research-only: evaluation only
    assert "base_initial_prompt" in e3["conditions"] and e3["conditions"][0] == "base"
    assert "precision" in e3["report"]


def test_e3_ami_split_reproduces_from_its_seed():
    import random

    split = _load("evaluation.yaml")["e3"]["ami_split"]
    rng = random.Random(split["seed"])  # noqa: S311  (reproducible split, not security)
    pool = split["pool"]
    dev = [rng.choice(sorted(s for s in pool if s.startswith(site))) for site in ("ES", "IS", "TS")]
    assert dev == _load("evaluation.yaml")["e3"]["tuning"]["ami_dev_sessions"]
