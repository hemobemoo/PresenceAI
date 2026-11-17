"""synth/scripts.py: C1 script parsing and the automated checks (ADR 0014 §2)."""

import json
from collections import Counter
from pathlib import Path

import pytest

from presenceai.evalconfig import REPO_ROOT, load_evaluation
from presenceai.synth import scripts

CFG = load_evaluation().c1
SCRIPTS = REPO_ROOT / CFG.scripts
LEXICON_CHECK = SCRIPTS.parent / "lexicon_check.json"
GOOD = (
    "Good morning, everyone. Today I want to share | a simple idea about planning. "
    "Small steps help us | finish big tasks."
)


def test_phrases_end_at_commas_sentence_marks_and_bars():
    s = scripts.parse_script("s01", "af_heart", "T", GOOD)
    assert [p.text for p in s.phrases] == [
        "Good morning,",
        "everyone.",
        "Today I want to share",
        "a simple idea about planning.",
        "Small steps help us",
        "finish big tasks.",
    ]
    assert [p.boundary_after for p in s.phrases] == [
        "clause_boundary",
        "sentence_boundary",
        "mid_sentence",
        "sentence_boundary",
        "mid_sentence",
        "end",
    ]
    assert s.words == 20


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("We met at 9 today.", "characters"),
        ("We don't stop.", "characters"),
        ("A well-known plan.", "characters"),
        ("Please read the plan.", "heteronym"),
        ("We start. | Then we go.", "bar"),
        (
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
            "fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo.",
            "sentence mark",
        ),
    ],
)
def test_checks_catch_bad_text(text: str, problem: str):
    found = scripts.check_text(text, CFG)
    assert any(problem in f for f in found), found


def test_checks_pass_good_text():
    assert scripts.check_text(GOOD, CFG) == []


def test_set_checks_count_scripts_per_voice_and_ids():
    s = scripts.parse_script("s01", "af_heart", "T", GOOD)
    problems = scripts.check_set((s, s), CFG)
    assert any("duplicate" in p for p in problems)
    assert any("scripts for" in p for p in problems)


# --- the committed scripts ---------------------------------------------------------------------


def test_committed_scripts_pass_every_check():
    loaded = scripts.load_scripts(SCRIPTS)
    assert scripts.check_set(loaded, CFG) == []
    assert len(loaded) == len(CFG.voices) * CFG.scripts_per_voice
    assert Counter(s.voice for s in loaded) == dict.fromkeys(CFG.voices, CFG.scripts_per_voice)


def test_committed_scripts_have_no_verdict_or_banned_terms():
    from presenceai.coach.guardrails import contains_banned_term, contains_verdict_term

    for s in scripts.load_scripts(SCRIPTS):
        assert not contains_banned_term(s.text, "en"), s.id
        assert not contains_verdict_term(s.text, "en"), s.id


def test_every_script_word_is_in_kokoros_word_lists():
    """The Colab run looks every word up in misaki's word lists before synthesis and writes the
    result; no word may be missing, so espeak-ng is never needed (ADR 0014 §1)."""
    if not LEXICON_CHECK.exists():
        pytest.skip("resources/c1/lexicon_check.json is written by the first Colab run")
    check = json.loads(LEXICON_CHECK.read_text(encoding="utf-8"))
    needed = scripts.words_by_accent(scripts.load_scripts(SCRIPTS), CFG)
    for accent, words in needed.items():
        rated = check["words"][accent]
        missing = sorted(w for w in words if not rated.get(w))
        assert not missing, f"{accent}: {missing[:20]}"


def test_bad_scripts_file_is_rejected(tmp_path: Path):
    p = tmp_path / "s.yaml"
    p.write_text("scripts:\n  - {id: s01, voice: af_heart, title: T}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text"):
        scripts.load_scripts(p)
