"""Report texts live in lang/<lang>/report.yaml (AGENTS.md: no hard-coded user-facing text).

The two languages must have the same keys and the same placeholder names, and speech code may
not carry its own wording (ADR 0016).
"""

import ast
import re
from pathlib import Path

import pytest

from presenceai import lang as lang_mod
from presenceai.coach.guardrails import contains_banned_term, contains_verdict_term
from presenceai.lang import LANGUAGES, Language, placeholders, report_texts, text

SPEECH_DIR = Path(lang_mod.__file__).parents[1] / "speech"
ARABIC = re.compile(r"[؀-ۿ]")


def test_same_keys_in_both_languages():
    assert set(report_texts("en")) == set(report_texts("ar"))


@pytest.mark.parametrize("key", sorted(report_texts("en")))
def test_same_placeholders_in_both_languages(key: str):
    assert placeholders(report_texts("en")[key]) == placeholders(report_texts("ar")[key]), key


@pytest.mark.parametrize("lang", LANGUAGES)
def test_no_empty_texts_and_no_verdict_or_banned_words(lang: Language):
    for key, value in report_texts(lang).items():
        assert value.strip(), key
        assert not contains_verdict_term(value, lang), f"{lang}:{key}"
        assert not contains_banned_term(value, lang), f"{lang}:{key}"


def test_arabic_texts_are_arabic():
    for key, value in report_texts("ar").items():
        if key.startswith("positions.") or key == "units.dB":
            continue
        if key.startswith("units.") and value == "dB":
            continue
        assert ARABIC.search(value), key


def test_text_fills_placeholders():
    assert text("en", "common.too_noisy", snr=7.4) == (
        "not measurable in this recording: too noisy (7 dB signal-to-noise)"
    )
    assert "7 dB" in text("ar", "common.too_noisy", snr=7.4)


def test_text_rejects_missing_or_extra_values():
    with pytest.raises(ValueError, match="snr"):
        text("en", "common.too_noisy")
    with pytest.raises(ValueError):
        text("en", "common.no_speech", snr=1.0)
    with pytest.raises(KeyError):
        text("en", "common.does_not_exist")


def test_unit_labels_and_language_names():
    assert lang_mod.unit_label("ar", "words/min") == "كلمة/دقيقة"
    assert lang_mod.language_name("en", "ar") == "Arabic"
    assert lang_mod.language_name("ar", "fr") == "لغة أخرى (الرمز fr)"


def _string_constants(path: Path) -> list[str]:
    """String literals in a module, docstrings and f-string parts included."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]


@pytest.mark.parametrize(
    "module",
    ["pauses", "rate", "fillers", "repetitions", "loudness", "pitch_variation", "analyze"],
)
def test_speech_modules_have_no_report_wording(module: str):
    """Every English report text must come from lang/; a sentence-like literal in speech code
    (three or more words) means someone wrote wording in code again."""
    texts = report_texts("en")
    units = {k.removeprefix("units.") for k in texts if k.startswith("units.")}  # stable codes
    english = {v for k, v in texts.items() if len(v.split()) >= 2 and not k.startswith("units.")}
    for value in _string_constants(SPEECH_DIR / f"{module}.py"):
        if value in units:
            continue
        words = value.split()
        assert value not in english, f"{module}.py repeats a report text: {value!r}"
        assert len(words) < 3, (
            f"{module}.py: {value!r} looks like report wording; put it in lang/<lang>/report.yaml"
        )
