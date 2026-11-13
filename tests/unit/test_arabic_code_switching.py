"""Arabic code-switching (ADR 0003): English terms allowed in wording, never added to quotes."""

import re

import pytest

from presenceai.coach.guardrails import latin_outside_transcript
from presenceai.lang import LANGUAGES, Language, load_pack_file

TRANSCRIPT = "سأعرض اليوم الـ slides الخاصة بالمشروع، ثم أتحدث عن النتائج."


def test_latin_fragment_not_in_transcript_is_flagged():
    # S4 re-run, dots, ar01: Latin fragments inside an Arabic quote
    quote = "يعني مشروع تطبيق يساعد_sin students على إيجاد قاعات"
    assert latin_outside_transcript(quote, "يعني المشروع تطبيق يساعد الطلاب") == ("sin", "students")


def test_english_term_that_is_in_the_transcript_is_not_flagged():
    assert latin_outside_transcript("الـ Slides الخاصة بالمشروع", TRANSCRIPT) == ()


def test_pure_arabic_quote_is_not_flagged():
    assert latin_outside_transcript("ثم أتحدث عن النتائج", TRANSCRIPT) == ()


@pytest.mark.parametrize("lang", LANGUAGES)
def test_coach_prompt_pack_has_the_same_keys(lang: Language):
    pack = load_pack_file(lang, "coach_prompt.yaml")
    assert set(pack) == {"language_rule", "keep_english_examples", "quotes_follow_transcript"}
    assert pack["language_rule"].strip()
    assert pack["quotes_follow_transcript"] is True


def test_arabic_rule_keeps_english_terms_in_latin_script_and_excludes_quotes():
    pack = load_pack_file("ar", "coach_prompt.yaml")
    rule = pack["language_rule"]
    assert "never write them in Arabic letters" in rule
    assert "does not apply to evidence_quote" in rule
    examples = pack["keep_english_examples"]
    assert {"slides", "feedback", "pace"} <= set(examples)
    assert all(re.fullmatch(r"[A-Za-z&]+", t) for t in examples)
