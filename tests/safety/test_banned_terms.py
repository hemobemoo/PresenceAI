"""Banned inference terms in EN and AR (FINAL_11 §2) and the `contains_banned_term` helper."""

from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from presenceai.coach.guardrails import banned_terms, contains_banned_term
from presenceai.lang import LANGUAGES, Language, load_ui_strings


@pytest.mark.parametrize("lang", LANGUAGES)
def test_list_exists_and_is_non_empty(lang: Language):
    assert len(banned_terms(lang)) >= 30


@pytest.mark.parametrize(
    "text",
    [
        "You seemed nervous during the opening.",
        "NERVOUS",
        "She sounds unsure about the result.",
        "Your confident tone helped.",
        "A lack of confidence showed.",
        "The speaker looked happy.",
        "Your personality came through.",
        "high self-confidence",
    ],
)
def test_english_positives(text: str):
    assert contains_banned_term(text, "en")


@pytest.mark.parametrize(
    "text",
    [
        "Speech rate rose from 148 to 191 wpm during the follow-up (01:12-01:40).",
        "Visible smile during 8% of the opening.",
        "Confidence: high.",
        "Measurement confidence is medium because the audio was noisy.",
        "Gaze away from the camera for 8 s.",
        "The scenario covers happy-path testing.",  # 'happy-path' is one token
        "",
    ],
)
def test_english_negatives(text: str):
    assert not contains_banned_term(text, "en")


@pytest.mark.parametrize(
    "text",
    [
        "بدوت متوترا في البداية",
        "بدوتَ مُتَوَتِّرًا",  # with diacritics
        "كان المتحدث سعيدا",
        "والقلق واضح",
        "لديك الثقة بالنفس",
        "يبدو غير متأكد",
    ],
)
def test_arabic_positives(text: str):
    assert contains_banned_term(text, "ar")


@pytest.mark.parametrize(
    "text",
    [
        "ارتفع معدل الكلام من 148 إلى 191 كلمة في الدقيقة",
        "ابتسامة ظاهرة خلال 8% من الافتتاحية",
        "إرسال نص التفريغ إلى نموذج ذكاء اصطناعي مستضاف",
        "مستوى الثقة: مرتفع",
        "",
    ],
)
def test_arabic_negatives(text: str):
    assert not contains_banned_term(text, "ar")


@pytest.mark.parametrize("lang", LANGUAGES)
def test_every_listed_term_matches_itself(lang: Language):
    for term in banned_terms(lang):
        assert contains_banned_term(term, lang), term


def _strings(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for value in node.values():  # pyright: ignore[reportUnknownVariableType]
            yield from _strings(value)
    else:
        yield str(node)


@pytest.mark.parametrize("lang", LANGUAGES)
def test_ui_strings_contain_no_banned_term(lang: Language):
    for text in _strings(load_ui_strings(lang)):
        assert not contains_banned_term(text, lang), text


@given(st.text(alphabet="0123456789 .,:;%()-/\n\t"))
def test_digits_and_punctuation_never_match(text: str):
    assert not contains_banned_term(text, "en")
    assert not contains_banned_term(text, "ar")


def test_unknown_language_rejected():
    with pytest.raises(ValueError):
        contains_banned_term("hello", "fr")  # type: ignore[arg-type]
