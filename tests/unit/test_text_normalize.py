"""text/normalize.py: one normalisation for filler and repetition matching (ADR 0012)."""

import pytest

from presenceai.text.normalize import filler_key, normalize_token


@pytest.mark.parametrize(
    ("token", "lang", "expected"),
    [
        ("Hello,", "en", "hello"),
        ("WE.", "en", "we"),
        ('"So"', "en", "so"),
        ("don't", "en", "don't"),  # apostrophes inside words stay
        ("too", "en", "too"),  # English letters are not collapsed for repetitions
        ("يَعْنِي", "ar", "يعني"),  # diacritics removed
        ("يعـــني", "ar", "يعني"),  # tatweel removed
        ("أحمد", "ar", "احمد"),  # alef forms unified
        ("إلى", "ar", "الى"),
        ("آه", "ar", "اه"),
        ("طيييب", "ar", "طيب"),  # elongation collapsed
        ("الفكرة،", "ar", "الفكرة"),  # Arabic comma stripped
        ("ماذا؟", "ar", "ماذا"),
    ],
)
def test_normalize_token(token: str, lang: str, expected: str):
    assert normalize_token(token, lang) == expected  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("a", "b"),
    [("Ummm", "um"), ("Hmm.", "hmm"), ("uhh", "uh"), ("errr", "er"), ("ممم", "مم"), ("آآ", "اا")],
)
def test_filler_keys_ignore_letter_stretching(a: str, b: str):
    lang = "ar" if "؀" <= b[0] <= "ۿ" else "en"
    assert filler_key(a, lang) == filler_key(b, lang)  # pyright: ignore[reportArgumentType]


def test_filler_key_keeps_different_words_apart():
    assert filler_key("umbrella", "en") != filler_key("um", "en")
    assert filler_key("likely", "en") != filler_key("like", "en")


def test_punctuation_only_token_is_empty():
    assert normalize_token("...", "en") == ""
    assert normalize_token("،", "ar") == ""
