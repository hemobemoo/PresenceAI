"""Token normalisation shared by filler and repetition matching (ADR 0012).

`normalize_token`: case-folded; punctuation and symbols stripped from both ends (inner
apostrophes and decimal points stay). Arabic additionally: diacritics (harakat, superscript alef)
and tatweel removed, alef forms أ إ آ ٱ unified to ا, and runs of the same Arabic letter
collapsed (elongation: "طيييب" -> "طيب").

`filler_key`: `normalize_token`, then runs of any repeated letter collapsed, so stretched
hesitations match their list entry ("Ummm" = "um", "ممم" = "مم"). It is used only to compare
against the filler lists, never for repetitions ("too" stays different from "to").
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

_DIACRITICS = re.compile("[ً-ْٰـ]")  # harakat, superscript alef, tatweel
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_ARABIC_RUN = re.compile(r"([ء-ي])\1+")
_ANY_RUN = re.compile(r"(\w)\1+")


def _is_edge_junk(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def _strip_edges(token: str) -> str:
    start, end = 0, len(token)
    while start < end and _is_edge_junk(token[start]):
        start += 1
    while end > start and _is_edge_junk(token[end - 1]):
        end -= 1
    return token[start:end]


def normalize_token(token: str, language: Literal["en", "ar"]) -> str:
    out = _strip_edges(token.casefold())
    if language == "ar":
        out = _DIACRITICS.sub("", out).translate(_ALEF)
        out = _ARABIC_RUN.sub(r"\1", out)
        out = _strip_edges(out)
    return out


def filler_key(token: str, language: Literal["en", "ar"]) -> str:
    return _ANY_RUN.sub(r"\1", normalize_token(token, language))
