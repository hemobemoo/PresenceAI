"""Guardrails on generated text: block emotion, mood and trait inference (FINAL_11 §2).

Also checks Arabic evidence quotes for Latin-script text that is not in the transcript
(`latin_outside_transcript`, ADR 0003). Code-switching to English terms is allowed in the
coach's own Arabic wording, never inside a quote.

P0 provides the banned-term lists (``lang/<lang>/banned_terms.yaml``) and
`contains_banned_term`. Regeneration and template fallback come in P7.

Matching is token based, so "happy-path" does not match "happy". Arabic text is
normalised (diacritics, tatweel, alef/ya/ta-marbuta variants) and common clitic
prefixes and suffixes are stripped from each text token before comparison.
"""

from __future__ import annotations

import re
from functools import cache

from presenceai.lang import LANGUAGES, Language, load_pack_file

_LATIN_TOKEN = re.compile(r"[A-Za-z]+")
_EN_TOKEN = re.compile(r"[a-z]+(?:['-][a-z]+)*")
_AR_TOKEN = re.compile(r"[ء-ي]+")
_AR_DIACRITICS = re.compile(r"[ً-ْٰـ]")  # harakat, superscript alef, tatweel
_AR_CHAR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})
_AR_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك")
_AR_SUFFIXES = ("ين", "ون", "ات", "ها", "هم", "كم", "ا", "ه", "ك", "ي")
_AR_MIN_STEM = 2


def _normalise_ar(text: str) -> str:
    return _AR_DIACRITICS.sub("", text).translate(_AR_CHAR_MAP)


def _tokens(text: str, lang: Language) -> list[str]:
    if lang == "en":
        return _EN_TOKEN.findall(text.casefold())
    return _AR_TOKEN.findall(_normalise_ar(text))


def _strip_prefixes(token: str) -> set[str]:
    out = {token}
    for _ in range(2):  # at most two stacked clitics, e.g. و + ب
        for form in list(out):
            for prefix in _AR_PREFIXES:
                if form.startswith(prefix) and len(form) - len(prefix) >= _AR_MIN_STEM:
                    out.add(form[len(prefix) :])
    return out


def _candidates(token: str, lang: Language) -> frozenset[str]:
    if lang == "en":
        return frozenset({token})
    forms = _strip_prefixes(token)
    for form in list(forms):
        for suffix in _AR_SUFFIXES:
            if form.endswith(suffix) and len(form) - len(suffix) >= _AR_MIN_STEM:
                forms.add(form[: -len(suffix)])
    return frozenset(forms)


def _check_lang(lang: str) -> Language:
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language: {lang!r}")
    return lang  # type: ignore[return-value]


@cache
def banned_terms(lang: Language) -> tuple[str, ...]:
    """All banned terms for a language, flattened across categories."""
    data = load_pack_file(_check_lang(lang), "banned_terms.yaml")
    terms: list[str] = []
    for group in data["terms"].values():
        terms.extend(str(t) for t in group)
    return tuple(terms)


def _seqs(terms: tuple[str, ...], lang: Language) -> tuple[tuple[str, ...], ...]:
    seqs = {tuple(_tokens(term, lang)) for term in terms}
    seqs.discard(())
    return tuple(sorted(seqs))


@cache
def _term_token_seqs(lang: Language) -> tuple[tuple[str, ...], ...]:
    return _seqs(banned_terms(lang), lang)


def _contains(text: str, lang: Language, seqs: tuple[tuple[str, ...], ...]) -> bool:
    cands = [_candidates(tok, lang) for tok in _tokens(text, lang)]
    for seq in seqs:
        n = len(seq)
        for i in range(len(cands) - n + 1):
            if all(seq[j] in cands[i + j] for j in range(n)):
                return True
    return False


def contains_banned_term(text: str, lang: Language) -> bool:
    """True if `text` contains any banned emotion, mood or trait term for `lang`."""
    lang = _check_lang(lang)
    return _contains(text, lang, _term_token_seqs(lang))


@cache
def verdict_terms(lang: Language) -> tuple[str, ...]:
    """Verdict words about the voice ("monotone", "رتيب", ...) that no report may use (ADR 0013)."""
    data = load_pack_file(_check_lang(lang), "verdict_terms.yaml")
    return tuple(str(t) for t in data["terms"])


def _verdict_tokens(text: str, lang: Language) -> list[str]:
    """Whole words, before Arabic letter normalisation (owner, 2026-11-17).

    Only diacritics and tatweel are removed; alef, ya and ta-marbuta forms are kept, so «آلي»
    ("robotic") does not match «إلى» ("to"). The lists name each inflected form themselves.
    """
    if lang == "en":
        return _EN_TOKEN.findall(text.casefold())
    return _AR_TOKEN.findall(_AR_DIACRITICS.sub("", text))


@cache
def _verdict_seqs(lang: Language) -> tuple[tuple[str, ...], ...]:
    seqs = {tuple(_verdict_tokens(term, lang)) for term in verdict_terms(lang)}
    seqs.discard(())
    return tuple(sorted(seqs))


def contains_verdict_term(text: str, lang: Language) -> bool:
    """True if `text` contains a verdict word for `lang` as a whole word. In Arabic an attached
    article or preposition (ال، ب، و، ...) is allowed in front; no suffix is stripped."""
    lang = _check_lang(lang)
    words = _verdict_tokens(text, lang)
    cands = [_strip_prefixes(w) if lang == "ar" else {w} for w in words]
    for seq in _verdict_seqs(lang):
        n = len(seq)
        for i in range(len(cands) - n + 1):
            if all(seq[j] in cands[i + j] for j in range(n)):
                return True
    return False


def latin_outside_transcript(quote: str, transcript: str) -> tuple[str, ...]:
    """Latin-script words in an Arabic `quote` that do not occur in `transcript` (ignoring case).

    A non-empty result means the quote was not copied from the transcript, for example a
    model mixing English fragments into an Arabic quote. Empty means no such text.
    """
    present = {t.casefold() for t in _LATIN_TOKEN.findall(transcript)}
    return tuple(t for t in _LATIN_TOKEN.findall(quote) if t.casefold() not in present)
