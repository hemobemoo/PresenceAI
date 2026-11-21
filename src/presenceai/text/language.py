"""Language (rubric dimension, ADR 0017 and ADR 0018).

- undefined_terms: rare terms used without a definition cue in the same or the next sentence,
  each term counted once, at its first use. A candidate term is a NOUN, VERB or ADJ (not a
  proper name, number or stop word, at least 3 letters), or an all-capital acronym of at least
  2 letters. It is rare when its Zipf frequency (wordfreq, English) is below the threshold for
  both its surface form and its lemma; an acronym only when its own frequency is below it, so
  "AI", "TV" or "OK" are not flagged (owner, 2026-11-21). A misrecognised word from Whisper can
  appear as a rare term (LIMITATIONS).
- mean_sentence_words: words per sentence; missing (None) when the punctuation is unreliable,
  so only this condition of the rubric is dropped (ADR 0017).
"""

from __future__ import annotations

from collections.abc import Callable

from wordfreq import zipf_frequency

from presenceai.lang import text
from presenceai.schemas.assessment import Measurement
from presenceai.text.document import Document, match
from presenceai.text.measure import measurement
from presenceai.text.nlp import Tok

KEYS = ("text.language.undefined_terms", "text.language.mean_sentence_words")
TERM_POS = frozenset({"NOUN", "VERB", "ADJ"})
Zipf = Callable[[str], float]


def _zipf_en(word: str) -> float:
    return zipf_frequency(word, "en")


def is_acronym(t: Tok) -> bool:
    return len(t.text) >= 2 and t.text.isalpha() and t.text.isupper()


def is_rare_term(t: Tok, threshold: float, zipf: Zipf = _zipf_en) -> bool:
    if is_acronym(t):
        return zipf(t.text.casefold()) < threshold
    if t.pos not in TERM_POS or t.is_stop or t.like_num or not t.text.isalpha():
        return False
    if len(t.text) < 3:
        return False
    return max(zipf(t.text.casefold()), zipf(t.lemma.casefold())) < threshold


def measure(
    doc: Document, values: dict[str, float], zipf: Zipf = _zipf_en
) -> dict[str, Measurement | None]:
    threshold = values["lang_rare_zipf"]
    seen: set[str] = set()
    undefined = []
    sentences = doc.sentences
    for i, s in enumerate(sentences):
        defined = bool(match(s.text, "definition")) or (
            i + 1 < len(sentences) and bool(match(sentences[i + 1].text, "definition"))
        )
        for t in s.tokens:
            key = t.lemma.casefold()
            if key in seen or not is_rare_term(t, threshold, zipf):
                continue
            seen.add(key)
            if not defined:
                undefined.append(s.tok_span(t, doc.transcript))
    mean = None
    if doc.punctuation_reliable and sentences:
        mean = measurement(
            doc, round(doc.n_words / len(sentences), 2), "words/sentence",
            text("en", "text.language.mean_sentence_words"), [], "all",
        )  # fmt: skip
    return {
        KEYS[0]: measurement(
            doc, len(undefined), "terms",
            text("en", "text.language.undefined_terms", zipf=threshold), undefined, "all",
        ),
        KEYS[1]: mean,
    }  # fmt: skip
