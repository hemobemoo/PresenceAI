"""Supporting material (rubric dimension, ADR 0017 definitions).

Types, by pattern rules:
- statistic: a PERCENT, MONEY or QUANTITY entity, or a CARDINAL written with digits; dates
  (years), times and ordinals are not statistics;
- example, analogy, quotation: cue lexicons; quotation also when the sentence has a pair of
  quotation marks. After the owner's review: "like a/an" is not an analogy after "would" ("I'd
  like a minute") or when "like" is a hesitation filler found by the speech detector; "said
  that" is not a quotation after "i" or "we".
Counting: one item per sentence and type; items = (sentence, type) pairs; types = distinct types.
"""

from __future__ import annotations

from presenceai.lang import text
from presenceai.schemas.assessment import Measurement
from presenceai.text.document import Document, Sentence, find, match
from presenceai.text.measure import measurement

KEYS = ("text.support.items", "text.support.types")
TYPES = ("statistic", "example", "analogy", "quotation")
_STAT_ENTS = frozenset({"PERCENT", "MONEY", "QUANTITY"})


def has_statistic(s: Sentence) -> bool:
    return any(
        t.ent in _STAT_ENTS or (t.ent == "CARDINAL" and any(c.isdigit() for c in t.text))
        for t in s.tokens
    )


LIKE = ("like a", "like an")
SAID_THAT = "said that"
FIRST_PERSON = frozenset({"i", "we"})


def _phrase_at(aligned: list[tuple[str, int]], phrase: str) -> list[int]:
    target = phrase.split()
    words = [w for w, _ in aligned]
    return [i for i in range(len(words) - len(target) + 1) if words[i : i + len(target)] == target]


def has_analogy(doc: Document, s: Sentence) -> bool:
    if any(p not in LIKE for p, _ in find(s.text, "analogy")):
        return True
    aligned = doc.aligned(s)
    for phrase in LIKE:
        for i in _phrase_at(aligned, phrase):
            after_would = i > 0 and aligned[i - 1][0] == "would"
            if not after_would and aligned[i][1] not in doc.hesitation_words:
                return True
    return False


def has_quotation(doc: Document, s: Sentence) -> bool:
    if s.text.count('"') >= 2 or ("“" in s.text and "”" in s.text):
        return True
    if any(p != SAID_THAT for p, _ in find(s.text, "quotation")):
        return True
    aligned = doc.aligned(s)
    return any(
        i == 0 or aligned[i - 1][0] not in FIRST_PERSON for i in _phrase_at(aligned, SAID_THAT)
    )


def types_in(doc: Document, s: Sentence) -> frozenset[str]:
    found = set()
    if has_statistic(s):
        found.add("statistic")
    if match(s.text, "example"):
        found.add("example")
    if has_analogy(doc, s):
        found.add("analogy")
    if has_quotation(doc, s):
        found.add("quotation")
    return frozenset(found)


def measure(doc: Document) -> dict[str, Measurement]:
    per_sentence = [(s, types_in(doc, s)) for s in doc.sentences]
    with_items = [s for s, kinds in per_sentence if kinds]
    items = sum(len(kinds) for _, kinds in per_sentence)
    kinds = set().union(*(k for _, k in per_sentence)) if per_sentence else set()
    spans = [doc.span(s) for s in with_items]
    return {
        KEYS[0]: measurement(doc, items, "items", text("en", "text.support.items"), spans, "all"),
        KEYS[1]: measurement(
            doc, len(kinds), "types", text("en", "text.support.types"), spans, "all"
        ),
    }
