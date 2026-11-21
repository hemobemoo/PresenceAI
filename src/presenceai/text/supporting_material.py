"""Supporting material (rubric dimension, ADR 0017 definitions).

Types, by pattern rules:
- statistic: a PERCENT, MONEY or QUANTITY entity, or a CARDINAL written with digits; dates
  (years), times and ordinals are not statistics;
- example, analogy, quotation: cue lexicons; quotation also when the sentence has a pair of
  quotation marks.
Counting: one item per sentence and type; items = (sentence, type) pairs; types = distinct types.
"""

from __future__ import annotations

from presenceai.lang import text
from presenceai.schemas.assessment import Measurement
from presenceai.text.document import Document, Sentence, match
from presenceai.text.measure import measurement

KEYS = ("text.support.items", "text.support.types")
TYPES = ("statistic", "example", "analogy", "quotation")
_STAT_ENTS = frozenset({"PERCENT", "MONEY", "QUANTITY"})


def has_statistic(s: Sentence) -> bool:
    return any(
        t.ent in _STAT_ENTS or (t.ent == "CARDINAL" and any(c.isdigit() for c in t.text))
        for t in s.tokens
    )


def has_quotation(s: Sentence) -> bool:
    quoted = s.text.count('"') >= 2 or ("“" in s.text and "”" in s.text)
    return quoted or match(s.text, "quotation") is not None


def types_in(s: Sentence) -> frozenset[str]:
    found = set()
    if has_statistic(s):
        found.add("statistic")
    for kind in ("example", "analogy"):
        if match(s.text, kind):
            found.add(kind)
    if has_quotation(s):
        found.add("quotation")
    return frozenset(found)


def measure(doc: Document) -> dict[str, Measurement]:
    per_sentence = [(s, types_in(s)) for s in doc.sentences]
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
