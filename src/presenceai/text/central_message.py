"""Central message (rubric dimension, ADR 0017 definitions).

- statements: sentences matching the thesis-cue lexicon, anywhere;
- in_opening: the thesis sentence (the first thesis-cue sentence in the opening) exists;
- in_closing: a closing sentence other than the thesis sentence contains a key phrase;
- key_phrase_recurrences: body sentences that contain a key phrase.

A key phrase is the lemma of a noun-phrase head (NOUN or PROPN, not a stop word) of the thesis
sentence; matching is by lemma, not exact string and not meaning.
"""

from __future__ import annotations

from presenceai.lang import text
from presenceai.schemas.assessment import Measurement
from presenceai.text.document import Document, Sentence, match
from presenceai.text.measure import NOMINAL, measurement

KEYS = (
    "text.central_message.statements",
    "text.central_message.in_opening",
    "text.central_message.in_closing",
    "text.central_message.key_phrase_recurrences",
)


def key_phrases(thesis: Sentence | None) -> frozenset[str]:
    if thesis is None:
        return frozenset()
    return frozenset(
        t.lemma.casefold()
        for t in thesis.tokens
        if t.chunk_head and t.pos in NOMINAL and not t.is_stop
    )


def has_key(s: Sentence, keys: frozenset[str]) -> bool:
    return any(t.pos in NOMINAL and t.lemma.casefold() in keys for t in s.tokens)


def measure(doc: Document, values: dict[str, float]) -> dict[str, Measurement]:
    opening, closing = values["opening_share_pct"], values["closing_share_pct"]
    statements = [s for s in doc.sentences if match(s.text, "thesis")]
    thesis = next((s for s in statements if doc.in_opening(s)), None)
    keys = key_phrases(thesis)
    others = [s for s in doc.sentences if s is not thesis and has_key(s, keys)]
    in_closing = [s for s in others if doc.in_closing(s)]
    recurrences = [s for s in others if doc.in_body(s)]
    return {
        KEYS[0]: measurement(
            doc, len(statements), "sentences", text("en", "text.central_message.statements"),
            (doc.span(s) for s in statements), "all",
        ),
        KEYS[1]: measurement(
            doc, thesis is not None, "yes/no",
            text("en", "text.central_message.in_opening", share=opening),
            [doc.span(thesis)] if thesis else [], "opening",
        ),
        KEYS[2]: measurement(
            doc, bool(in_closing), "yes/no",
            text("en", "text.central_message.in_closing", share=closing),
            (doc.span(s) for s in in_closing), "closing",
        ),
        KEYS[3]: measurement(
            doc, len(recurrences), "sentences",
            text("en", "text.central_message.recurrences", opening=opening, closing=closing),
            (doc.span(s) for s in recurrences), "body",
        ),
    }  # fmt: skip
