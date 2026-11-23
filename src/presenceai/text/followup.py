"""Follow-up answer content in S2 (ADR 0022): relevance and limitation cues.

- **Relevance:** cosine similarity between the question and the answer transcript, from the
  knowledge-base embedding model (multilingual-e5-small, ADR 0020); the question is embedded as
  a "query" and the answer as a "passage", as e5 expects for question-answer retrieval. It is a
  similarity of words, not an understanding of whether the question was answered: an evasive
  answer that reuses the question's words can score high. English only until there are Arabic
  calibration pairs; its thresholds are placeholders until the owner's 20 pairs are scored.
- **Limitation cue:** an explicit limitation phrase from `lang/en/text_cues.yaml: limitation`
  (never a hedging phrase). A lexicon match, not an understanding of meaning.
- **Summary:** the weaker answer decides (owner, 2026-11-23): the lowest relevance and the
  longest latency over the answers; a limitation counts if any answer has one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from presenceai.knowledge.embed import Embedder
from presenceai.schemas.followup import AnswerMeasures, FollowupSummary
from presenceai.text.document import match


def relevance(question: str, answer: str, embedder: Embedder) -> float:
    """Cosine similarity of the question (query) and the answer (passage).

    An empty answer has nothing to match and scores 0.0: it is a measured non-answer, not a
    measure that could not be computed."""
    if not answer.strip():
        return 0.0
    q = embedder.embed([question], "query")[0]
    a = embedder.embed([answer], "passage")[0]
    return float(np.dot(q, a) / max(float(np.linalg.norm(q) * np.linalg.norm(a)), 1e-12))


def limitation_cue(answer: str) -> str | None:
    """The first explicit limitation phrase in the answer, or None."""
    return match(answer, "limitation")


def summarise(answers: Sequence[AnswerMeasures]) -> FollowupSummary:
    """The weaker answer decides; answers whose measure is None are left out of it."""
    if not answers:
        raise ValueError("a follow-up summary needs at least one answer")
    rel = [(a.relevance, a.index) for a in answers if a.relevance is not None]
    lat = [(a.latency_s, a.index) for a in answers if a.latency_s is not None]
    lim = [a.index for a in answers if a.limitation is not None]
    low = min(rel) if rel else None
    slow = max(lat) if lat else None
    return FollowupSummary(
        answers=tuple(answers),
        relevance_min=low[0] if low else None,
        relevance_min_index=low[1] if low else None,
        latency_max_s=slow[0] if slow else None,
        latency_max_index=slow[1] if slow else None,
        limitation_acknowledged=bool(lim),
        limitation_index=lim[0] if lim else None,
    )
