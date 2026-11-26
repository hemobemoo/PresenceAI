"""Central message: a possible main point, shown and never scored (ADR 0025).

No rule-based or embedding method reached the pre-set bar on the owner's labelled C2 talks
(ADR 0025), so the central-message dimension is information-only until the P7 judge, and the v1
thesis-cue measures are no longer computed. What runs is method A as the owner accepted it for
display:

- candidates: sentences that start in the first `window_pct` % of the words (configs/rubric.yaml
  `possible_main_point`) and are not questions;
- the first candidate that contains a phrase of `main_point_purpose` (lang/en/text_cues.yaml),
  as whole words after normalisation, is the possible main point.

The measurement is 1 (a sentence found, its span and time as evidence) or 0 (the searched
opening as evidence, never shown as "no main point found"). Confidence is always low.
"""

from __future__ import annotations

from presenceai.lang import text
from presenceai.schemas.assessment import Confidence, Evidence, Measurement, TimeSpan
from presenceai.text.document import Document, Sentence, find

KEY = "text.central_message.possible_main_point"
KEYS = (KEY,)
CUE = "main_point_purpose"


def window_end(doc: Document, window_pct: float) -> int:
    """First word index after the opening window (at least one word)."""
    return max(1, round(doc.n_words * window_pct / 100)) if doc.n_words else 0


def possible_main_point(doc: Document, window_pct: float) -> Sentence | None:
    end = window_end(doc, window_pct)
    for s in doc.sentences:
        if s.first_word >= end:
            break
        if s.text.rstrip().endswith("?"):
            continue
        if find(s.text, CUE):
            return s
    return None


def measure(doc: Document, window_pct: float) -> dict[str, Measurement]:
    s = possible_main_point(doc, window_pct)
    words = doc.transcript.words
    if s is not None:
        evidence = (
            Evidence(
                transcript=doc.span(s),
                time=TimeSpan(start_s=words[s.first_word].start_s, end_s=words[s.last_word].end_s),
            ),
        )
    else:
        evidence = (Evidence(transcript=doc.transcript.span(0, window_end(doc, window_pct))),)
    return {
        KEY: Measurement(
            value=1.0 if s is not None else 0.0,
            unit="yes/no",
            confidence=Confidence.LOW,
            method=text("en", "text.central_message.possible_main_point", share=window_pct),
            evidence=evidence,
            language="en",
            live=False,
        )
    }
