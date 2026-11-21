"""Building text `Measurement`s with evidence (ADR 0018).

Every measurement quotes transcript spans exactly. A count of zero quotes the part of the
transcript that was searched (opening, closing, body or the whole talk), so even "none found"
has evidence. Confidence is at most medium: the detectors are rule-based and their accuracy is
only known after E4. It is low when the punctuation is unreliable (ADR 0011), because sentences
come from the punctuation.
"""

from __future__ import annotations

from collections.abc import Iterable

from presenceai.schemas.assessment import Confidence, Evidence, Measurement, TranscriptSpan
from presenceai.text.document import Document

NOMINAL = frozenset({"NOUN", "PROPN"})


def confidence(doc: Document) -> Confidence:
    return Confidence.MEDIUM if doc.punctuation_reliable else Confidence.LOW


def measurement(
    doc: Document,
    value: float,
    unit: str,
    method: str,
    spans: Iterable[TranscriptSpan],
    searched: str,
) -> Measurement:
    evidence = tuple(Evidence(transcript=s) for s in spans) or (
        Evidence(transcript=doc.region(searched)),
    )
    return Measurement(
        value=float(value),
        unit=unit,
        confidence=confidence(doc),
        method=method,
        evidence=evidence,
        language="en",
        live=False,
    )
