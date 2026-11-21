"""Evidence verification (AGENTS.md rule 4; P2 exit criterion; ADR 0019).

A transcript quote is verified when the transcript has exactly that text at its offsets; a time
span when it lies within the recording. For rule-based findings a failure is a bug: `verify`
raises `EvidenceError`, and the tests fail. (In the app, P3, one failed finding must not crash
the report: that dimension shows "not measurable" and the failure is logged as counts only.
Language-model claims whose quote is not found are dropped silently, AGENTS.md rule 4.)
"""

from __future__ import annotations

from collections.abc import Iterable

from presenceai.schemas.assessment import Evidence
from presenceai.schemas.speech import Transcript

_EPS = 1e-6


class EvidenceError(ValueError):
    """A rule-based finding quotes something the transcript or recording does not contain."""


def is_verified(e: Evidence, transcript: Transcript, duration_s: float) -> bool:
    if e.transcript is not None:
        q = e.transcript
        if not (0 <= q.start_char < q.end_char <= len(transcript.text)):
            return False
        if transcript.text[q.start_char : q.end_char] != q.text:
            return False
    if e.time is not None:
        t = e.time
        if not (-_EPS <= t.start_s <= t.end_s <= duration_s + _EPS):
            return False
    return True


def verify(evidence: Iterable[Evidence], transcript: Transcript, duration_s: float) -> None:
    """Raise EvidenceError when any evidence item is not verified. Counts only, no text."""
    items = list(evidence)
    failed = sum(not is_verified(e, transcript, duration_s) for e in items)
    if not items or failed:
        raise EvidenceError(f"{failed} of {len(items)} evidence items not verified")
