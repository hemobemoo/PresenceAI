"""Saving and opening assessment reports (owner, 2026-11-24, ADR 0023 item 12).

- The full `AssessmentReport` holds the user's words (transcript, evidence quotes, finding
  text, not-measurable reasons), so it is stored **encrypted with MEDIA_KEY**
  (`sessions/<id>/assessment.json.enc`).
- A **text-free summary** goes to SQLite in the clear: per dimension the level, capped_from,
  score, confidence, N of M and whether it failed; per measure its numeric value, unit code and
  confidence. The progress page and the engine (P5) read only this summary.
- If the key is lost or wrong, the report opens with its text unavailable and the summary still
  loads, so progress survives.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from presenceai.lang import text
from presenceai.schemas.assessment import Measurement
from presenceai.schemas.report import AssessmentReport
from presenceai.schemas.storage import DimensionSummary, MeasureValue, SessionSummary
from presenceai.storage.db import Database
from presenceai.storage.media import MediaStore, MediaUnreadable


class ReportView(BaseModel):
    """What the report page shows: the summary always, the full report when it can be read."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    summary: SessionSummary
    report: AssessmentReport | None = None
    text_note: str | None = None  # set when the report text cannot be shown


def summarise(session_id: str, report: AssessmentReport) -> SessionSummary:
    """Numbers and codes only; never a transcript, quote, finding text or reason."""
    a = report.assessment
    failed = set(a.failed_dimensions)
    dimensions = tuple(
        DimensionSummary(
            dimension=r.dimension,
            level=r.level,
            capped_from=r.capped_from,
            score=r.score,
            confidence=r.confidence.value if r.confidence else None,
            measured=f.measured,
            total=f.total,
            failed=r.dimension in failed,
        )
        for r, f in zip(a.dimensions, a.findings, strict=True)
    )
    found: dict[str, MeasureValue] = {}
    sources: Iterable[Mapping[str, Measurement]] = (
        report.speech.measurements,
        report.text.measurements,
        report.followups.measurements if report.followups else {},
    )
    for measurements in sources:
        for key, m in measurements.items():
            found[key] = MeasureValue(
                measure=key, value=m.value, unit=m.unit, confidence=m.confidence.value
            )
    for r in a.dimensions:  # measures the rubric derives (mid-sentence pauses, overrun)
        for c in r.conditions:
            if c.value is not None and c.measure not in found:
                found[c.measure] = MeasureValue(measure=c.measure, value=c.value)
    measures = tuple(found[k] for k in sorted(found))
    return SessionSummary(session_id=session_id, dimensions=dimensions, measures=measures)


def save(db: Database, store: MediaStore, session_id: str, report: AssessmentReport) -> None:
    """Encrypt the report, store its text-free summary, and mark the session analysed."""
    store.write_assessment(session_id, report.model_dump_json().encode("utf-8"))
    db.save_summary(summarise(session_id, report))
    db.set_status(session_id, "analysed")


def open_report(
    db: Database, store: MediaStore, session_id: str, *, lang: Literal["en", "ar"]
) -> ReportView:
    """Never raises for a lost or wrong key: the text is then reported unavailable."""
    summary = db.summary(session_id)
    unavailable = ReportView(summary=summary, text_note=text(lang, "storage.text_unavailable"))
    if not store.has_assessment(session_id):
        return unavailable
    try:
        raw = store.read_assessment(session_id)
    except MediaUnreadable:
        return unavailable
    return ReportView(summary=summary, report=AssessmentReport.model_validate_json(raw))
