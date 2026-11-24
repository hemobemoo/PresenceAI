"""The report page (P3-d): one card per rubric dimension.

Each card shows the level (or "not measurable"), the **N of M** badge (always, ADR 0019), the
**capped** badge when the partial cap applied, the confidence, the finding in observable words
and its evidence (quotes with times). When the report text cannot be decrypted (a lost or wrong
MEDIA_KEY, ADR 0023 item 12) the cards come from the text-free summary: levels and badges stay,
the words are replaced by a note.

The user's words are escaped before they go into Markdown, so a transcript can never add links,
images or formatting to the page.
"""

from __future__ import annotations

import re

from presenceai.lang import Language, ui_text
from presenceai.rubric import loader
from presenceai.schemas.assessment import Evidence
from presenceai.schemas.storage import DimensionSummary
from presenceai.storage.reports import ReportView

_MD = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~])")


def escape(text: str) -> str:
    """Markdown-safe text: every formatting character is escaped."""
    return _MD.sub(r"\\\1", text)


def clock(seconds: float) -> str:
    s = max(0, round(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def _names(scenario: str, lang: Language) -> dict[str, str]:
    try:
        rubric = loader.load_rubric(scenario)
    except (OSError, ValueError):
        return {}
    return {d.id: (d.name_ar if lang == "ar" else d.name) for d in rubric.dimensions}


def _badges(d: DimensionSummary, lang: Language) -> str:
    parts: list[str] = []
    if d.level is None:
        key = "report.failed" if d.failed else "report.not_measurable"
        parts.append(f"**{ui_text(lang, key)}**")
    else:
        parts.append(f"**{ui_text(lang, 'report.level', level=d.level)}**")
    parts.append(f"`{ui_text(lang, 'report.based_on', measured=d.measured, total=d.total)}`")
    if d.capped_from is not None and d.level is not None:
        parts.append(f"`{ui_text(lang, 'report.capped', level=d.level, reached=d.capped_from)}`")
    if d.confidence is not None:
        conf = ui_text(lang, f"report.confidences.{d.confidence}")
        parts.append(ui_text(lang, "report.confidence", confidence=conf))
    return " · ".join(parts)


def _evidence(e: Evidence, lang: Language) -> str:
    bits: list[str] = []
    if e.answer is not None:
        bits.append(ui_text(lang, "report.answer", n=e.answer + 1))
    if e.transcript is not None:
        bits.append(f"“{escape(e.transcript.text)}”")
    if e.time is not None:
        bits.append(f"({clock(e.time.start_s)}–{clock(e.time.end_s)})")
    return "- " + " ".join(bits)


def render(view: ReportView, lang: Language, scenario: str, *, max_evidence: int = 5) -> str:
    """The report as Markdown, in `lang`."""
    names = _names(scenario, lang)
    out: list[str] = [f"## {ui_text(lang, 'report.heading')}"]
    if view.text_note:
        out.append(f"> {escape(view.text_note)}")
    findings = {f.dimension: f for f in view.report.assessment.findings} if view.report else {}
    order = {dim: i for i, dim in enumerate(names)}  # the rubric's order
    dims = sorted(view.summary.dimensions, key=lambda d: order.get(d.dimension, len(order)))
    for d in dims:
        out.append(f"### {escape(names.get(d.dimension, d.dimension))}")
        out.append(_badges(d, lang))
        f = findings.get(d.dimension)
        if f is None:
            continue
        out.append(escape(f.statement))
        if f.evidence:
            out.append(f"*{ui_text(lang, 'report.evidence')}*")
            out.extend(_evidence(e, lang) for e in f.evidence[:max_evidence])
            if len(f.evidence) > max_evidence:
                more = len(f.evidence) - max_evidence
                out.append(f"- {ui_text(lang, 'report.more_evidence', n=more)}")
    return "\n\n".join(out)


def transcript_text(view: ReportView) -> str:
    """The transcript for its own read-only box (plain text, never Markdown)."""
    return view.report.speech.transcript.text if view.report else ""
