"""The report page (P3-d): one card per rubric dimension.

Each card shows the level (or "not measurable"), the **N of M** badge (always, ADR 0019), the
**capped** badge when the partial cap applied, the confidence, the finding in observable words
and its evidence (quotes with times). When the report text cannot be decrypted (a lost or wrong
MEDIA_KEY, ADR 0023 item 12) the cards come from the text-free summary: levels and badges stay,
the words are replaced by a note.

The user's words are escaped before they go into Markdown, so a transcript can never add links,
images or formatting to the page.

After the owner's real S1 runs (2026-11-25): a long quote shows its first `QUOTE_WORDS` words and
"…" (the full quote stays in the report data and the export); each card names the measures that
were not measured and why, and those left out because the camera was off.

ADR 0025 (owner, 2026-11-26): a line above the cards counts the scored dimensions ("Levels based
on N of M dimensions") and names the information-only ones; an information-only card has an
"Information only, not scored" badge instead of a level and N of M measures. The level-1
central-message hint is gone (the detector cannot support "no main point found"); the general
tip is on the practice page.
"""

from __future__ import annotations

import re

from presenceai.lang import Language, text, ui_text
from presenceai.rubric import loader
from presenceai.schemas.assessment import Evidence
from presenceai.schemas.rubric import DimensionResult
from presenceai.schemas.storage import DimensionSummary, SessionSummary
from presenceai.storage.reports import ReportView

_MD = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~])")
QUOTE_WORDS = 25  # longer quotes are shortened on the card only


def escape(text: str) -> str:
    """Markdown-safe text: every formatting character is escaped."""
    return _MD.sub(r"\\\1", text)


def shorten(quote: str, words: int = QUOTE_WORDS) -> str:
    """The first `words` words and "…" for a long quote; a short quote unchanged."""
    parts = quote.split()
    return quote if len(parts) <= words else " ".join(parts[:words]) + " …"


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
    if d.information_only:
        return f"**{ui_text(lang, 'report.information_only')}**"
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
        bits.append(f"“{escape(shorten(e.transcript.text))}”")
    if e.time is not None:
        bits.append(f"({clock(e.time.start_s)}–{clock(e.time.end_s)})")
    return "- " + " ".join(bits)


def _measure_label(lang: Language, key: str) -> str:
    try:
        return text(lang, f"assessment.measures.{key}")
    except KeyError:
        return key


def _unmeasured(r: DimensionResult, lang: Language) -> list[str]:
    """One line per measure that was not measured (and why), and one for those left out of the
    count because the camera was off."""
    lines = [
        "- "
        + escape(
            ui_text(
                lang,
                "report.not_measured",
                measure=_measure_label(lang, k),
                reason=r.missing_reasons.get(k, text(lang, "assessment.missing_other")),
            )
        )
        for k in r.missing
    ]
    if r.excluded:
        names = text(lang, "assessment.separator").join(_measure_label(lang, k) for k in r.excluded)
        lines.append("- " + escape(ui_text(lang, "report.not_counted_camera", measures=names)))
    return lines


def dimensions_line(summary: SessionSummary, names: dict[str, str], lang: Language) -> str:
    """ "Levels based on N of M dimensions." and the information-only ones by name."""
    dims = summary.dimensions
    scored = sum(d.level is not None for d in dims)
    line = ui_text(lang, "report.dimensions_line", measured=scored, total=len(dims))
    info = [names.get(d.dimension, d.dimension) for d in dims if d.information_only]
    if info:
        sep = text(lang, "assessment.separator")
        line = f"{line} {ui_text(lang, 'report.not_scored_line', names=sep.join(info))}"
    return line


def render(view: ReportView, lang: Language, scenario: str, *, max_evidence: int = 5) -> str:
    """The report as Markdown, in `lang`."""
    names = _names(scenario, lang)
    out: list[str] = [f"## {ui_text(lang, 'report.heading')}"]
    if view.text_note:
        out.append(f"> {escape(view.text_note)}")
    if view.report is not None and view.report.followups is not None:  # S2 (P3-e)
        out.append(f"**{ui_text(lang, 'report.questions')}**")
        out.extend(f"{i}\\. {escape(q)}" for i, q in enumerate(view.report.followups.questions, 1))
    findings = {f.dimension: f for f in view.report.assessment.findings} if view.report else {}
    results = {r.dimension: r for r in view.report.assessment.dimensions} if view.report else {}
    order = {dim: i for i, dim in enumerate(names)}  # the rubric's order
    dims = sorted(view.summary.dimensions, key=lambda d: order.get(d.dimension, len(order)))
    if dims:
        out.append(escape(dimensions_line(view.summary, names, lang)))
    for d in dims:
        out.append(f"### {escape(names.get(d.dimension, d.dimension))}")
        out.append(_badges(d, lang))
        f = findings.get(d.dimension)
        if f is None:
            continue
        out.append(escape(f.statement))
        r = results.get(d.dimension)
        if r is not None and d.level is not None:  # a not-measurable card already says why
            out.extend(_unmeasured(r, lang))
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
