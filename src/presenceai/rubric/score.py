"""Rubric scoring: speech report + text report -> Assessment with verified evidence (ADR 0019).

Rule-based only in P2-c (the language-model judge and Laya come later, ADR 0017). For each
dimension:
- the condition results (value, threshold, holds; missing measures stay missing);
- the level (ADR 0017), the score y = (level - 1) / 3, and the confidence: the lowest among the
  measures that were computed. A level from low-confidence measures only is still given, at low
  confidence; the engine down-weights it (FINAL_05 §2);
- how many of its measures were computed ("based on 1 of 4 measures");
- the next level's unmet conditions as structured data (their wording is for the P7 coach slice,
  in the templates as well as the language model, so users without consent (d) get a next step).
One finding per dimension, in observable wording, with the evidence of its computed measures.
Every finding's evidence is verified; a failure in a rule-based finding raises EvidenceError.

S2 (ADR 0022): the follow-up measures come from a `FollowupReport`; their evidence names the
answer it comes from and is verified against that answer's transcript. A measure with a
condition on an uncalibrated threshold (`RubricThresholds.uncalibrated`) is reported with its
value but not scored. A follow-up answered with silence adds the answer window to the finding.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from presenceai.lang import text
from presenceai.rubric import derived
from presenceai.rubric.evidence import verify
from presenceai.rubric.levels import condition_results, level_from, measures_of
from presenceai.rubric.loader import RubricThresholds, level_score
from presenceai.schemas.assessment import Confidence, Evidence, Measurement, TimeSpan
from presenceai.schemas.followup import FollowupReport
from presenceai.schemas.rubric import Assessment, Dimension, DimensionResult, Finding, Rubric
from presenceai.schemas.speech import SpeechReport
from presenceai.schemas.text import TextReport
from presenceai.settings import SpeechSettings

_RANK = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}
_MISSING_KEY = {
    "camera": "assessment.missing_camera",
    "session_limit": "assessment.missing_limit",
    "followups": "assessment.missing_followups",
}


def _report_lang(lang: str) -> Literal["en", "ar"]:
    return "ar" if lang == "ar" else "en"


def _uncalibrated(rubric: Rubric, thresholds: RubricThresholds) -> set[str]:
    """Measures with a condition on a placeholder threshold: reported, not scored."""
    held = set(thresholds.uncalibrated)
    return {
        c.measure
        for d in rubric.dimensions
        for a in d.levels
        for c in a.conditions
        if c.threshold in held
    }


def collect(
    rubric: Rubric,
    thresholds: RubricThresholds,
    speech: SpeechReport,
    text_report: TextReport,
    cfg: SpeechSettings,
    followups: FollowupReport | None = None,
) -> tuple[dict[str, Measurement], dict[str, str]]:
    """Every rubric measure: its Measurement, or the reason it is missing."""
    lang = _report_lang(speech.transcript.language)
    found = dict(speech.measurements) | dict(text_report.measurements)
    reasons = dict(speech.not_measurable) | dict(text_report.not_measurable)
    if followups is not None:
        found |= followups.measurements
        reasons |= followups.not_measurable
    mid, why = derived.mid_sentence_per_min(speech, cfg, thresholds.values["mid_pause_min_s"])
    if mid is not None:
        found[derived.KEY] = mid
    elif why is not None:
        reasons[derived.KEY] = why
    out_found: dict[str, Measurement] = {}
    out_missing: dict[str, str] = {}
    for key, spec in rubric.measures.items():
        if key in found:
            out_found[key] = found[key]
        else:
            fallback = _MISSING_KEY.get(spec.requires, "assessment.missing_other")
            out_missing[key] = reasons.get(key) or text(lang, fallback)
    for key in _uncalibrated(rubric, thresholds) & set(out_found):
        value = out_found.pop(key).value
        out_missing[key] = text(lang, "assessment.uncalibrated", value=round(value, 3))
    return out_found, out_missing


def _dimension(
    d: Dimension,
    found: Mapping[str, Measurement],
    missing: Mapping[str, str],
    t: RubricThresholds,
) -> DimensionResult:
    keys = measures_of(d)
    values = {k: (found[k].value if k in found else None) for k in keys}
    results = condition_results(d, values, t)
    level = level_from(results)
    measured = tuple(k for k in keys if k in found)
    absent = tuple(k for k in keys if k not in found)
    if level is None:
        return DimensionResult(
            dimension=d.id,
            skills=d.skills,
            missing=absent,
            conditions=results,
            not_measurable="; ".join(dict.fromkeys(missing[k] for k in absent)),
        )
    confidence = min((found[k].confidence for k in measured), key=_RANK.__getitem__)
    nxt = tuple(r for r in results if r.level == level + 1 and r.holds is False)
    return DimensionResult(
        dimension=d.id,
        skills=d.skills,
        level=level,
        score=level_score(level),
        confidence=confidence,
        measured=measured,
        missing=absent,
        conditions=results,
        next_level=nxt,
    )


def _finding(
    rubric: Rubric,
    d: Dimension,
    r: DimensionResult,
    found: Mapping[str, Measurement],
    missing: Mapping[str, str],
    lang: str,
) -> Finding:
    lang_ = "ar" if lang == "ar" else "en"
    name = d.name_ar if lang_ == "ar" else d.name
    sep = text(lang_, "assessment.separator")
    total = len(measures_of(d))
    if r.level is None:
        reasons = sep.join(dict.fromkeys(missing[k] for k in r.missing))
        statement = text(
            lang_, "assessment.not_measurable", dimension=name, total=total, reasons=reasons
        )
        return Finding(dimension=d.id, statement=statement, measured=0, total=total)
    labels = {k: text(lang_, f"assessment.measures.{k}") for k in r.measured}
    values = sep.join(
        text(lang_, "assessment.value", measure=labels[k], value=found[k].value) for k in r.measured
    )
    scale = rubric.scale_ar if lang_ == "ar" else rubric.scale
    statement = text(
        lang_,
        "assessment.finding",
        dimension=name,
        level=r.level,
        level_name=scale[r.level],
        measured=len(r.measured),
        total=total,
        measures=sep.join(labels.values()),
        values=values,
    )
    evidence: tuple[Evidence, ...] = tuple(
        dict.fromkeys(e for k in r.measured for e in found[k].evidence)
    )
    return Finding(
        dimension=d.id,
        statement=statement,
        level=r.level,
        confidence=r.confidence,
        measured=len(r.measured),
        total=total,
        evidence=evidence,
    )


def _with_silent_answers(
    f: Finding, d: Dimension, followups: FollowupReport | None, lang: str
) -> Finding:
    """Name each follow-up answered with silence, with its answer window as time evidence."""
    if followups is None or f.level is None:
        return f
    if not any(k.startswith("followup.") for k in measures_of(d)):
        return f
    silent = [a for a in followups.summary.answers if a.status == "no_answer" and a.window_s]
    if not silent:
        return f
    notes = " ".join(
        text(_report_lang(lang), "assessment.followup_no_answer", answer=a.index + 1,
             window=a.window_s)
        for a in silent
    )  # fmt: skip
    windows = tuple(
        Evidence(time=TimeSpan(start_s=0.0, end_s=a.window_s or 0.0), answer=a.index)
        for a in silent
    )
    return f.model_copy(
        update={
            "statement": f"{f.statement} {notes}",
            "evidence": tuple(dict.fromkeys((*f.evidence, *windows))),
        }
    )


def assess(
    rubric: Rubric,
    thresholds: RubricThresholds,
    speech: SpeechReport,
    text_report: TextReport,
    cfg: SpeechSettings,
    followups: FollowupReport | None = None,
) -> Assessment:
    lang = speech.transcript.language
    found, missing = collect(rubric, thresholds, speech, text_report, cfg, followups)
    results = tuple(_dimension(d, found, missing, thresholds) for d in rubric.dimensions)
    findings = tuple(
        _with_silent_answers(_finding(rubric, d, r, found, missing, lang), d, followups, lang)
        for d, r in zip(rubric.dimensions, results, strict=True)
    )
    answers = followups.transcripts if followups is not None else ()
    for f in findings:
        if f.level is not None:
            verify(f.evidence, speech.transcript, speech.transcript.duration_s, answers)
    return Assessment(
        scenario=rubric.id,
        rubric_version=rubric.version,
        thresholds_status=thresholds.status,
        language="ar" if lang == "ar" else "en",
        dimensions=results,
        findings=findings,
    )
