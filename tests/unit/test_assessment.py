"""Rubric scoring and the assessment (P2-c, ADR 0019): levels, partial evidence, verification.

No models: the speech report is built by hand, the text report uses the fake parser.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import pytest

from presenceai.coach.guardrails import contains_banned_term, contains_verdict_term
from presenceai.evalconfig import REPO_ROOT
from presenceai.rubric import derived, evidence, loader, score
from presenceai.schemas.assessment import Confidence, Evidence, Measurement, TimeSpan
from presenceai.schemas.speech import (
    AsrQuality,
    PauseEvent,
    PausePosition,
    SpeechRegion,
    SpeechReport,
)
from presenceai.text import analyze

TALK = (
    "Have you ever frozen in front of an audience? "
    "Today I will show that short daily practice improves presentation skills. "
    "First, practice builds memory. "
    "For example, 45 percent of students said ten minutes a day helped. "
    "Next, imagine your talk is like a song you rehearse. "
    "According to one coach, rehearsal is the skill that matters. "
    "In conclusion, practice a little every day to grow your skills."
)  # 67 words, one every 0.5 s: speech from 0.0 to 33.3 s
ARABIC = re.compile(r"[؀-ۿ]")


def _measurement(value: float, conf: Confidence, end_s: float = 33.3) -> Measurement:
    return Measurement(
        value=value, unit="per 100 words", confidence=conf, method="test",
        evidence=(Evidence(time=TimeSpan(start_s=0.0, end_s=end_s)),), language="en", live=False,
    )  # fmt: skip


@pytest.fixture
def build(sb: Any, fake_parser: Any, speech_cfg: Any) -> Any:
    def make(
        talk: str = TALK,
        *,
        language: str = "en",
        fillers: float | None = 3.0,
        filler_conf: Confidence = Confidence.MEDIUM,
        mid_pauses: int = 1,
        filler_end_s: float = 33.3,
    ) -> tuple[SpeechReport, Any]:
        t = sb.spaced(talk, language=language, duration_s=40.0)
        last = t.words[-1].end_s
        regions = (SpeechRegion(start_s=0.0, end_s=10.0), SpeechRegion(start_s=11.2, end_s=last))
        events = tuple(
            PauseEvent(start_s=10.0, end_s=11.2, position=PausePosition.MID_SENTENCE,
                       confidence=Confidence.MEDIUM, word_before=19, word_after=20)
            for _ in range(mid_pauses)
        )  # fmt: skip
        measurements = {}
        if fillers is not None:
            measurements["fillers.per_100_words"] = _measurement(fillers, filler_conf, filler_end_s)
        report = SpeechReport(
            source_name="talk.wav", model="whisper_small", transcript=t,
            asr_quality=AsrQuality(word_count=len(t.words)), speech_regions=regions,
            pauses=events, measurements=measurements,
        )  # fmt: skip
        text_report = analyze.analyze_text(t, parser=fake_parser, pauses=speech_cfg.pauses)
        rubric = loader.load_rubric("s1_presentation")
        thresholds = loader.load_thresholds("s1_presentation")
        return report, score.assess(rubric, thresholds, report, text_report, speech_cfg)

    return make


def _dim(a: Any, dim: str) -> Any:
    return next(r for r in a.dimensions if r.dimension == dim)


def _finding(a: Any, dim: str) -> Any:
    return next(f for f in a.findings if f.dimension == dim)


# --- the derived measure ----------------------------------------------------------------------


def test_mid_sentence_pauses_per_minute_of_speaking_time(build: Any, speech_cfg: Any):
    report, _ = build(mid_pauses=2)
    m, why = derived.mid_sentence_per_min(report, speech_cfg, 1.0)
    assert why is None and m is not None
    assert m.value == pytest.approx(2 / (33.3 / 60), abs=1e-3)  # first onset to last offset
    assert all(e.time is not None and e.transcript is not None for e in m.evidence)
    m, _ = derived.mid_sentence_per_min(report, speech_cfg, 1.5)  # pauses are 1.2 s
    assert m is not None and m.value == 0 and m.evidence[0].time is not None


def test_mid_sentence_pauses_missing_without_positions(build: Any, speech_cfg: Any):
    report, _ = build("we practise our talk every single day " * 6)  # no sentence marks
    m, why = derived.mid_sentence_per_min(report, speech_cfg, 1.0)
    assert m is None and why is not None and "sentence marks" in why
    report, _ = build("اليوم سأتحدث عن التدريب. " * 10, language="ar")
    m, why = derived.mid_sentence_per_min(report, speech_cfg, 1.0)
    assert m is None and why is not None


# --- levels, partial evidence, confidence -----------------------------------------------------


def test_levels_and_partial_evidence(build: Any):
    _, a = build()
    delivery = _dim(a, "delivery")
    assert delivery.measured == ("fillers.per_100_words", "pauses.mid_sentence_per_min")
    assert set(delivery.missing) == {"vision.gaze.at_camera_share", "timing.overrun_s"}
    assert delivery.level == 3  # 3 fillers (<= 6) but 1.8 mid-sentence pauses/min (> 1)
    assert delivery.score == pytest.approx(2 / 3)
    gap = {r.measure for r in delivery.next_level}
    assert gap == {"pauses.mid_sentence_per_min"}  # what level 4 still needs, as data
    f = _finding(a, "delivery")
    assert "based on 2 of 4 measures" in f.statement and f.measured == 2 and f.total == 4
    assert _dim(a, "central_message").level == 4  # stated, restated, 2 recurrences
    assert _dim(a, "organization").level == 3  # 2 transitions (< 3 for level 4)


def test_confidence_is_the_lowest_deciding_measure(build: Any):
    _, a = build(filler_conf=Confidence.LOW)
    assert _dim(a, "delivery").confidence == Confidence.LOW
    assert _dim(a, "language").confidence == Confidence.MEDIUM


def test_missing_measures_never_lower_a_level(build: Any):
    _, a = build(fillers=None, mid_pauses=0)
    delivery = _dim(a, "delivery")
    assert delivery.measured == ("pauses.mid_sentence_per_min",)
    assert delivery.level == 4  # 0 mid-sentence pauses; nothing else counted against it


def test_arabic_content_not_measurable_and_delivery_from_fillers(build: Any):
    _, a = build("اليوم سأتحدث عن التدريب اليومي. " * 12, language="ar",
                 filler_conf=Confidence.LOW)  # fmt: skip
    for dim in ("central_message", "organization", "language", "supporting_material"):
        r = _dim(a, dim)
        assert r.level is None and r.not_measurable and r.score is None
        assert _finding(a, dim).measured == 0 and not _finding(a, dim).evidence
    delivery = _dim(a, "delivery")
    assert delivery.measured == ("fillers.per_100_words",) and delivery.level == 4
    assert delivery.confidence == Confidence.LOW  # still a level, at low confidence
    f = _finding(a, "delivery")
    assert ARABIC.search(f.statement) and "1" in f.statement and "4" in f.statement


# --- evidence verification (the P2 exit criterion) --------------------------------------------


def test_every_finding_has_verified_evidence(build: Any):
    report, a = build()
    assert a.evidence_verified
    for f in a.findings:
        if f.level is not None:
            assert f.evidence
            evidence.verify(f.evidence, report.transcript, report.transcript.duration_s)


def test_verification_rejects_shifted_quotes_and_times(build: Any):
    report, a = build()
    f = _finding(a, "central_message")
    span = next(e.transcript for e in f.evidence if e.transcript is not None)
    shifted = span.model_copy(update={"start_char": span.start_char + 1})
    with pytest.raises(evidence.EvidenceError):
        evidence.verify([Evidence(transcript=shifted)], report.transcript, 40.0)
    late = Evidence(time=TimeSpan(start_s=39.0, end_s=41.0))
    with pytest.raises(evidence.EvidenceError):
        evidence.verify([late], report.transcript, 40.0)


def test_a_rule_based_finding_with_bad_evidence_is_a_bug(build: Any):
    with pytest.raises(evidence.EvidenceError):
        build(filler_end_s=45.0)  # the fillers evidence runs past the 40 s recording


# --- wording ----------------------------------------------------------------------------------


def test_findings_use_observable_wording(build: Any):
    cases: tuple[tuple[Literal["en", "ar"], str], ...] = (
        ("en", TALK),
        ("ar", "اليوم سأتحدث عن التدريب اليومي. " * 12),
    )
    for language, talk in cases:
        _, a = build(talk, language=language)
        for f in a.findings:
            assert not contains_verdict_term(f.statement, language), f.statement
            assert not contains_banned_term(f.statement, language), f.statement
            assert "{" not in f.statement


# --- the assess command -----------------------------------------------------------------------


def test_assess_command_writes_one_report(
    monkeypatch: pytest.MonkeyPatch, fake_whisper: Any, fake_parser: Any, silent_wav: Path,
    tmp_path: Path,
):  # fmt: skip
    from typer.testing import CliRunner

    from presenceai import cli
    from presenceai.speech import asr, pitch, vad
    from presenceai.text import nlp

    from .test_analyze_cli import fake_pitch, fake_vad

    monkeypatch.setattr(asr, "load_model", lambda name, opts: fake_whisper())
    monkeypatch.setattr(vad, "detect_speech", fake_vad)
    monkeypatch.setattr(pitch, "track", fake_pitch)
    monkeypatch.setattr(nlp, "spacy_parser", lambda: fake_parser)
    out = tmp_path / "assessment.json"
    result = CliRunner().invoke(
        cli.app, ["assess", str(silent_wav), "--lang", "en", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text("utf-8"))
    assert set(data) == {"schema_version", "speech", "text", "assessment", "followups"}
    assert data["followups"] is None
    assert data["assessment"]["evidence_verified"] is True
    assert len(data["assessment"]["findings"]) == 5


# --- slow: the owner's B1 English reports, on the laptop only ---------------------------------


@pytest.mark.slow
def test_b1_english_reports_have_verified_findings(speech_cfg: Any):
    from presenceai.pipeline.assess import assess_report
    from presenceai.speech.asr import ModelNotAvailable
    from presenceai.text.nlp import spacy_parser

    reports = sorted((REPO_ROOT / "data" / "b1" / "reports").glob("b1_en_*.json"))
    if not reports:
        pytest.skip("no B1 reports (owner's laptop only)")
    try:
        parser = spacy_parser()
    except ModelNotAvailable:
        pytest.skip("en_core_web_sm not downloaded")
    findings = 0
    for path in reports:  # counts only: no transcript text is printed or written
        speech = SpeechReport.model_validate_json(path.read_text("utf-8"))
        result = assess_report(speech, parser=parser, cfg=speech_cfg, scenario="s1_presentation")
        assert result.assessment.evidence_verified
        findings += sum(f.level is not None for f in result.assessment.findings)
    assert findings > 0
