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
        limit_s: float | None = None,
        strict: bool = True,
        camera_on: bool = False,
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
        return report, score.assess(
            rubric, thresholds, report, text_report, speech_cfg, limit_s=limit_s, strict=strict,
            camera_on=camera_on,
        )  # fmt: skip

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
    assert delivery.missing == ("timing.overrun_s",)  # an upload: no time limit (counted in M)
    assert delivery.excluded == ("vision.gaze.at_camera_share",)  # camera off: left out of M
    assert delivery.level == 3  # 3 fillers (<= 6) but 1.8 mid-sentence pauses/min (> 1)
    assert delivery.score == pytest.approx(2 / 3)
    gap = {r.measure for r in delivery.next_level}
    assert gap == {"pauses.mid_sentence_per_min"}  # what level 4 still needs, as data
    f = _finding(a, "delivery")
    assert "based on 2 of 3 measures" in f.statement and f.measured == 2 and f.total == 3
    cm = _dim(a, "central_message")  # information-only (ADR 0025): no level, score or cap
    assert cm.information_only and cm.level is None and cm.score is None and cm.capped_from is None
    assert _dim(a, "organization").level == 3  # 2 transitions (< 3 for level 4)


def test_confidence_is_the_lowest_deciding_measure(build: Any):
    _, a = build(filler_conf=Confidence.LOW)
    assert _dim(a, "delivery").confidence == Confidence.LOW
    assert _dim(a, "language").confidence == Confidence.MEDIUM


def test_missing_measures_are_dropped_not_read_as_low(build: Any):
    _, a = build(fillers=None, mid_pauses=0)
    delivery = _dim(a, "delivery")
    assert delivery.measured == ("pauses.mid_sentence_per_min",)
    # 0 mid-sentence pauses; nothing else counted against it, so the conditions reach level 4.
    # The partial cap is the one deliberate exception: 1 of 3 measures caps it at 3 (ADR 0019).
    assert delivery.capped_from == 4 and delivery.level == 3


def test_arabic_content_not_measurable_and_delivery_from_fillers(build: Any):
    _, a = build("اليوم سأتحدث عن التدريب اليومي. " * 12, language="ar",
                 filler_conf=Confidence.LOW)  # fmt: skip
    for dim in ("organization", "language", "supporting_material"):
        r = _dim(a, dim)
        assert r.level is None and r.not_measurable and r.score is None
        assert _finding(a, dim).measured == 0 and not _finding(a, dim).evidence
    cm = _finding(a, "central_message")  # information-only; no possible main point in Arabic
    assert _dim(a, "central_message").information_only and not cm.evidence
    assert "فكرة رئيسية محتملة" not in cm.statement
    delivery = _dim(a, "delivery")
    assert delivery.measured == ("fillers.per_100_words",)
    assert delivery.level == 3 and delivery.capped_from == 4  # 1 of 3 measures: capped
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


# --- P3-a: partial levels, overrun, one failed finding (ADR 0019 amendment) -------------------


def test_partial_cap_values_come_from_the_config():
    cap = loader.load_partial_cap()
    assert cap.max_share == 0.5 and cap.max_level == 3


def test_at_most_half_of_the_measures_caps_the_level_at_3(build: Any):
    # camera on, gaze missing for a system reason: 2 of 4 computed, both meet level 4
    _, a = build(fillers=3.0, mid_pauses=0, camera_on=True)
    delivery = _dim(a, "delivery")
    assert len(delivery.measured) == 2 and not delivery.excluded
    assert delivery.level == 3 and delivery.capped_from == 4
    assert delivery.score == pytest.approx(2 / 3)  # the score follows the capped level
    f = _finding(a, "delivery")
    assert f.level == 3 and f.capped_from == 4
    assert "limited to level 3" in f.statement and "based on 2 of 4 measures" in f.statement


def test_more_than_half_of_the_measures_is_not_capped(build: Any):
    _, a = build(fillers=3.0, mid_pauses=0, limit_s=180.0, camera_on=True)  # + overrun: 3 of 4
    delivery = _dim(a, "delivery")
    assert len(delivery.measured) == 3
    assert delivery.level == 4 and delivery.capped_from is None
    assert "limited to level" not in _finding(a, "delivery").statement


def test_the_cap_never_changes_levels_1_to_3(build: Any):
    _, a = build(fillers=None, mid_pauses=2)  # 1 of 4: 3.6 pauses/min -> level 2
    delivery = _dim(a, "delivery")
    assert delivery.level == 2 and delivery.capped_from is None


def test_arabic_camera_on_with_gaze_missing_for_a_system_reason_is_capped(build: Any):
    talk = "اليوم سأتحدث عن التدريب اليومي. " * 12
    _, a = build(talk, language="ar", limit_s=180.0, camera_on=True)
    delivery = _dim(a, "delivery")
    assert set(delivery.measured) == {"fillers.per_100_words", "timing.overrun_s"}  # 2 of 4
    assert "vision.gaze.at_camera_share" in delivery.missing and not delivery.excluded
    assert delivery.level == 3 and delivery.capped_from == 4
    assert ARABIC.search(_finding(a, "delivery").statement)


def test_arabic_camera_off_delivery_is_2_of_3_and_not_capped(build: Any):
    # accepted by the owner (ADR 0019, P3-a item 2): camera off is the user's choice
    talk = "اليوم سأتحدث عن التدريب اليومي. " * 12
    _, a = build(talk, language="ar", limit_s=180.0, filler_conf=Confidence.LOW)
    delivery = _dim(a, "delivery")
    assert delivery.level == 4 and delivery.capped_from is None
    assert delivery.confidence == Confidence.LOW
    f = _finding(a, "delivery")
    assert f.measured == 2 and f.total == 3


def test_english_upload_with_the_camera_off_is_not_capped(build: Any):
    _, a = build(fillers=3.0, mid_pauses=0)  # upload: no limit; camera off -> 2 of 3
    delivery = _dim(a, "delivery")
    assert delivery.level == 4 and delivery.capped_from is None
    f = _finding(a, "delivery")
    assert (f.measured, f.total) == (2, 3) and "based on 2 of 3 measures" in f.statement


def test_n_of_m_leaves_camera_measures_out_only_when_the_camera_is_off(build: Any):
    _, off = build(limit_s=180.0)
    _, on = build(limit_s=180.0, camera_on=True)
    assert (_finding(off, "delivery").measured, _finding(off, "delivery").total) == (3, 3)
    assert (_finding(on, "delivery").measured, _finding(on, "delivery").total) == (3, 4)
    assert _dim(off, "delivery").excluded == ("vision.gaze.at_camera_share",)


def test_opted_out_measures_are_left_out_of_m(build: Any, speech_cfg: Any):
    report, _ = build()
    rubric = loader.load_rubric("s1_presentation")
    text_report = analyze.analyze_text(report.transcript, parser=lambda t: [[] for _ in t],
                                       pauses=speech_cfg.pauses)  # fmt: skip
    a = score.assess(
        rubric, loader.load_thresholds("s1_presentation"), report, text_report, speech_cfg,
        opted_out=frozenset({"pauses.mid_sentence_per_min"}),
    )  # fmt: skip
    delivery = _dim(a, "delivery")
    assert "pauses.mid_sentence_per_min" in delivery.excluded
    assert "pauses.mid_sentence_per_min" not in delivery.measured
    assert _finding(a, "delivery").total == 2  # fillers + overrun; gaze (camera off) too left out


def test_every_finding_states_n_of_m_measures(build: Any):
    for kwargs in ({}, {"limit_s": 180.0}, {"fillers": None}):
        _, a = build(**kwargs)
        for f in a.findings:
            assert f.total >= 1 and 0 <= f.measured <= f.total
            if _dim(a, f.dimension).information_only:
                assert "not scored" in f.statement and "measures" not in f.statement
            elif f.level is not None:
                assert f"based on {f.measured} of {f.total} measures" in f.statement
            else:
                assert f"none of its {f.total} measures" in f.statement
    _, a = build()
    assert _finding(a, "central_message").measured == 0  # information-only: nothing scored


def test_overrun_against_the_session_limit(build: Any):
    _, a = build(limit_s=30.0)  # speech ends at 33.3 s
    delivery = _dim(a, "delivery")
    over = next(c for c in delivery.conditions if c.measure == "timing.overrun_s")
    assert over.value == pytest.approx(3.3, abs=1e-6)
    f = _finding(a, "delivery")
    times = [e.time for e in f.evidence if e.time is not None]
    assert any(t.start_s == 30.0 and t.end_s == pytest.approx(33.3) for t in times)
    _, a = build(limit_s=180.0)
    over = next(c for c in _dim(a, "delivery").conditions if c.measure == "timing.overrun_s")
    assert over.value == 0 and over.holds is True


def test_overrun_of_an_upload_is_missing_with_its_reason(build: Any, speech_cfg: Any):
    report, a = build()  # no limit: a file, not an in-app recording
    assert "timing.overrun_s" in _dim(a, "delivery").missing
    m, why = derived.overrun(report, None)
    assert m is None and why == "uploaded file: no time limit"
    report, _ = build("اليوم سأتحدث عن التدريب اليومي. " * 12, language="ar")
    _, why = derived.overrun(report, None)
    assert why is not None and ARABIC.search(why)


def test_overrun_needs_speech(build: Any):
    report, _ = build()
    silent = report.model_copy(update={"speech_regions": ()})
    m, why = derived.overrun(silent, 180.0)
    assert m is None and why


def test_in_the_app_one_failed_finding_does_not_crash_the_report(
    build: Any, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level("WARNING"):
        _, a = build(filler_end_s=45.0, strict=False)  # fillers evidence past the recording
    delivery = _dim(a, "delivery")
    assert delivery.level is None and delivery.not_measurable == "could not be checked"
    f = _finding(a, "delivery")
    assert f.level is None and not f.evidence and "not measurable" in f.statement
    assert a.failed_dimensions == ("delivery",)
    assert _dim(a, "organization").level == 3  # the rest of the report stands
    assert _dim(a, "central_message").information_only
    logged = caplog.text
    assert "delivery" in logged and "EvidenceError" in logged
    assert "practice" not in logged.lower()  # counts only, never transcript text


def test_in_the_app_any_error_in_one_dimension_is_isolated(
    build: Any, monkeypatch: pytest.MonkeyPatch
):
    real = score._finding

    def boom(rubric: Any, d: Any, *args: Any) -> Any:
        if d.id == "organization":
            raise KeyError("a template key")
        return real(rubric, d, *args)

    monkeypatch.setattr(score, "_finding", boom)
    _, a = build(strict=False)
    assert a.failed_dimensions == ("organization",)
    assert _dim(a, "organization").level is None
    assert _dim(a, "delivery").level is not None
    with pytest.raises(KeyError):
        build()  # outside the app (CLI, tests) a failure still stops


def test_in_the_app_a_failed_text_analysis_leaves_delivery(build: Any, speech_cfg: Any):
    from presenceai.pipeline.assess import assess_report

    report, _ = build()

    def broken(texts: Any) -> Any:
        raise RuntimeError("parser crashed")

    out = assess_report(
        report, parser=broken, cfg=speech_cfg, scenario="s1_presentation", strict=False
    )
    a = out.assessment
    for dim in ("organization", "language", "supporting_material"):
        assert _dim(a, dim).level is None
        assert "text analysis failed" in (_dim(a, dim).not_measurable or "")
    assert _dim(a, "central_message").information_only  # no text: nothing quoted
    assert not _finding(a, "central_message").evidence
    assert _dim(a, "delivery").level is not None
    with pytest.raises(RuntimeError):
        assess_report(report, parser=broken, cfg=speech_cfg, scenario="s1_presentation")


# --- central message information-only (owner, 2026-11-26, ADR 0025) ---------------------------


def test_central_message_is_shown_not_scored_with_the_possible_main_point(build: Any):
    report, a = build()
    r, f = _dim(a, "central_message"), _finding(a, "central_message")
    assert r.information_only and r.level is None and r.score is None and r.not_measurable is None
    assert f.level is None and f.confidence == Confidence.LOW
    quote = "Today I will show that short daily practice improves presentation skills."
    assert f.statement.endswith(f"Possible main point (low confidence): “{quote}”")
    assert "not scored" in f.statement and "hand-labelled practice talks" in f.statement
    assert [e.transcript.text for e in f.evidence if e.transcript] == [quote]
    assert all(e.time is not None for e in f.evidence)
    evidence.verify(f.evidence, report.transcript, report.transcript.duration_s)


@pytest.mark.parametrize("lang", ["en", "ar"])
def test_without_a_possible_main_point_the_card_says_nothing_about_one(
    build: Any, lang: Literal["en", "ar"]
):
    from presenceai.lang import text

    talk = "We met. We talked. It was fine. We left. We slept. We woke. We ate. We went home. " * 3
    _, a = build(talk)
    f = _finding(a, "central_message")
    assert not f.evidence and f.confidence is None
    assert f.statement == text("en", "assessment.information_only.central_message",
                               dimension="Central message")  # fmt: skip
    for wording in ("no main point", "not found", "No sentence"):
        assert wording.lower() not in f.statement.lower()
    reason = text(lang, "assessment.information_only.central_message", dimension="x")
    assert "ADR" not in reason and "P7" not in reason  # owner, 2026-11-26: no plan codes
    assert ("language-model check" if lang == "en" else "فحص بالنموذج اللغوي") in reason


def test_information_only_is_config_driven_for_both_scenarios():
    from presenceai.rubric.loader import load_thresholds

    for scenario in ("s1_presentation", "s2_project_defense"):
        assert load_thresholds(scenario).information_only == ("central_message",)


def test_information_only_must_name_a_rubric_dimension(tmp_path: Any):
    import yaml

    from presenceai.rubric import loader

    data = yaml.safe_load(loader.CONFIG.read_text("utf-8"))
    data["s1_presentation"]["information_only"] = ["no_such_dimension"]
    cfg = tmp_path / "rubric.yaml"
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown dimensions"):
        loader.load_rubric("s1_presentation", config=cfg)
