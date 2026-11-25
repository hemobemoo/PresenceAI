"""S2-c (ADR 0022): follow-up answers in the assessment, with evidence that names the answer.

No models: speech reports are built by hand, relevance uses the fake bag-of-words embedder.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from presenceai.pipeline.assess import no_parser
from presenceai.pipeline.followup import ANSWERED, LATENCY, LIMITATION, RELEVANCE, followup_report
from presenceai.rubric import evidence, loader, score
from presenceai.schemas.assessment import Evidence, TimeSpan
from presenceai.schemas.followup import FollowupReport
from presenceai.schemas.speech import AsrQuality, SpeechRegion, SpeechReport
from presenceai.text import analyze

from .test_assessment import TALK
from .test_followup import _BagOfWords

S2 = "s2_project_defense"
Q1 = "How did you test the pause detector?"
Q2 = "What are the limits of the gaze model?"
A1 = "We tested the pause detector on synthetic clips with known silences."
A2 = "One limitation is that we did not test the gaze model on other people."
WINDOW = 60.0


def _answer(sb: Any, text: str, *, onset: float = 1.5, language: str = "en") -> SpeechReport:
    """An answer recording: speech starts `onset` s after the question ends."""
    t = sb.spaced(text, language=language, duration_s=WINDOW)
    t = sb.transcript(
        [(w.text, w.start_s + onset, w.end_s + onset) for w in t.words],
        language=language, duration_s=WINDOW,
    )  # fmt: skip
    last = t.words[-1].end_s
    return SpeechReport(
        source_name="answer.wav", model="whisper_small", transcript=t,
        asr_quality=AsrQuality(word_count=len(t.words)),
        speech_regions=(SpeechRegion(start_s=onset, end_s=last),),
    )  # fmt: skip


def _silence(sb: Any, language: str = "en") -> SpeechReport:
    t = sb.transcript([], language=language, duration_s=WINDOW)
    return SpeechReport(
        source_name="answer.wav", model="whisper_small", transcript=t,
        asr_quality=AsrQuality(word_count=0),
    )  # fmt: skip


def _talk(sb: Any, language: str = "en") -> SpeechReport:
    t = sb.spaced(TALK, language=language, duration_s=40.0)
    return SpeechReport(
        source_name="talk.wav", model="whisper_small", transcript=t,
        asr_quality=AsrQuality(word_count=len(t.words)),
        speech_regions=(SpeechRegion(start_s=0.0, end_s=t.words[-1].end_s),),
    )  # fmt: skip


def _calibrated() -> loader.RubricThresholds:
    return loader.load_thresholds(S2).model_copy(update={"uncalibrated": ()})


def _gated() -> loader.RubricThresholds:
    keys = ("fu_relevance_high", "fu_relevance_mid", "fu_relevance_low")
    return loader.load_thresholds(S2).model_copy(update={"uncalibrated": keys})


def _assess(
    sb: Any, fake_parser: Any, cfg: Any, fu: FollowupReport | None, t: Any, language: str = "en"
) -> Any:
    talk = _talk(sb, language)
    parser = fake_parser if language == "en" else no_parser
    text_report = analyze.analyze_text(talk.transcript, parser=parser, pauses=cfg.pauses)
    return score.assess(loader.load_rubric(S2), t, talk, text_report, cfg, fu)


def _fu(sb: Any, reports: list[SpeechReport | None], language: str = "en") -> FollowupReport:
    return followup_report(
        [Q1, Q2][: len(reports)], reports, [WINDOW] * len(reports), language=language,  # type: ignore[arg-type]
        embedder=_BagOfWords(),
    )  # fmt: skip


def _part(a: Any, dim: str = "followup_handling") -> tuple[Any, Any]:
    r = next(r for r in a.dimensions if r.dimension == dim)
    f = next(f for f in a.findings if f.dimension == dim)
    return r, f


def test_answered_followups_are_scored_with_answer_evidence(sb: Any, fake_parser: Any,
                                                            speech_cfg: Any):  # fmt: skip
    fu = _fu(sb, [_answer(sb, A1, onset=1.5), _answer(sb, A2, onset=3.0)])
    assert set(fu.measurements) == {RELEVANCE, LATENCY, LIMITATION, ANSWERED}
    lat = fu.measurements[LATENCY]
    assert lat.value == 3.0 and lat.evidence == (
        Evidence(time=TimeSpan(start_s=0.0, end_s=3.0), answer=1),
    )  # the slower answer (2nd) decides, timed from the end of its question
    lim = fu.measurements[LIMITATION]
    quote = lim.evidence[0].transcript
    assert lim.value == 1.0 and lim.evidence[0].answer == 1
    assert quote is not None and quote.text == "One limitation"
    a = _assess(sb, fake_parser, speech_cfg, fu, _calibrated())
    r, f = _part(a)
    assert r.level is not None and f.measured == 4 and f.total == 4
    assert {e.answer for e in f.evidence} <= {0, 1} and all(e.answer is not None
                                                            for e in f.evidence)  # fmt: skip
    assert a.evidence_verified


def test_relevance_is_reported_but_not_scored_while_uncalibrated(sb: Any, fake_parser: Any,
                                                                 speech_cfg: Any):  # fmt: skip
    fu = _fu(sb, [_answer(sb, A1), _answer(sb, A2)])
    r, f = _part(_assess(sb, fake_parser, speech_cfg, fu, _gated()))
    assert RELEVANCE in r.missing and RELEVANCE not in r.measured
    assert f.measured == 3 and f.total == 4  # "based on 3 of 4 measures"
    held = [c for c in r.conditions if c.measure == RELEVANCE]
    assert held and all(c.value is None and c.holds is None for c in held)


def test_uncalibrated_must_name_real_thresholds():
    with pytest.raises(ValueError, match="unknown thresholds"):
        loader.RubricThresholds(status="provisional", decided_in="ADR 0022", values={"a": 1.0},
                                uncalibrated=("b",))  # fmt: skip


def test_silent_answer_scores_level_1_with_the_window_as_evidence(sb: Any, fake_parser: Any,
                                                                  speech_cfg: Any):  # fmt: skip
    fu = _fu(sb, [_answer(sb, A2), _silence(sb)])
    assert fu.summary.answers[1].status == "no_answer"
    assert fu.measurements[LATENCY].value == WINDOW and fu.measurements[RELEVANCE].value == 0.0
    r, f = _part(_assess(sb, fake_parser, speech_cfg, fu, _calibrated()))
    assert r.level == 1
    window = Evidence(time=TimeSpan(start_s=0.0, end_s=WINDOW), answer=1)
    assert window in f.evidence
    assert "Answer 2: no speech in the 60 s answer window." in f.statement


def test_failed_recording_leaves_measures_missing_never_low(sb: Any, fake_parser: Any,
                                                            speech_cfg: Any):  # fmt: skip
    fu = _fu(sb, [None, _answer(sb, A1)])
    assert fu.summary.answers[0].status == "recording_failed" and fu.transcripts[0] is None
    assert fu.measurements[LATENCY].evidence[0].answer == 1  # only the recorded answer counts
    r, _ = _part(_assess(sb, fake_parser, speech_cfg, fu, _calibrated()))
    assert r.level is not None

    only_failed = _fu(sb, [None])
    r, f = _part(_assess(sb, fake_parser, speech_cfg, only_failed, _calibrated()))
    assert r.level is None and f.measured == 0  # not measurable, not level 1


def test_no_followups_is_not_measurable(sb: Any, fake_parser: Any, speech_cfg: Any):
    r, f = _part(_assess(sb, fake_parser, speech_cfg, None, _calibrated()))
    assert r.level is None and "no follow-up answer recorded" in f.statement


def test_arabic_answers_measure_latency_only(sb: Any, fake_parser: Any, speech_cfg: Any):
    fu = _fu(sb, [_answer(sb, "نعم اختبرناه", language="ar")], language="ar")
    assert set(fu.measurements) == {LATENCY, ANSWERED}  # both decided from the audio
    assert set(fu.not_measurable) == {RELEVANCE, LIMITATION}
    r, f = _part(_assess(sb, fake_parser, speech_cfg, fu, _calibrated(), language="ar"))
    assert r.level is not None and f.measured == 2 and f.total == 4
    # latency (1.5 s) and answered meet level 4; 2 of 4 measures caps it at 3 (ADR 0022 item 1)
    assert r.level == 3 and r.capped_from == 4 and f.capped_from == 4


def test_answer_evidence_is_checked_against_that_answer(sb: Any):
    talk, ans = _talk(sb), _answer(sb, A1)
    span = ans.transcript.span(0, 2)
    good = Evidence(transcript=span, answer=0)
    evidence.verify([good], talk.transcript, 40.0, (ans.transcript,))
    with pytest.raises(evidence.EvidenceError):
        evidence.verify([good], talk.transcript, 40.0, ())  # no such answer
    with pytest.raises(evidence.EvidenceError):
        evidence.verify([Evidence(transcript=span)], talk.transcript, 40.0, (ans.transcript,))


def test_assess_command_with_followups(
    monkeypatch: pytest.MonkeyPatch, fake_whisper: Any, fake_parser: Any, silent_wav: Path,
    tmp_path: Path,
):  # fmt: skip
    from typer.testing import CliRunner

    from presenceai import cli
    from presenceai.knowledge import embed
    from presenceai.speech import asr, pitch, vad
    from presenceai.text import nlp

    from .test_analyze_cli import fake_pitch, fake_vad

    monkeypatch.setattr(asr, "load_model", lambda name, opts: fake_whisper())
    monkeypatch.setattr(vad, "detect_speech", fake_vad)
    monkeypatch.setattr(pitch, "track", fake_pitch)
    monkeypatch.setattr(nlp, "spacy_parser", lambda: fake_parser)
    monkeypatch.setattr(embed, "E5Embedder", _BagOfWords)
    shutil.copy(silent_wav, tmp_path / "a1.wav")
    spec = tmp_path / "followups.json"
    spec.write_text(json.dumps({"answers": [
        {"question": Q1, "file": "a1.wav", "window_s": 2.0},
        {"question": Q2, "file": "missing.wav", "window_s": 2.0},
    ]}), encoding="utf-8")  # fmt: skip
    out = tmp_path / "assessment.json"
    args = ["assess", str(silent_wav), "--lang", "en", "--out", str(out), "--followups", str(spec)]
    result = CliRunner().invoke(cli.app, [*args, "--scenario", S2])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text("utf-8"))
    assert data["assessment"]["scenario"] == S2 and len(data["assessment"]["findings"]) == 6
    statuses = [a["status"] for a in data["followups"]["summary"]["answers"]]
    assert statuses[1] == "recording_failed"  # a missing file is a failed recording
    assert data["followups"]["sources"] == ["a1.wav", "missing.wav"]

    s1 = CliRunner().invoke(cli.app, args)  # --followups needs the S2 scenario
    assert s1.exit_code == 2


OFF = "Lunch is served in the cafeteria from noon every weekday."


def test_config_scores_relevance_once_calibrated(sb: Any, fake_parser: Any, speech_cfg: Any):
    fu = _fu(sb, [_answer(sb, A1), _answer(sb, A2)])
    assert fu.measurements[RELEVANCE].confidence.value == "low"  # relevance stays low confidence
    r, f = _part(_assess(sb, fake_parser, speech_cfg, fu, loader.load_thresholds(S2)))
    assert RELEVANCE in r.measured and f.measured == 4


def test_low_relevance_is_described_as_what_was_measured(sb: Any, fake_parser: Any,
                                                         speech_cfg: Any):  # fmt: skip
    fu = _fu(sb, [_answer(sb, A2), _answer(sb, OFF)])
    rel = fu.summary.answers[1].relevance
    assert rel is not None and rel < 0.82
    r, f = _part(_assess(sb, fake_parser, speech_cfg, fu, loader.load_thresholds(S2)))
    assert r.level == 1  # the weaker answer is below the one on-topic cut
    assert "Answer 2 shares little wording with the question" in f.statement
    assert "below 0.82" in f.statement
    for banned in ("off-topic", "off topic", "did not answer", "didn't answer", "you"):
        assert banned not in f.statement.casefold(), banned
    quoted = [e for e in f.evidence if e.answer == 1 and e.transcript is not None]
    assert quoted and quoted[0].transcript is not None and quoted[0].transcript.text == OFF
