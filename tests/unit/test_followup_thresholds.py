"""S2 thresholds after the owner's first real run (2026-11-25): a reading allowance for latency,
and relevance from bank questions reported but not scored."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from presenceai.knowledge.embed import Kind
from presenceai.pipeline.assess import assess_report
from presenceai.pipeline.followup import (
    ANSWERED,
    LATENCY,
    RELEVANCE,
    followup_report,
    reading_allowance,
)
from presenceai.schemas.speech import AsrQuality, SpeechRegion, SpeechReport

SHORT = "Why this model?"  # 3 words: 0.75 s to read at 4 words per second
LONG = (  # 18 words: 4.5 s to read
    "Which part of the project was the hardest for you, and how exactly did you deal with it?"
)
WINDOW = 60.0


class Words:
    """Word-overlap embedder so relevance is computed without e5."""

    name = "words-test"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: Sequence[str], kind: Kind) -> np.ndarray:
        import zlib

        m = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                m[i, zlib.crc32(w.encode()) % 256] += 1
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def _answer(sb: Any, text: str, onset: float) -> SpeechReport:
    t = sb.spaced(text, language="en", duration_s=WINDOW)
    t = sb.transcript([(w.text, w.start_s + onset, w.end_s + onset) for w in t.words],
                      language="en", duration_s=WINDOW)  # fmt: skip
    return SpeechReport(
        source_name="a.wav", model="whisper_small", transcript=t,
        asr_quality=AsrQuality(word_count=len(t.words)),
        speech_regions=(SpeechRegion(start_s=onset, end_s=t.words[-1].end_s),),
    )  # fmt: skip


def _silence(sb: Any) -> SpeechReport:
    t = sb.transcript([], language="en", duration_s=WINDOW)
    return SpeechReport(source_name="a.wav", model="whisper_small", transcript=t,
                        asr_quality=AsrQuality(word_count=0))  # fmt: skip


def test_reading_allowance_is_question_words_at_4_per_second():
    assert reading_allowance(SHORT, 4) == 0.75
    assert reading_allowance(LONG, 4) == 4.5
    assert reading_allowance(LONG, None) == 0.0  # no allowance configured: raw latency


def test_a_short_and_a_long_question_with_the_same_latency_score_differently(sb: Any):
    """Both answers start 5.5 s after their question appears. After the short question that is
    4.75 s of thinking (fails level 3's 4 s); after the long one, 1.0 s (meets level 4's 2 s)."""

    def thinking(question: str) -> float | None:
        fu = followup_report(
            [question], [_answer(sb, "because it was small enough", 5.5)], [WINDOW],
            language="en", embedder=None, reading_wps=4,
        )  # fmt: skip
        return fu.summary.latency_max_s

    short, long = thinking(SHORT), thinking(LONG)
    assert short is not None and long is not None
    assert short == pytest.approx(4.75) and long == pytest.approx(1.0)
    assert short > 4 >= 2 >= long  # level 3 fails after the short one; level 4 holds after the long


def test_bank_relevance_is_reported_not_scored_and_own_questions_score(sb: Any):
    q1, q2 = "Why did you use Gradio?", "How did you test the gaze model?"
    answers = [_answer(sb, "we used gradio because it is python", 1.0),
               _answer(sb, "we tested the gaze model on recordings", 1.0)]  # fmt: skip
    both_bank = followup_report([q1, q2], answers, [WINDOW] * 2, language="en",
                                embedder=Words(), relevance_scored=[False, False])  # fmt: skip
    assert RELEVANCE not in both_bank.measurements
    why = both_bank.not_measurable[RELEVANCE]
    assert why.startswith("measured ") and "bank" in why and "0." in why
    assert all(a.relevance is not None for a in both_bank.summary.answers)  # still reported
    mixed = followup_report([q1, q2], answers, [WINDOW] * 2, language="en",
                            embedder=Words(), relevance_scored=[True, False])  # fmt: skip
    own = mixed.summary.answers[0].relevance
    assert mixed.summary.relevance_min == own and RELEVANCE in mixed.measurements


def test_a_silent_bank_answer_still_scores_zero(sb: Any):
    fu = followup_report(["Why?"], [_silence(sb)], [WINDOW], language="en", embedder=Words(),
                         relevance_scored=[False])  # fmt: skip
    assert fu.summary.answers[0].status == "no_answer" and fu.summary.relevance_min == 0.0
    assert fu.measurements[LATENCY].value == pytest.approx(WINDOW - reading_allowance("Why?", None))


def _level(fu: Any, fake_parser: Any, speech_cfg: Any, fake_assessment: Any) -> Any:
    report = assess_report(fake_assessment.speech, parser=fake_parser, cfg=speech_cfg,
                           scenario="s2_project_defense", followups=fu)  # fmt: skip
    return next(d for d in report.assessment.dimensions if d.dimension == "followup_handling")


def test_with_bank_relevance_unscored_level_2_is_both_answers_answered(
    sb: Any, fake_parser: Any, speech_cfg: Any, fake_assessment: Any
):
    """Owner, 2026-11-25 (ADR 0022 open item 6): when relevance is information-only, level 2 is
    reached when every answer window holds an answer (status "answered"). Here both answers
    come too late for level 3 (5.25 s of thinking each)."""
    slow = followup_report(
        [SHORT, SHORT], [_answer(sb, "because it was small", 6.0)] * 2, [WINDOW] * 2,
        language="en", embedder=Words(), reading_wps=4, relevance_scored=[False, False],
    )  # fmt: skip
    assert slow.measurements[ANSWERED].value == 1.0
    assert len(slow.measurements[ANSWERED].evidence) == 2  # one per answer
    r = _level(slow, fake_parser, speech_cfg, fake_assessment)
    assert r.level == 2 and RELEVANCE in r.missing and ANSWERED in r.measured


def test_a_real_non_answer_to_a_bank_question_stays_level_1(
    sb: Any, fake_parser: Any, speech_cfg: Any, fake_assessment: Any
):
    fu = followup_report(
        [SHORT, SHORT], [_answer(sb, "because it was small", 6.0), _silence(sb)], [WINDOW] * 2,
        language="en", embedder=Words(), reading_wps=4, relevance_scored=[False, False],
    )  # fmt: skip
    assert fu.measurements[ANSWERED].value == 0.0
    assert fu.measurements[ANSWERED].evidence[0].answer == fu.summary.answers[1].index  # silent
    assert _level(fu, fake_parser, speech_cfg, fake_assessment).level == 1


def test_an_answer_that_cannot_be_judged_leaves_answered_missing_never_low(sb: Any):
    """An early stop or an untranscribed answer is not a non-answer (ADR 0019: missing, never
    low), and not "answered" either: the measure is missing."""
    early = SpeechReport(source_name="a.wav", model="whisper_small",
                         transcript=sb.transcript([], language="en", duration_s=10.0),
                         asr_quality=AsrQuality(word_count=0))  # fmt: skip
    fu = followup_report(
        [SHORT, SHORT], [_answer(sb, "because it was small", 1.0), early], [WINDOW] * 2,
        language="en", embedder=Words(), reading_wps=4, relevance_scored=[False, False],
    )  # fmt: skip
    assert fu.summary.answers[1].status == "incomplete"
    assert ANSWERED not in fu.measurements and ANSWERED in fu.not_measurable


def test_own_questions_still_need_relevance_for_level_2(
    sb: Any, fake_parser: Any, speech_cfg: Any, fake_assessment: Any
):
    off = followup_report(
        ["Why did you use Gradio?"] * 2, [_answer(sb, "the weather was nice today", 6.0)] * 2,
        [WINDOW] * 2, language="en", embedder=Words(), reading_wps=4,
    )  # fmt: skip
    assert off.measurements[ANSWERED].value == 1.0  # answered, but off the question
    assert _level(off, fake_parser, speech_cfg, fake_assessment).level == 1
