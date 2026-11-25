"""S2 thresholds after the owner's first real run (2026-11-25): a reading allowance for latency,
and relevance from bank questions reported but not scored."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from presenceai.knowledge.embed import Kind
from presenceai.pipeline.assess import assess_report
from presenceai.pipeline.followup import LATENCY, RELEVANCE, followup_report, reading_allowance
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


def test_with_bank_relevance_unscored_level_2_cannot_be_reached(
    sb: Any, fake_parser: Any, speech_cfg: Any, fake_assessment: Any
):
    """Consequence for the owner to decide (ADR 0022): level 2's only condition is relevance, so
    with bank relevance unscored follow-up handling is 1, 3 or 4, never 2."""
    slow = followup_report(
        [SHORT, SHORT], [_answer(sb, "because it was small", 6.0)] * 2, [WINDOW] * 2,
        language="en", embedder=Words(), reading_wps=4, relevance_scored=[False, False],
    )  # fmt: skip
    report = assess_report(fake_assessment.speech, parser=fake_parser, cfg=speech_cfg,
                           scenario="s2_project_defense", followups=slow)  # fmt: skip
    r = next(d for d in report.assessment.dimensions if d.dimension == "followup_handling")
    assert r.level == 1 and RELEVANCE in r.missing
