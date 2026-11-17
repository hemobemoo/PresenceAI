"""evaluation/e2.py: E2 scoring rules fixed in ADR 0014 §6 before any run."""

from typing import Any

import pytest

from presenceai.evalconfig import load_evaluation
from presenceai.evaluation import e2
from presenceai.schemas.assessment import Confidence
from presenceai.schemas.c1 import ClipTruth, TruePause
from presenceai.schemas.speech import PauseEvent, PausePosition

EV = load_evaluation()
CFG = EV.e2


def tp(start: float, dur: float, pos: str = "sentence_boundary", src: str = "inserted") -> Any:
    return TruePause(start_s=start, end_s=start + dur, position=pos, source=src)  # type: ignore[arg-type]


def dp(start: float, dur: float, pos: str = "sentence_boundary") -> PauseEvent:
    return PauseEvent(
        start_s=start, end_s=start + dur, position=PausePosition(pos), confidence=Confidence.HIGH
    )


def truth(pauses: tuple[TruePause, ...], wpm: float = 150.0, fillers: int = 0) -> ClipTruth:
    return ClipTruth(
        clip_id="s01_r0", script_id="s01", voice="af_heart", render=0, sample_rate=24_000,
        duration_s=60.0, words=120, fillers=fillers, speech_onset_s=0.5, speech_offset_s=59.5,
        speech_spans=((0.5, 59.5),), wpm=wpm, articulation_wpm=wpm + 20, target_wpm=wpm,
        speed=1.0, attempts=1, pauses=pauses, internal_silences=(), text="x",
    )  # fmt: skip


# --- matching ---------------------------------------------------------------------------------


def test_match_needs_overlap_and_onset_within_020_s():
    true = (tp(10.0, 1.0), tp(20.0, 1.0), tp(30.0, 1.0))
    det = (dp(10.15, 0.9), dp(20.3, 0.8), dp(31.5, 0.5))  # ok; onset 0.30 off; no overlap
    m = e2.match(true, det, CFG)
    assert [(x.true_index, x.detected_index) for x in m] == [(0, 0)]
    assert m[0].onset_error_s == pytest.approx(0.15)
    assert m[0].duration_error_s == pytest.approx(-0.1)


def test_match_is_one_to_one_largest_overlap_first():
    true = (tp(10.0, 1.0),)
    det = (dp(10.05, 0.3), dp(10.1, 0.85))
    m = e2.match(true, det, CFG)
    assert [(x.true_index, x.detected_index) for x in m] == [(0, 1)]


def test_true_pauses_shorter_than_025_s_are_not_scored():
    s = e2.score_clip(truth((tp(5.0, 0.2),)), (dp(5.0, 0.2),), 150.0, "test", "clean", EV)
    assert s.true_pauses == 0


# --- F1 with and without the exclusion zone (owner's point 5) ----------------------------------


def test_exclusion_zone_counts_and_both_f1s():
    true = (
        tp(10.0, 0.45),  # in the 0.5 s zone (0.40-0.60): neither way at 0.5
        tp(20.0, 0.80),  # clear positive at 0.5
        tp(30.0, 0.30),  # clear negative at 0.5
        tp(40.0, 1.05),  # in the 1.0 s zone
    )
    det = (dp(10.0, 0.52), dp(20.0, 0.75), dp(30.0, 0.55), dp(40.0, 0.95))
    s = e2.score_clip(truth(true), det, 150.0, "test", "clean", EV)
    at = {t.threshold_s: t for t in s.thresholds}
    # 0.5 s: excluded 1 (0.45). 0.80->0.75 TP; 0.30->0.55 FP; 1.05->0.95 TP
    assert at[0.5].excluded == 1
    assert (at[0.5].tp, at[0.5].fp, at[0.5].fn) == (2, 1, 0)
    # without exclusion the 0.45 true pause (negative) matched to 0.52 (positive) is a FP
    assert (at[0.5].tp_all, at[0.5].fp_all, at[0.5].fn_all) == (2, 2, 0)
    # 1.0 s: 1.05 is excluded; without exclusion it is a miss (detected 0.95)
    assert at[1.0].excluded == 1
    assert (at[1.0].tp, at[1.0].fp, at[1.0].fn) == (0, 0, 0)
    assert (at[1.0].tp_all, at[1.0].fp_all, at[1.0].fn_all) == (0, 0, 1)


def test_unmatched_detections_and_misses():
    true = (tp(10.0, 1.5),)
    det = (dp(50.0, 1.2),)
    s = e2.score_clip(truth(true), det, 150.0, "test", "clean", EV)
    at = {t.threshold_s: t for t in s.thresholds}
    assert (at[1.0].tp, at[1.0].fp, at[1.0].fn) == (0, 1, 1)


# --- wpm and duration -------------------------------------------------------------------------


def test_wpm_error_and_duration_error():
    true = (tp(10.0, 1.0), tp(20.0, 0.6))
    det = (dp(10.05, 0.9), dp(20.0, 0.66))
    s = e2.score_clip(truth(true, wpm=150.0), det, 156.0, "test", "clean", EV)
    assert s.wpm_error_pct == pytest.approx(4.0)
    assert sorted(s.duration_abs_errors_s) == pytest.approx([0.06, 0.1])


def test_a_clip_without_a_measured_rate_counts_as_a_100_percent_error():
    s = e2.score_clip(truth(()), (), None, "test", "clean", EV)
    assert s.wpm_error_pct == 100.0


def test_summary_judges_wpm_on_all_clips_with_filler_breakdown():
    a = e2.score_clip(truth((), wpm=150.0, fillers=0), (), 153.0, "test", "clean", EV)  # 2 %
    b = e2.score_clip(truth((), wpm=150.0, fillers=2), (), 141.0, "test", "clean", EV)  # 6 %
    summ = e2.summarise([a, b], EV)
    assert summ.wpm_mape_pct == pytest.approx(4.0)
    assert summ.wpm_mape_no_fillers_pct == pytest.approx(2.0)
    assert summ.wpm_mape_fillers_pct == pytest.approx(6.0)
    assert summ.passes["wpm"] is True


def test_summary_pass_flags_use_adr_0009_targets():
    true = tuple(tp(10.0 * k, 1.5) for k in range(1, 11))
    det = tuple(dp(10.0 * k + 0.05, 1.3) for k in range(1, 10))  # one miss, durations 0.2 off
    s = e2.score_clip(truth(true), det, 150.0, "test", "clean", EV)
    summ = e2.summarise([s], EV)
    assert summ.f1[1.0] == pytest.approx(2 * 9 / (2 * 9 + 1))  # 0.947 >= 0.90
    assert summ.passes["f1_1.0"] is True
    assert summ.duration_mae_s == pytest.approx(0.2)
    assert summ.passes["duration"] is False
    assert summ.passed is False


def test_summary_refuses_to_judge_development_clips():
    s = e2.score_clip(truth(()), (), 150.0, "dev", "clean", EV)
    summ = e2.summarise([s], EV)
    assert summ.passed is None  # dev runs are for tuning only


# --- pause position (ADR 0011) ------------------------------------------------------------


def test_position_share_and_error_rate():
    true = (tp(10.0, 1.0, "mid_sentence"), tp(20.0, 1.0, "sentence_boundary"),
            tp(30.0, 1.0, "clause_boundary"))  # fmt: skip
    det = (dp(10.0, 1.0, "sentence_boundary"), dp(20.0, 1.0, "sentence_boundary"),
           dp(30.0, 1.0, "unknown"))  # fmt: skip
    s = e2.score_clip(truth(true), det, 150.0, "test", "clean", EV)
    summ = e2.summarise([s], EV)
    assert summ.true_mid_share == pytest.approx(1 / 3)
    assert summ.detected_mid_share == pytest.approx(0.0)
    assert summ.position_error_rate == pytest.approx(2 / 3)
    assert summ.position_unknown_share == pytest.approx(1 / 3)


def test_kokoro_pauses_are_counted_in_the_summary():
    true = (tp(10.0, 1.0), tp(20.0, 0.3, "mid_sentence", "kokoro"))
    s = e2.score_clip(truth(true), (), 150.0, "test", "clean", EV)
    assert e2.summarise([s], EV).kokoro_pauses == 1


def test_headline_position_measure_ignores_sentence_clause_swaps():
    """ADR 0014 §6 (owner, before the test run): boundary vs mid-sentence errors, both ways."""
    true = (
        tp(10.0, 1.0, "sentence_boundary"), tp(20.0, 1.0, "clause_boundary"),
        tp(30.0, 1.0, "mid_sentence"), tp(40.0, 1.0, "mid_sentence"),
        tp(50.0, 1.0, "sentence_boundary"), tp(60.0, 1.0, "mid_sentence"),
    )  # fmt: skip
    det = (
        dp(10.0, 1.0, "clause_boundary"),  # swap: not an error here
        dp(20.0, 1.0, "mid_sentence"),  # boundary -> mid
        dp(30.0, 1.0, "sentence_boundary"),  # mid -> boundary
        dp(40.0, 1.0, "mid_sentence"),  # right
        dp(50.0, 1.0, "sentence_boundary"),  # right
        dp(60.0, 1.0, "unknown"),  # left out, reported apart
    )  # fmt: skip
    summ = e2.summarise([e2.score_clip(truth(true), det, 150.0, "test", "clean", EV)], EV)
    assert summ.mid_as_boundary_rate == pytest.approx(1 / 2)
    assert summ.boundary_as_mid_rate == pytest.approx(1 / 3)
    assert summ.position_table["sentence_boundary"]["clause_boundary"] == 1
    assert summ.position_table["mid_sentence"]["unknown"] == 1
    assert summ.position_error_rate == pytest.approx(4 / 6)  # extra information


def test_noise_conditions_get_no_verdict():
    """ADR 0014 §7: the noise conditions are descriptive, with no pass target."""
    s = e2.score_clip(truth(()), (), 150.0, "test", "snr10", EV)
    summ = e2.summarise([s], EV)
    assert summ.passed is None and summ.passes == {}
