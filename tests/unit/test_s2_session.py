"""S2 session on one stream (P3-e): marks at sample positions, windows, retry, hard stop."""

from __future__ import annotations

import numpy as np
import pytest

from presenceai.live.recorder import RecordingBuffer
from presenceai.live.s2_session import Phase, S2Session, Take

RATE = 16_000
S = RATE  # samples per second


def _session(**kw: float) -> S2Session:
    s = S2Session(("Q1?", "Q2?"), rate=RATE, explanation_limit_s=90, extra_s=60, window_s=60, **kw)  # type: ignore[arg-type]
    s.begin()
    return s


def test_explanation_then_two_answers_by_sample_marks():
    s = _session()
    assert s.phase is Phase.EXPLANATION and s.tick(100 * S) is None  # not cut at 90 s
    s.next_question(110 * S)  # "Next: question 1" at 110 s of received audio
    assert s.explanation_end == 110 * S and s.phase is Phase.ANSWER and s.current == 0
    assert s.window_elapsed_s(140 * S) == 30.0
    s.done_answering(150 * S)  # answered early
    assert s.takes[0] == Take(0, 110 * S, 150 * S, full=False)
    assert s.current == 1 and s.window_start() == 150 * S  # question 2 appears at once
    take = s.tick(210 * S)  # the whole 60 s window
    assert take == Take(1, 150 * S, 210 * S, full=True)
    s.checked(silent=False)
    assert s.phase is Phase.DONE and s.takes[1] == take


def test_the_explanation_hard_stop_opens_question_1():
    s = _session()
    s.tick(150 * S - 1)
    assert s.phase is Phase.EXPLANATION
    s.tick(150 * S + 37)  # 90 + 60 s reached between two ticks: the mark is exactly 150 s
    assert s.explanation_hard_stopped and s.explanation_end == 150 * S
    assert s.phase is Phase.ANSWER and s.window_start() == 150 * S


def test_one_retry_for_a_silent_window_then_no_answer():
    s = _session()
    s.next_question(30 * S)
    first = s.tick(90 * S)
    assert first is not None
    s.checked(silent=True)
    assert s.phase is Phase.RETRY_OFFER and 0 not in s.takes
    s.retry(95 * S)  # the silent take is discarded; the same question reopens
    assert s.phase is Phase.ANSWER and s.current == 0 and s.attempts(0) == 2
    second = s.tick(155 * S)
    assert second == Take(0, 95 * S, 155 * S, full=True)
    s.checked(silent=True)  # silent again: no second retry
    assert s.takes[0] == second and s.current == 1 and s.window_start() == 155 * S


def test_keeping_a_silent_window_opens_the_next_question_when_it_appears():
    s = _session()
    s.next_question(30 * S)
    s.tick(90 * S)
    s.checked(silent=True)
    s.keep_silent(97 * S)  # the user read the offer for 7 s
    assert s.takes[0] == Take(0, 30 * S, 90 * S, full=True) and s.window_start() == 97 * S


def test_stream_limit_covers_the_whole_session_with_retries():
    s = _session()
    assert s.stream_limit_s == 90 + 60 + 2 * 2 * 60 + 60  # 450 s
    with pytest.raises(ValueError):
        s.done_answering(10)  # not answering yet


def test_buffer_positions_and_slices_are_in_received_samples():
    b = RecordingBuffer(limit_s=10, extra_s=0, min_ratio=0.95, clock=lambda: 0.0)
    b.start()
    for k in range(5):
        b.add(RATE, np.full((1, 1600), k * 1000, dtype=np.int16))
    assert b.position() == 8000 and b.rate == RATE
    part = b.slice(1600, 4800)
    assert part.shape == (3200,) and part[0] == pytest.approx(1000 / 32768)
    b.set_limits(limit_s=450, extra_s=0)
    assert b.hard_stop_s == 450
