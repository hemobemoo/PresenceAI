"""In-app recording buffer (P3-d, ADR 0024): timer, hard stop, completeness. A fake clock."""

from __future__ import annotations

import numpy as np
import pytest

from presenceai.live.recorder import RecordingBuffer


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _frame(seconds: float, rate: int = 48_000) -> np.ndarray:
    return np.full((1, int(seconds * rate)), 1000, dtype=np.int16)  # FastRTC's (1, n) int16


def _buffer(clock: Clock, limit: float = 10.0) -> RecordingBuffer:
    b = RecordingBuffer(limit_s=limit, extra_s=5.0, min_ratio=0.95, clock=clock)
    b.start()
    return b


def test_a_complete_recording_in_memory():
    clock = Clock()
    b = _buffer(clock)
    for _ in range(50):  # 50 frames of 0.1 s, arriving every 0.1 s
        clock.t += 0.1
        assert b.add(48_000, _frame(0.1))
    assert b.elapsed_s() == pytest.approx(4.9)  # since the first frame
    clock.t += 30  # Stop pressed late: idle time is not lost audio
    r = b.finish()
    assert r is not None and r.sample_rate == 48_000 and r.samples.dtype == np.float32
    assert r.received_s == pytest.approx(5.0) and r.complete and not r.hard_stopped
    assert r.ratio == pytest.approx(1.0)
    assert float(r.samples.max()) == pytest.approx(1000 / 32768)
    assert "1000" not in repr(r)


def test_dropped_frames_make_it_incomplete():
    clock = Clock()
    b = _buffer(clock)
    for i in range(50):
        clock.t += 0.1
        if i % 5 != 4:  # one frame in five lost on the way
            b.add(48_000, _frame(0.1))
    r = b.finish()
    assert r is not None and not r.complete and r.ratio < 0.95


def test_the_hard_stop_ignores_frames_after_limit_plus_extra():
    clock = Clock()
    b = _buffer(clock, limit=10.0)  # hard stop at 15 s
    assert b.add(48_000, _frame(0.1))
    clock.t += 15.0
    assert not b.add(48_000, _frame(0.1)) and b.hard_stopped
    assert b.elapsed_s() == pytest.approx(15.0)
    r = b.finish()
    assert r is not None and r.hard_stopped


def test_nothing_recorded_and_not_started():
    clock = Clock()
    b = RecordingBuffer(limit_s=10, extra_s=5, min_ratio=0.95, clock=clock)
    assert not b.add(48_000, _frame(0.1))  # not started
    b.start()
    assert b.finish() is None
    assert not b.active
