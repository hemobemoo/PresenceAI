"""Tests for experiments/s1_live_capture/sampling.py (which frames go to landmarks)."""

import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SPIKE = REPO / "experiments" / "s1_live_capture"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("s1_sampling", SPIKE / "sampling.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["s1_sampling"] = module
    spec.loader.exec_module(module)
    return module


sm: Any = _load()

TARGET = 15.0
TOL = 0.25


def camera(
    fps: float, seconds: float, *, start: float = 0.0, jitter_ms: float = 0.0, seed: int = 1
) -> list[float]:
    """Frame arrival times for a camera at ``fps``, with optional uniform jitter."""
    rng = random.Random(seed)  # noqa: S311 - deterministic test jitter, not crypto
    n = round(fps * seconds)
    return [start + i / fps + rng.uniform(-jitter_ms, jitter_ms) / 1000 for i in range(n)]


def taken(sampler: Any, times: list[float]) -> list[float]:
    return [t for t in times if sampler.take(t)]


def per_second(times: list[float]) -> Counter[int]:
    return Counter(int(t) for t in times)


def max_run_of_skips(sampler: Any, times: list[float]) -> int:
    run = worst = 0
    for t in times:
        if sampler.take(t):
            run = 0
        else:
            run += 1
            worst = max(worst, run)
    return worst


@pytest.mark.parametrize("jitter_ms", [0.0, 5.0])
def test_30fps_camera_gives_about_every_2nd_frame(jitter_ms: float):
    times = camera(30, 10, jitter_ms=jitter_ms)
    got = taken(sm.IntervalSampler(TARGET, TOL), times)
    assert 145 <= len(got) <= 155  # ~15 fps over 10 s
    counts = per_second(got)
    assert all(14 <= counts[s] <= 16 for s in range(10))
    # every 2nd frame: never two skips in a row
    assert max_run_of_skips(sm.IntervalSampler(TARGET, TOL), times) <= 1


def test_15fps_camera_gives_every_frame():
    times = camera(15, 10, jitter_ms=5.0)
    assert taken(sm.IntervalSampler(TARGET, TOL), times) == times


@pytest.mark.parametrize("fps", [5.0, 10.0, 12.0])
def test_camera_slower_than_target_gives_every_frame(fps: float):
    times = camera(fps, 10)
    assert taken(sm.IntervalSampler(TARGET, TOL), times) == times


def test_no_catch_up_burst_after_a_stall():
    # 2 s at 30 fps, a 1 s stall, then 30 fps again: the frames after the stall are sampled at
    # the normal rate, not all taken to "make up" for the gap.
    times = camera(30, 2) + camera(30, 2, start=3.0)
    sampler = sm.IntervalSampler(TARGET, TOL)
    got = taken(sampler, times)
    after = [t for t in got if 3.0 <= t < 3.5]
    assert len(after) <= 9  # ~15 fps over 0.5 s, plus the restart frame


def test_camera_rate_changes_during_the_run():
    # 30 fps -> 15 fps (lights dimmed) -> 30 fps, 4 s each
    times = camera(30, 4, jitter_ms=5) + camera(15, 4, start=4.0, jitter_ms=5)
    times += camera(30, 4, start=8.0, jitter_ms=5)
    got = taken(sm.IntervalSampler(TARGET, TOL), times)
    counts = per_second(got)
    assert all(14 <= counts[s] <= 16 for s in range(12)), counts
    slow = [t for t in times if 4.0 <= t < 8.0]
    assert all(t in got for t in slow)  # every frame while the camera is at 15 fps


def test_interval_sampler_counts_received_frames():
    sampler = sm.IntervalSampler(TARGET, TOL)
    taken(sampler, camera(30, 1))
    assert sampler.count == 30


def test_interval_sampler_rejects_bad_settings():
    with pytest.raises(ValueError):
        sm.IntervalSampler(0, TOL)
    with pytest.raises(ValueError):
        sm.IntervalSampler(TARGET, 0.5)


def test_every_nth_sampler():
    sampler = sm.EveryNthSampler(3)
    assert [sampler.take(0.0) for _ in range(7)] == [True, False, False, True, False, False, True]
    assert sampler.count == 7


def test_choose_sampling():
    kwargs = {"stand_in_every": 1, "target_fps": TARGET, "tolerance": TOL}
    spec = sm.choose_sampling(mode="pose-lite+face", process_every_override=None, **kwargs)
    assert isinstance(spec.make(), sm.IntervalSampler)
    assert spec.describe() == "sampling=time target_fps=15 tolerance=0.25"
    spec = sm.choose_sampling(mode="none", process_every_override=None, **kwargs)
    assert spec.describe() == "process_every=1"
    spec = sm.choose_sampling(mode="pose-lite", process_every_override=2, **kwargs)
    assert isinstance(spec.make(), sm.EveryNthSampler)
    assert spec.describe() == "process_every=2"
