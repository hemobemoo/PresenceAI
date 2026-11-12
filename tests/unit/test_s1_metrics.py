"""Tests for the S1 spike metrics (experiments/s1_live_capture/metrics.py)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SPIKE = REPO / "experiments" / "s1_live_capture" / "metrics.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("s1_metrics", SPIKE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["s1_metrics"] = module
    spec.loader.exec_module(module)
    return module


m: Any = _load()


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _recorder(clock: FakeClock, warmup_s: float = 2.0) -> Any:
    return m.RunRecorder("run1", "fastrtc", warmup_s=warmup_s, clock=clock, cpu=lambda: 12.5)


# ---------------------------------------------------------------- percentile / slope


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        ([], 0.5, None),
        ([7.0], 0.95, 7.0),
        ([1.0, 2.0, 3.0, 4.0], 0.5, 2.5),
        ([0.0, 10.0], 0.95, 9.5),
        ([5.0, 1.0, 3.0], 0.0, 1.0),
        ([5.0, 1.0, 3.0], 1.0, 5.0),
    ],
)
def test_percentile(values: list[float], q: float, expected: float | None):
    assert m.percentile(values, q) == (pytest.approx(expected) if expected is not None else None)


def test_slope_constant_is_zero():
    assert m.slope_per_min([0, 1000, 2000, 3000], [100, 100, 100, 100]) == pytest.approx(0.0)


def test_slope_growing_is_positive():
    # +10 ms per second of run time = +600 ms per minute
    ts_ms = [0, 1000, 2000, 3000]
    assert m.slope_per_min(ts_ms, [100, 110, 120, 130]) == pytest.approx(600.0)


def test_slope_needs_two_distinct_points():
    assert m.slope_per_min([5], [1]) is None
    assert m.slope_per_min([5, 5], [1, 2]) is None


# ---------------------------------------------------------------- meter text


def test_meter_round_trip():
    text = m.format_meter(level=0.25, seq=17, t_server_ms=1731412345678.125)
    reading = m.parse_meter(text)
    assert reading.seq == 17
    assert reading.level == pytest.approx(0.25)
    assert reading.t_server_ms == pytest.approx(1731412345678.125)


@pytest.mark.parametrize("text", ["", "waiting", "level=x seq=1 t_server_ms=2"])
def test_meter_parse_rejects_garbage(text: str):
    assert m.parse_meter(text) is None


# ---------------------------------------------------------------- recorder


def test_recorder_buckets_complete_seconds_only():
    clock = FakeClock()
    rec = _recorder(clock)
    rec.video_frame(processed=True, process_ms=4.0)  # starts the run at t=1000.0
    clock.t = 1000.4
    rec.video_frame(processed=False)
    rec.audio_chunk(0.02)
    rec.meter_published()
    clock.t = 1001.2
    rec.video_frame(processed=True, process_ms=6.0)
    clock.t = 1001.3
    rows = rec.flush()
    assert len(rows) == 1  # second 1 is still open
    row = rows[0]
    assert row.t_s == 0
    assert row.video_frames_received == 2
    assert row.video_frames_processed == 1
    assert row.audio_chunks == 1
    assert row.audio_seconds == pytest.approx(0.02)
    assert row.meter_updates == 1
    assert row.process_ms_p50 == pytest.approx(4.0)
    assert row.cpu_percent == pytest.approx(12.5)
    assert row.warmup is True


def test_recorder_emits_gap_seconds_and_warmup_flag():
    clock = FakeClock()
    rec = _recorder(clock, warmup_s=2.0)
    rec.audio_chunk(0.02)
    clock.t = 1003.5
    rec.audio_chunk(0.02)
    clock.t = 1004.2
    rec.audio_chunk(0.02)
    clock.t = 1004.5
    rows = rec.flush()
    assert [r.t_s for r in rows] == [0, 1, 2, 3]
    assert [r.warmup for r in rows] == [True, True, False, False]
    assert rows[1].audio_chunks == 0  # a stall inside the run is kept
    assert rows[1].process_ms_p50 is None
    assert rows[3].audio_chunks == 1
    assert rec.flush() == []  # nothing new


def test_recorder_drops_partial_last_second_and_trailing_idle():
    clock = FakeClock()
    rec = _recorder(clock)
    rec.audio_chunk(0.02)
    clock.t = 1001.1
    rec.audio_chunk(0.02)
    clock.t = 1002.3
    rec.audio_chunk(0.02)  # last media event: second 2 was cut short by Stop
    clock.t = 1003.0
    rec.meter_published()  # meter values after media stops are not activity
    clock.t = 1009.0
    assert [r.t_s for r in rec.flush()] == [0, 1]
    clock.t = 1020.0
    assert rec.flush() == []


def test_recorder_does_nothing_before_first_event():
    clock = FakeClock()
    rec = _recorder(clock)
    clock.t = 1010.0
    assert rec.flush() == []


def test_recorder_sink_receives_rows():
    clock = FakeClock()
    seen: list[Any] = []
    rec = m.RunRecorder("r", "fallback", warmup_s=0, clock=clock, cpu=lambda: 0.0, sink=seen.append)
    rec.audio_chunk(0.1)
    clock.t = 1001.5
    rec.audio_chunk(0.1)
    clock.t = 1002.1
    rec.audio_chunk(0.1)
    clock.t = 1002.5
    rec.flush()
    assert [r.t_s for r in seen] == [0, 1]


# ---------------------------------------------------------------- CSV


def test_csv_round_trip(tmp_path: Path):
    clock = FakeClock()
    rec = _recorder(clock)
    rec.video_frame(processed=True, process_ms=3.0)
    clock.t = 1002.5
    rec.audio_chunk(0.02)
    clock.t = 1003.0
    rows = rec.flush()
    assert len(rows) == 2
    path = tmp_path / "server.csv"
    m.write_csv(path, rows)
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(m.SECOND_FIELDS)
    assert m.read_second_rows(path) == rows


def test_csv_append_keeps_single_header(tmp_path: Path):
    path = tmp_path / "a.csv"
    row = m.SecondRow("r", "fastrtc", 0, True, 1, 1, 1, 0.02, 1, 2.0, 2.0, 5.0)
    m.write_csv(path, [row], append=True)
    m.write_csv(path, [row], append=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


# ---------------------------------------------------------------- summarise


def _rows(fps: list[int], warmup_s: int = 5) -> list[Any]:
    return [
        m.SecondRow(
            run_id="r",
            transport="fastrtc",
            t_s=i,
            warmup=i < warmup_s,
            video_frames_received=f,
            video_frames_processed=f,
            audio_chunks=50,
            audio_seconds=1.0,
            meter_updates=10,
            process_ms_p50=3.0,
            process_ms_max=5.0,
            cpu_percent=20.0 + i,
        )
        for i, f in enumerate(fps)
    ]


def _lat(meter_ms: list[float], start_ms: float = 0.0, step_ms: float = 100.0) -> list[Any]:
    return [
        m.LatencySample(
            "fastrtc", i, 1e12 + start_ms + i * step_ms, 1e12 + start_ms + i * step_ms + v
        )
        for i, v in enumerate(meter_ms)
    ]


def _summ(rows: list[Any], lat: list[Any], **kw: Any) -> Any:
    args: dict[str, Any] = {"min_fps": 8.0, "budget_ms": 300.0, "warmup_s": 5.0, "min_share": 0.95}
    args.update(kw)
    return m.summarise("fastrtc", rows, lat, **args)


def test_summary_pass():
    rows = _rows([0, 1, 2, 3, 4] + [10] * 40)  # warm-up seconds have low fps: ignored
    lat = _lat([900.0] * 50 + [50.0] * 200)  # first 5 s (50 samples) are warm-up
    s = _summ(rows, lat)
    assert s.seconds_total == 45
    assert s.seconds_used == 40
    assert s.fps_p50 == pytest.approx(10)
    assert s.fps_min == pytest.approx(10)
    assert s.share_fps_ok == pytest.approx(1.0)
    assert s.meter_ms_p95 == pytest.approx(50.0)
    assert s.meter_ms_max == pytest.approx(50.0)
    assert s.audio_realtime_ratio == pytest.approx(1.0)
    assert s.audio_chunk_rate_p50 == pytest.approx(50)
    assert s.cpu_max == pytest.approx(64.0)
    assert s.passed is True


def test_summary_fps_share_boundary():
    # 20 used seconds, one below min_fps -> share 0.95 -> still PASS (>=)
    s = _summ(_rows([10] * 5 + [10] * 19 + [7]), _lat([50.0] * 300))
    assert s.share_fps_ok == pytest.approx(0.95)
    assert s.passed is True
    # two below -> 0.90 -> FAIL
    s = _summ(_rows([10] * 5 + [10] * 18 + [7, 7]), _lat([50.0] * 300))
    assert s.passed is False
    assert "fps" in s.reason


def test_summary_latency_budget_is_strict_and_max_not_used():
    ok = _lat([50.0] * 99 + [5000.0], start_ms=0.0, step_ms=100.0)
    # warm-up excludes the first 50 samples (5 s); one 5 s outlier among 50 used: p95 still low
    s = _summ(_rows([10] * 30), ok)
    assert s.meter_ms_max == pytest.approx(5000.0)
    assert s.passed is True
    at_budget = _lat([300.0] * 100)
    s = _summ(_rows([10] * 30), at_budget)
    assert s.passed is False  # p95 must be < budget
    assert "meter" in s.reason


def test_summary_lag_trend():
    lat = _lat([float(i) for i in range(200)], step_ms=100.0)  # +1 ms per 100 ms
    s = _summ(_rows([10] * 30), lat)
    assert s.lag_slope_ms_per_min == pytest.approx(600.0)


def test_summary_without_data_fails():
    s = _summ([], [])
    assert s.passed is False
    assert s.seconds_used == 0
    assert s.meter_ms_p95 is None


def test_thresholds_come_from_live_config():
    min_fps, budget_ms = m.load_thresholds(REPO / "configs" / "live.yaml")
    assert min_fps == 8
    assert budget_ms == 300


def test_spike_config_loads():
    cfg = m.load_spike_config(REPO / "experiments" / "configs" / "s1_live_capture.yaml")
    assert cfg.warmup_s == 5
    assert cfg.min_share_fps_ok == pytest.approx(0.95)
    assert cfg.echo_size == (160, 90)
