"""Tests for experiments/s1_live_capture/landmarks.py (fake engines; MediaPipe needs models)."""

import importlib.util
import itertools
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SPIKE = REPO / "experiments" / "s1_live_capture"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("s1_landmarks", SPIKE / "landmarks.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["s1_landmarks"] = module
    spec.loader.exec_module(module)
    return module


lm: Any = _load()


class Sink:
    def __init__(self) -> None:
        self.processed: list[tuple[float, bool | None, bool | None]] = []
        self.dropped = 0
        self.lock = threading.Lock()

    def frame_processed(
        self, *, process_ms: float, pose_found: bool | None = None, face_found: bool | None = None
    ) -> None:
        with self.lock:
            self.processed.append((process_ms, pose_found, face_found))

    def frame_dropped(self) -> None:
        with self.lock:
            self.dropped += 1


class FakeEngine:
    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.frames: list[np.ndarray] = []
        self.timestamps: list[int] = []
        self.closed = False

    def __call__(self, rgb: np.ndarray, timestamp_ms: int) -> Any:
        time.sleep(self.delay_s)
        self.frames.append(rgb)
        self.timestamps.append(timestamp_ms)
        return lm.LandmarkResult(pose_found=True, face_found=False)

    def close(self) -> None:
        self.closed = True


def _worker(engine: FakeEngine, sink: Sink, *, bgr: bool = False) -> Any:
    w = lm.LandmarkWorker(lambda: engine, sink, process_size=(64, 36), bgr=bgr)
    w.start()
    return w


def _frame(value: int = 0, shape: tuple[int, int] = (360, 640)) -> np.ndarray:
    f = np.zeros((*shape, 3), dtype=np.uint8)
    f[..., 0] = value  # channel 0 marks the frame
    return f


def test_modes_and_required_models(tmp_path: Path):
    assert lm.MODES == ("none", "pose-lite", "pose-full", "pose-lite+face")
    assert lm.required_models("none", tmp_path) == []
    names = [p.name for p in lm.required_models("pose-lite+face", tmp_path)]
    assert names == ["pose_landmarker_lite.task", "face_landmarker.task"]
    assert [p.name for p in lm.missing_models("pose-full", tmp_path)] == [
        "pose_landmarker_full.task"
    ]
    (tmp_path / "pose_landmarker_full.task").write_bytes(b"x")
    assert lm.missing_models("pose-full", tmp_path) == []


def test_resize_nearest_shape():
    out = lm.resize_nearest(_frame(), (160, 90))
    assert out.shape == (90, 160, 3)


def test_worker_processes_submitted_frames_and_reports_results():
    engine, sink = FakeEngine(), Sink()
    w = _worker(engine, sink)
    for i in range(3):
        w.submit(_frame(i))
        assert w.wait_idle(2.0)
    w.stop()
    assert len(sink.processed) == 3
    assert sink.dropped == 0
    assert all(p[1] is True and p[2] is False for p in sink.processed)
    assert engine.frames[0].shape == (36, 64, 3)  # downscaled to process_size
    assert engine.closed


def test_worker_drops_frames_while_busy_and_keeps_latest():
    engine, sink = FakeEngine(delay_s=0.2), Sink()
    w = _worker(engine, sink)
    w.submit(_frame(1))
    time.sleep(0.05)  # worker is now busy with frame 1
    for v in (2, 3, 4):
        w.submit(_frame(v))
    assert w.wait_idle(3.0)
    w.stop()
    assert sink.dropped == 2  # frames 2 and 3 replaced before they were processed
    assert [int(f[0, 0, 0]) for f in engine.frames] == [1, 4]


def test_worker_converts_bgr_to_rgb():
    engine, sink = FakeEngine(), Sink()
    w = _worker(engine, sink, bgr=True)
    w.submit(_frame(200))  # BGR: blue channel = 200
    assert w.wait_idle(2.0)
    w.stop()
    rgb = engine.frames[0]
    assert int(rgb[0, 0, 2]) == 200
    assert int(rgb[0, 0, 0]) == 0
    assert rgb.flags["C_CONTIGUOUS"]


def test_worker_timestamps_strictly_increase():
    engine, sink = FakeEngine(), Sink()
    w = _worker(engine, sink)
    for _ in range(5):
        w.submit(_frame())
        assert w.wait_idle(2.0)
    w.stop()
    ts = engine.timestamps
    assert all(b > a for a, b in itertools.pairwise(ts))


def test_worker_reports_engine_start_failure():
    def broken() -> Any:
        raise RuntimeError("model missing")

    w = lm.LandmarkWorker(broken, Sink(), process_size=(64, 36), bgr=False)
    w.start()
    w.submit(_frame())
    deadline = time.monotonic() + 2
    while w.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    w.stop()
    assert "model missing" in str(w.error)


def test_stand_in_engine_reports_no_landmarks():
    engine = lm.build_engine("none", REPO / "models")
    result = engine(np.zeros((36, 64, 3), dtype=np.uint8), 1)
    assert result == lm.LandmarkResult(pose_found=None, face_found=None)


@pytest.mark.needs_models
@pytest.mark.parametrize("mode", ["pose-lite", "pose-full", "pose-lite+face"])
def test_mediapipe_engine_runs_on_blank_frame(mode: str):
    if lm.missing_models(mode, REPO / "models"):
        pytest.skip("run scripts/download_models.py first")
    engine = lm.build_engine(mode, REPO / "models")
    try:
        result = engine(np.zeros((360, 640, 3), dtype=np.uint8), 1)
    finally:
        engine.close()
    assert result.pose_found is False
    assert result.face_found is (False if "face" in mode else None)
