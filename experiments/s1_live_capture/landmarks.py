"""Spike S1 landmark step: MediaPipe Pose (lite/full) and Face Landmarker in VIDEO mode.

`LandmarkWorker` runs the engine on its own thread with a single "latest frame" slot. It never
blocks the caller (FastRTC calls `video_receive` on its event loop): if a new frame arrives while
the previous one is still waiting, the waiting one is dropped and counted. Processed fps is
therefore what the laptop can actually sustain.

Face Landmarker runs with output_face_blendshapes=False: geometry only. Only found / not found
counts leave this module; landmarks themselves are not stored.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt

MODES = ("none", "pose-lite", "pose-full", "pose-lite+face")
_MODEL_FILES: dict[str, tuple[str, ...]] = {
    "none": (),
    "pose-lite": ("pose_landmarker_lite.task",),
    "pose-full": ("pose_landmarker_full.task",),
    "pose-lite+face": ("pose_landmarker_lite.task", "face_landmarker.task"),
}


class LandmarkResult(NamedTuple):
    pose_found: bool | None  # None = pose model not running in this mode
    face_found: bool | None


class Engine(Protocol):
    def __call__(self, rgb: npt.NDArray[np.uint8], timestamp_ms: int) -> LandmarkResult: ...
    def close(self) -> None: ...


class Sink(Protocol):
    def frame_processed(
        self, *, process_ms: float, pose_found: bool | None = None, face_found: bool | None = None
    ) -> None: ...
    def frame_dropped(self) -> None: ...


def required_models(mode: str, models_dir: Path) -> list[Path]:
    if mode not in _MODEL_FILES:
        raise ValueError(f"unknown landmarks mode {mode!r}; choose from {MODES}")
    return [models_dir / name for name in _MODEL_FILES[mode]]


def missing_models(mode: str, models_dir: Path) -> list[Path]:
    return [p for p in required_models(mode, models_dir) if not p.is_file()]


def resize_nearest(frame: npt.NDArray[Any], size: tuple[int, int]) -> npt.NDArray[Any]:
    """Nearest-neighbour resize to (width, height) with NumPy indexing only."""
    width, height = size
    rows = np.linspace(0, frame.shape[0] - 1, height).astype(np.intp)
    cols = np.linspace(0, frame.shape[1] - 1, width).astype(np.intp)
    return frame[np.ix_(rows, cols)]


class StandIn:
    """The first S1 runs' stand-in: grey mean of the frame, no landmarks."""

    def __init__(self) -> None:
        self.last_mean = 0.0

    def __call__(self, rgb: npt.NDArray[np.uint8], timestamp_ms: int) -> LandmarkResult:
        self.last_mean = float(rgb.mean())
        return LandmarkResult(None, None)

    def close(self) -> None:
        pass


class MediaPipeEngine:
    """Pose Landmarker (+ optional Face Landmarker), VIDEO running mode, one person."""

    def __init__(self, mode: str, models_dir: Path) -> None:
        import mediapipe as mp  # imported lazily: unit tests run without MediaPipe models

        missing = missing_models(mode, models_dir)
        if missing:
            names = ", ".join(p.name for p in missing)
            raise FileNotFoundError(
                f"missing {names}; run `uv run python scripts/download_models.py --pin`"
            )
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        video = vision.RunningMode.VIDEO
        pose_file, *rest = required_models(mode, models_dir)
        self._mp = mp
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base(model_asset_path=str(pose_file)),
                running_mode=video,
                num_poses=1,
            )
        )
        self._face = None
        if rest:
            self._face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=base(model_asset_path=str(rest[0])),
                    running_mode=video,
                    num_faces=1,
                    output_face_blendshapes=False,  # geometry only for this spike
                    output_facial_transformation_matrixes=False,
                )
            )

    def __call__(self, rgb: npt.NDArray[np.uint8], timestamp_ms: int) -> LandmarkResult:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        pose = self._pose.detect_for_video(image, timestamp_ms)
        face_found = None
        if self._face is not None:
            face = self._face.detect_for_video(image, timestamp_ms)
            face_found = bool(face.face_landmarks)
        return LandmarkResult(bool(pose.pose_landmarks), face_found)

    def close(self) -> None:
        self._pose.close()
        if self._face is not None:
            self._face.close()


def build_engine(mode: str, models_dir: Path) -> Engine:
    required_models(mode, models_dir)  # validates the mode
    return StandIn() if mode == "none" else MediaPipeEngine(mode, models_dir)


class LandmarkWorker:
    """Runs an engine on its own thread; keeps only the latest submitted frame."""

    def __init__(
        self,
        factory: Callable[[], Engine],
        sink: Sink,
        *,
        process_size: tuple[int, int],
        bgr: bool,
    ) -> None:
        self._factory = factory
        self._sink = sink
        self._size = process_size
        self._bgr = bgr
        self._cond = threading.Condition()
        self._pending: npt.NDArray[np.uint8] | None = None
        self._busy = False
        self._stopped = False
        self._last_ts = 0
        self._thread = threading.Thread(target=self._run, name="s1-landmarks", daemon=True)
        self.error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout_s)

    def submit(self, frame: npt.NDArray[np.uint8]) -> None:
        """Hand a frame to the worker; replaces (and counts as dropped) a frame still waiting."""
        with self._cond:
            if self._pending is not None:
                self._sink.frame_dropped()
            self._pending = frame
            self._cond.notify_all()

    def wait_idle(self, timeout_s: float) -> bool:
        """For tests: wait until nothing is pending or being processed."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._pending is not None or self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def _next_timestamp(self) -> int:
        # MediaPipe VIDEO mode needs strictly increasing timestamps.
        self._last_ts = max(self._last_ts + 1, int(time.monotonic() * 1000))
        return self._last_ts

    def _run(self) -> None:
        try:
            engine = self._factory()
        except BaseException as exc:  # reported to the app, which logs it
            self.error = exc
            return
        try:
            while True:
                with self._cond:
                    while self._pending is None and not self._stopped:
                        self._cond.wait()
                    if self._stopped:
                        return
                    frame, self._pending = self._pending, None
                    if frame is None:
                        continue
                    self._busy = True
                t0 = time.perf_counter()
                small = resize_nearest(frame, self._size)
                rgb = np.ascontiguousarray(small[..., ::-1] if self._bgr else small)
                result = engine(rgb, self._next_timestamp())
                ms = (time.perf_counter() - t0) * 1000.0
                self._sink.frame_processed(
                    process_ms=ms, pose_found=result.pose_found, face_found=result.face_found
                )
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()
        finally:
            engine.close()
