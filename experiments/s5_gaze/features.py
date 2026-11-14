"""Spike S5 gaze features from MediaPipe Face Landmarker output (FINAL_06 §4, ADR 0004).

Per frame: iris position relative to the eye corners and lids (both eyes), eyelid aperture, head
yaw/pitch/roll from the facial transformation matrix, and face size. Frames with no face, or
with the eyes closed (mean eyeBlink blendshape >= threshold), are `unknown`.

Only these numbers leave this module. No image is kept or written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# MediaPipe face mesh indices (478-point model with iris). "r" = the subject's right eye.
EYES: dict[str, dict[str, int]] = {
    "r": {"outer": 33, "inner": 133, "upper": 159, "lower": 145, "iris": 468},
    "l": {"outer": 263, "inner": 362, "upper": 386, "lower": 374, "iris": 473},
}
FEATURES: tuple[str, ...] = (
    "r_iris_x",
    "r_iris_y",
    "l_iris_x",
    "l_iris_y",
    "r_aperture",
    "l_aperture",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "face_size",
)
MODEL_FILE = "face_landmarker.task"

Points = npt.NDArray[np.float64]  # (478, 2) pixel coordinates


@dataclass(frozen=True)
class FrameFeatures:
    face_found: bool
    eyes_closed: bool
    values: dict[str, float]  # empty when no face

    @property
    def unknown(self) -> bool:
        return not self.face_found or self.eyes_closed


def eye_features(pts: Points, eye: dict[str, int]) -> tuple[float, float, float]:
    """(horizontal ratio, vertical ratio, aperture) of one eye.

    Horizontal: iris centre projected on the outer->inner corner axis (0 = outer, 1 = inner).
    Vertical: iris centre between upper (0) and lower (1) lid. Aperture: lid gap / eye width.
    """
    outer, inner = pts[eye["outer"]], pts[eye["inner"]]
    upper, lower, iris = pts[eye["upper"]], pts[eye["lower"]], pts[eye["iris"]]
    axis = inner - outer
    width2 = float(axis @ axis)
    if width2 <= 0:
        raise ValueError("eye corners coincide")
    x = float((iris - outer) @ axis) / width2
    lid = lower - upper
    lid2 = float(lid @ lid)
    y = float((iris - upper) @ lid) / lid2 if lid2 > 0 else 0.5
    aperture = math.sqrt(lid2 / width2)
    return x, y, aperture


def head_angles(matrix: npt.NDArray[Any]) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a 4x4 transform, R = Rz(roll) @ Ry(yaw) @ Rx(pitch)."""
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def frame_features(
    pts: Points | None,
    matrix: npt.NDArray[Any] | None,
    blink: tuple[float, float] | None,
    *,
    blink_threshold: float,
    frame_height: int,
) -> FrameFeatures:
    if pts is None or matrix is None:
        return FrameFeatures(False, False, {})
    rx, ry, ra = eye_features(pts, EYES["r"])
    lx, ly, la = eye_features(pts, EYES["l"])
    yaw, pitch, roll = head_angles(matrix)
    size = float(np.linalg.norm(pts[EYES["r"]["outer"]] - pts[EYES["l"]["outer"]])) / frame_height
    closed = blink is not None and (blink[0] + blink[1]) / 2 >= blink_threshold
    values = dict(zip(FEATURES, (rx, ry, lx, ly, ra, la, yaw, pitch, roll, size), strict=True))
    return FrameFeatures(True, closed, values)


class FaceEngine:
    """Face Landmarker in VIDEO mode with blendshapes (blink only) and the pose matrix."""

    def __init__(self, models_dir: Path, blink_threshold: float) -> None:
        import mediapipe as mp  # lazy: unit tests run without MediaPipe models

        path = models_dir / MODEL_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path.name}; run `uv run python scripts/download_models.py --pin`"
            )
        vision = mp.tasks.vision
        self._mp = mp
        self._blink_threshold = blink_threshold
        self._landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,  # used only for eyeBlinkLeft/Right (unknown rule)
                output_facial_transformation_matrixes=True,
            )
        )

    def __call__(self, rgb: npt.NDArray[np.uint8], timestamp_ms: int) -> FrameFeatures:
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect_for_video(image, timestamp_ms)
        if not res.face_landmarks or not res.facial_transformation_matrixes:
            return frame_features(
                None, None, None, blink_threshold=self._blink_threshold, frame_height=h
            )
        pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], dtype=np.float64)
        blink = None
        if res.face_blendshapes:
            scores = {c.category_name: c.score for c in res.face_blendshapes[0]}
            blink = (scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0))
        return frame_features(
            pts,
            np.asarray(res.facial_transformation_matrixes[0]),
            blink,
            blink_threshold=self._blink_threshold,
            frame_height=h,
        )

    def close(self) -> None:
        self._landmarker.close()
