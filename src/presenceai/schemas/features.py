"""Feature-level types shared between modules.

`ExpressionLabel` is the complete set of facial expression events the app may emit
(docs/plan/FINAL_04_MULTIMODAL_MEASUREMENT.md section 4.3, FINAL_11 section 2).
No emotion, mood, attitude or intention label may be added here.

`GazeZone` is the MVP gaze-zone label set (FINAL_04 section 4.2, ADR 0004). Spike S5 could not
separate the lens from the screen area, so `at_camera` covers both ("toward the audience").
A separate `at_screen` label is future work.
"""

from __future__ import annotations

from enum import StrEnum


class ExpressionLabel(StrEnum):
    SMILE_VISIBLE = "smile_visible"
    BROWS_RAISED = "brows_raised"
    BROWS_LOWERED = "brows_lowered"
    EYES_CLOSED = "eyes_closed"
    MOUTH_MOVING_SILENT = "mouth_moving_silent"


class GazeZone(StrEnum):
    AT_CAMERA = "at_camera"  # lens or the screen area near it: toward the audience
    DOWN_NOTES = "down_notes"
    AWAY = "away"
    UNKNOWN = "unknown"  # no face found, or eyes closed / not reliably visible
