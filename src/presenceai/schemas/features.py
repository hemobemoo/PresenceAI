"""Feature-level types shared between modules.

`ExpressionLabel` is the complete set of facial expression events the app may emit
(docs/plan/FINAL_04_MULTIMODAL_MEASUREMENT.md section 4.3, FINAL_11 section 2).
No emotion, mood, attitude or intention label may be added here.
"""

from __future__ import annotations

from enum import StrEnum


class ExpressionLabel(StrEnum):
    SMILE_VISIBLE = "smile_visible"
    BROWS_RAISED = "brows_raised"
    BROWS_LOWERED = "brows_lowered"
    EYES_CLOSED = "eyes_closed"
    MOUTH_MOVING_SILENT = "mouth_moving_silent"
