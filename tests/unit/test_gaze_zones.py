"""MVP gaze-zone labels after spike S5 (ADR 0004): `at_screen` merged into `at_camera`."""

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from presenceai.schemas.features import GazeZone

REPO = Path(__file__).resolve().parents[1].parent


def test_mvp_gaze_zones_are_exactly_four_labels():
    assert {z.value for z in GazeZone} == {"at_camera", "down_notes", "away", "unknown"}


def test_at_screen_is_not_an_mvp_label():
    with pytest.raises(ValueError):
        GazeZone("at_screen")


def test_s5_config_records_the_fallback_and_matches_the_schema():
    sys.path.insert(0, str(REPO / "experiments" / "s5_gaze"))
    protocol: Any = importlib.import_module("protocol")
    cfg = protocol.load_config()
    assert cfg["merge_for_fallback"] == {"at_screen": "at_camera"}
    assert set(cfg["mvp_zones"]) | {"unknown"} == {z.value for z in GazeZone}
    merged = {cfg["merge_for_fallback"].get(z, z) for z in cfg["zones"]}
    assert merged == set(cfg["mvp_zones"])
