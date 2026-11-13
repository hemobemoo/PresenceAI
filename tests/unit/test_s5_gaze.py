"""Tests for experiments/s5_gaze: schedule, features, unknown rule, manifest check, evaluation."""

import csv
import hashlib
import importlib
import itertools
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "s5_gaze"))
protocol: Any = importlib.import_module("protocol")
features: Any = importlib.import_module("features")
evaluate: Any = importlib.import_module("evaluate")
CFG = protocol.load_config()


# ---- protocol


def test_config_square_is_balanced():
    protocol.check_balanced(CFG["balanced_rows"])


def test_unbalanced_square_rejected():
    with pytest.raises(ValueError):
        protocol.check_balanced([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]])


def _holds(session: str) -> list[Any]:
    return [s for s in protocol.build_schedule(CFG, session) if s.phase == "hold"]


def test_each_session_has_four_holds_per_zone_and_one_per_position():
    for session in protocol.SESSIONS:
        holds = _holds(session)
        assert len(holds) == 16
        for zone in CFG["zones"]:
            assert sum(h.label == zone for h in holds) == 4
        for pos in range(4):
            assert sorted(h.label for h in holds[pos::4]) == sorted(CFG["zones"])


def test_sessions_do_not_share_the_same_sequence():
    seqs = {s: tuple(h.label for h in _holds(s)) for s in protocol.SESSIONS}
    assert len(set(seqs.values())) == len(seqs)
    assert seqs["A"][:4] != seqs["B"][:4]


def test_block_timing_and_away_sides():
    sched = protocol.build_schedule(CFG, "A")
    t = CFG["timing"]
    assert [s.phase for s in sched[:3]] == [
        "calibration_camera",
        "calibration_notes",
        "calibration_neutral",
    ]
    cue, settle, hold = sched[3:6]
    assert (cue.phase, settle.phase, hold.phase) == ("cue", "settle", "hold")
    assert math.isclose(settle.end_s - settle.start_s, t["settle_s"])
    assert math.isclose(hold.end_s - settle.start_s, t["hold_s"])
    sides = [s.away_side for s in sched if s.phase == "hold" and s.label == "away"]
    assert sides == ["left", "right", "left", "right"]
    assert all(b.start_s == a.end_s for a, b in itertools.pairwise(sched))


def test_segment_at_boundaries():
    sched = protocol.build_schedule(CFG, "A")
    assert protocol.segment_at(sched, 0.0).phase == "calibration_camera"
    assert protocol.segment_at(sched, sched[1].start_s).phase == "calibration_notes"
    assert protocol.segment_at(sched, sched[-1].end_s) is None


def test_timing_file_records_order(tmp_path: Path):
    sched = protocol.build_schedule(CFG, "B")
    path = tmp_path / "blocks.csv"
    protocol.write_timing(path, sched, {"session": "B"}, {0: 0.01})
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == len(sched)
    assert {"balanced_row", "rep", "label", "actual_start_s"} <= set(rows[0])
    assert [int(r["balanced_row"]) for r in rows if r["phase"] == "hold"][::4] == [1, 2, 3, 0]


# ---- features


def _face(iris_r: tuple[float, float], iris_l: tuple[float, float], lid: float = 10.0) -> Any:
    pts = np.zeros((478, 2))
    for eye, x0, iris in (("r", 100.0, iris_r), ("l", 200.0, iris_l)):
        e = features.EYES[eye]
        pts[e["outer"]] = (x0, 100.0)
        pts[e["inner"]] = (x0 + 40.0, 100.0)
        pts[e["upper"]] = (x0 + 20.0, 100.0 - lid / 2)
        pts[e["lower"]] = (x0 + 20.0, 100.0 + lid / 2)
        pts[e["iris"]] = (x0 + iris[0] * 40.0, 100.0 - lid / 2 + iris[1] * lid)
    return pts


def _rot(yaw: float, pitch: float, roll: float) -> Any:
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx
    return m


def test_eye_features_ratios_and_aperture():
    pts = _face((0.25, 0.5), (0.5, 0.75))
    rx, ry, ra = features.eye_features(pts, features.EYES["r"])
    lx, ly, _ = features.eye_features(pts, features.EYES["l"])
    assert math.isclose(rx, 0.25) and math.isclose(ry, 0.5)
    assert math.isclose(lx, 0.5) and math.isclose(ly, 0.75)
    assert math.isclose(ra, 10.0 / 40.0)


def test_head_angles_recovered():
    yaw, pitch, roll = features.head_angles(_rot(20.0, -10.0, 5.0))
    assert math.isclose(yaw, 20.0, abs_tol=1e-6)
    assert math.isclose(pitch, -10.0, abs_tol=1e-6)
    assert math.isclose(roll, 5.0, abs_tol=1e-6)


def test_unknown_rule_no_face_and_closed_eyes():
    thr = CFG["eyes_closed_blink"]
    no_face = features.frame_features(None, None, None, blink_threshold=thr, frame_height=360)
    assert no_face.unknown and not no_face.face_found
    pts, m = _face((0.5, 0.5), (0.5, 0.5)), _rot(0, 0, 0)
    closed = features.frame_features(pts, m, (0.9, 0.8), blink_threshold=thr, frame_height=360)
    assert closed.face_found and closed.eyes_closed and closed.unknown
    open_ = features.frame_features(pts, m, (0.1, 0.0), blink_threshold=thr, frame_height=360)
    assert not open_.unknown
    assert set(open_.values) == set(features.FEATURES)


# ---- manifest and evaluation

ZONE_CENTRE = {"at_camera": 0.0, "at_screen": 1.0, "down_notes": 2.0, "away": 3.0}


def _session_rows(session: str, rng: Any, *, unknown_every: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seg in protocol.build_schedule(CFG, session):
        if seg.phase not in ("hold", "calibration_camera"):
            continue
        centre = ZONE_CENTRE.get(seg.label, 0.0)
        for k in range(10):
            row: dict[str, Any] = {
                "t_s": seg.start_s + k * 0.05,
                "segment": seg.index,
                "phase": seg.phase,
                "label": seg.label,
                "rep": seg.rep,
                "face_found": not (unknown_every and k % unknown_every == 0),
                "eyes_closed": False,
                "process_ms": 10.0,
            }
            for i, name in enumerate(features.FEATURES):
                row[name] = centre * (i + 1) + rng.normal(0, 0.05)
            rows.append(row)
    return rows


def _write_session(tmp: Path, session: str, rows: list[dict[str, Any]]) -> None:
    data, out = tmp / "data", tmp / "out"
    data.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    path = data / f"run_{session}_features.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    manifest = out / "features_manifest.csv"
    new = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "session", "file", "rows", "sha256", "written_at"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        w.writerow(["run", session, path.name, len(rows), digest, "now"])


def test_separable_sessions_pass_and_unknown_share_is_reported(tmp_path: Path):
    rng = np.random.default_rng(0)
    _write_session(tmp_path, "A", _session_rows("A", rng, unknown_every=10))
    _write_session(tmp_path, "B", _session_rows("B", rng))
    sessions = evaluate.load_sessions(tmp_path / "data", tmp_path / "out" / "features_manifest.csv")
    excl = {e["session"]: e for e in evaluate.excluded_table(sessions)}
    assert excl["A"]["excluded_share"] == pytest.approx(0.1)
    assert excl["B"]["excluded_share"] == 0
    rows = evaluate.evaluate(sessions, CFG)
    assert {(r["train"], r["test"]) for r in rows} == {("A", "B"), ("B", "A")}  # no C/D yet
    assert len(rows) == 2 * 2 * 2 * 2  # folds x calibration x zones x models
    assert evaluate.decide(rows, CFG)["outcome"] == "pass"


def test_screen_camera_confusion_gives_fallback_pass(tmp_path: Path):
    rng = np.random.default_rng(1)
    for s in ("A", "B"):
        rows = _session_rows(s, rng)
        for r in rows:  # make at_screen indistinguishable from at_camera
            if r["label"] == "at_screen":
                for i, name in enumerate(features.FEATURES):
                    r[name] = rng.normal(0, 0.05) + 0.0 * i
        _write_session(tmp_path, s, rows)
    sessions = evaluate.load_sessions(tmp_path / "data", tmp_path / "out" / "features_manifest.csv")
    decision = evaluate.decide(evaluate.evaluate(sessions, CFG), CFG)
    assert decision["outcome"] == "pass_fallback"
    assert (
        decision["macro_f1_4_zones"] < CFG["pass"]["min_macro_f1"] <= decision["macro_f1_3_zones"]
    )


def test_missing_session_b_is_incomplete(tmp_path: Path):
    _write_session(tmp_path, "A", _session_rows("A", np.random.default_rng(2)))
    sessions = evaluate.load_sessions(tmp_path / "data", tmp_path / "out" / "features_manifest.csv")
    assert evaluate.decide(evaluate.evaluate(sessions, CFG), CFG)["outcome"] == "incomplete"


def test_changed_features_file_is_rejected(tmp_path: Path):
    _write_session(tmp_path, "A", _session_rows("A", np.random.default_rng(3)))
    path = tmp_path / "data" / "run_A_features.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(evaluate.ManifestError):
        evaluate.load_sessions(tmp_path / "data", tmp_path / "out" / "features_manifest.csv")
