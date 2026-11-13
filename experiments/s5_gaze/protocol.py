"""Spike S5 protocol: the prompted-target schedule for one session (ADR 0004).

A session is a calibration (camera, notes, neutral) followed by 16 blocks, 4 per zone. Each
block is a cue (target name on screen, not used), then a hold between two beeps whose first
`settle_s` is discarded. Labels come from the schedule, so no manual annotation is needed.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = REPO / "experiments" / "configs" / "s5_gaze.yaml"
DATA_DIR = REPO / "data" / "s5"  # git-ignored: per-frame features never enter the repository
RESULTS = REPO / "experiments" / "results" / "s5"
SESSIONS = ("A", "B", "C", "D")
USED_PHASES = ("hold", "calibration_camera")  # phases whose frames are kept for evaluation


@dataclass(frozen=True)
class Segment:
    index: int
    phase: str  # calibration_camera | calibration_notes | calibration_neutral | cue | settle | hold
    label: str  # zone for cue/settle/hold; "" for calibration
    start_s: float
    end_s: float
    rep: int  # 0-3 for blocks; -1 for calibration
    balanced_row: int  # row of the balanced square used for this repetition; -1 for calibration
    away_side: str  # left | right for away blocks; "" otherwise


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data  # pyright: ignore[reportUnknownVariableType]


def check_balanced(rows: list[list[int]]) -> None:
    """Latin square with every ordered neighbour pair exactly once (Williams design)."""
    n = len(rows)
    if any(sorted(r) != list(range(n)) for r in rows):
        raise ValueError("each row must use every zone once")
    if any(sorted(r[i] for r in rows) != list(range(n)) for i in range(n)):
        raise ValueError("each zone must appear once in every position")
    pairs = [(r[i], r[i + 1]) for r in rows for i in range(n - 1)]
    if len(set(pairs)) != len(pairs) or len(pairs) != n * (n - 1):
        raise ValueError("every ordered pair of zones must be neighbours exactly once")


def session_rows(cfg: dict[str, Any], session: str) -> list[int]:
    rows = cfg["balanced_rows"]
    start = int(cfg["start_row"][session])
    return [(start + r) % len(rows) for r in range(int(cfg["timing"]["reps"]))]


def build_schedule(cfg: dict[str, Any], session: str) -> list[Segment]:
    if session not in SESSIONS:
        raise ValueError(f"unknown session {session!r}")
    rows: list[list[int]] = cfg["balanced_rows"]
    check_balanced(rows)
    zones: list[str] = cfg["zones"]
    t = cfg["timing"]
    out: list[Segment] = []
    now = 0.0

    def add(phase: str, label: str, dur: float, rep: int, row: int, side: str) -> None:
        nonlocal now
        out.append(Segment(len(out), phase, label, now, now + dur, rep, row, side))
        now = round(now + dur, 6)

    for phase, dur in t["calibration"]:
        add(str(phase), "", float(dur), -1, -1, "")
    away_count = 0
    for rep, row in enumerate(session_rows(cfg, session)):
        for zi in rows[row]:
            zone = zones[zi]
            side = ""
            if zone == "away":
                side = "left" if away_count % 2 == 0 else "right"
                away_count += 1
            add("cue", zone, float(t["cue_s"]), rep, row, side)
            add("settle", zone, float(t["settle_s"]), rep, row, side)
            add("hold", zone, float(t["hold_s"]) - float(t["settle_s"]), rep, row, side)
    return out


def segment_at(schedule: list[Segment], t_s: float) -> Segment | None:
    """The segment containing time t_s (start inclusive, end exclusive), or None."""
    for seg in schedule:
        if seg.start_s <= t_s < seg.end_s:
            return seg
    return None


def write_timing(
    path: Path, schedule: list[Segment], meta: dict[str, Any], actual_start: dict[int, float]
) -> None:
    """One row per segment: planned times, the time it actually began, and the order used."""
    fields = [*meta, *asdict(schedule[0]), "actual_start_s"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for seg in schedule:
            w.writerow({**meta, **asdict(seg), "actual_start_s": actual_start.get(seg.index, "")})
