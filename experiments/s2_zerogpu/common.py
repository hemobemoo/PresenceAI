"""Spike S2 shared logic: config, paths, checksums and the pass rule (ADR 0007)."""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = REPO / "experiments" / "configs" / "s2_zerogpu.yaml"
RESULTS = REPO / "experiments" / "results" / "s2"
DATA_DIR = REPO / "data" / "s2"  # git-ignored (data/*): clips, built Space folder
MODELS_DIR = REPO / "models" / "s2"  # git-ignored (models/*): converted model


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data  # pyright: ignore[reportUnknownVariableType]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Call:
    """One full-clip analysis on the Space. gpu_s is the time inside @spaces.GPU (load included)."""

    kind: str  # "cold" or "warm"
    gpu_s: float
    text: str
    ok: bool = True  # False if the call raised (for example a quota or timeout error)


def decide(calls: Sequence[Call], max_median_warm: float, max_any: float) -> dict[str, Any]:
    """ADR 0007: non-empty transcripts; median warm GPU-s < limit; no call over the reservation."""
    warm = [c for c in calls if c.kind == "warm"]
    if not warm:
        raise ValueError("no warm calls")
    all_ok = all(c.ok for c in calls)
    nonempty = all_ok and all(c.text.strip() for c in calls)
    median_warm = statistics.median(c.gpu_s for c in warm)
    max_call = max(c.gpu_s for c in calls)
    median_ok = all_ok and median_warm < max_median_warm
    max_ok = all_ok and max_call <= max_any
    return {
        "calls": len(calls),
        "warm_calls": len(warm),
        "all_calls_ok": all_ok,
        "nonempty": nonempty,
        "median_warm_gpu_s": round(median_warm, 3),
        "max_median_warm_gpu_s": max_median_warm,
        "median_ok": median_ok,
        "max_call_gpu_s": round(max_call, 3),
        "max_any_call_gpu_s": max_any,
        "max_ok": max_ok,
        "outcome": "pass" if nonempty and median_ok and max_ok else "fail",
    }
