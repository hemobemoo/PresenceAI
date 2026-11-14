"""Spike S5 evaluation: gaze-zone macro-F1 from the recorded sessions (ADR 0004).

    uv run python experiments/s5_gaze/evaluate.py

Uses the latest manifest entry per session. Each features file in data/s5/ is checked against its
SHA-256 in experiments/results/s5/features_manifest.csv before use.

- Frames: `hold` phase only (the settle part of each hold is already excluded). Frames with no
  face or closed eyes are `unknown` and excluded from F1; the excluded share is reported.
- Calibration: features minus the session's mean over known `calibration_camera` frames.
- Models: logistic regression (standardised) and HistGradientBoosting; 4 zones and 3 zones
  (at_screen merged into at_camera); with and without calibration.
- Folds: main A->B and B->A; robustness A+B->C and A+B->D when those sessions exist.
- Decision (pass rule in the config): primary 4-zone mean >= threshold -> pass; else primary
  3-zone mean >= threshold -> pass_fallback; else fail (report only, change nothing).

Writes experiments/results/s5/<stamp>_eval.csv, _excluded.csv, _excluded_by_zone.csv,
_calibration.csv, _confusion.csv (primary model, main folds) and _decision.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

import protocol
from features import FEATURES

MODELS = ("lr", "hgb")
FCOLS: list[str] = list(FEATURES)


class ManifestError(ValueError):
    pass


def _truthy(s: Any) -> Any:
    return s.astype(str).str.lower().isin(["true", "1"])


def load_sessions(data_dir: Path, manifest: Path) -> dict[str, pd.DataFrame]:
    """Latest features file per session, verified against the manifest SHA-256."""
    if not manifest.exists():
        raise ManifestError(f"no manifest at {manifest}; record sessions first")
    with manifest.open(newline="", encoding="utf-8") as f:
        entries = list(csv.DictReader(f))
    latest: dict[str, dict[str, str]] = {}
    for e in entries:
        latest[e["session"]] = e  # later rows win
    out: dict[str, pd.DataFrame] = {}
    for session, e in sorted(latest.items()):
        path = data_dir / e["file"]
        if not path.is_file():
            raise ManifestError(f"{path} is listed in the manifest but missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != e["sha256"]:
            raise ManifestError(f"{path.name}: SHA-256 does not match the manifest")
        df = pd.read_csv(path)
        df["session"] = session
        df["run_id"] = e["run_id"]
        out[session] = df
    return out


def known_mask(df: pd.DataFrame) -> Any:
    return _truthy(df["face_found"]) & ~_truthy(df["eyes_closed"])


def excluded_table(sessions: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows = []
    for name, df in sessions.items():
        hold = df[df["phase"] == "hold"]
        no_face = int((~_truthy(hold["face_found"])).sum())
        closed = int((_truthy(hold["face_found"]) & _truthy(hold["eyes_closed"])).sum())
        n = len(hold)
        rows.append(
            {
                "session": name,
                "run_id": df["run_id"].iloc[0] if len(df) else "",
                "hold_frames": n,
                "unknown_no_face": no_face,
                "unknown_eyes_closed": closed,
                "excluded_share": round((no_face + closed) / n, 4) if n else None,
            }
        )
    return rows


def excluded_by_zone(sessions: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Excluded (unknown) share of hold frames per session and zone."""
    rows = []
    for name, df in sessions.items():
        hold = df[df["phase"] == "hold"]
        for zone in sorted(set(hold["label"].astype(str))):
            z = hold[hold["label"] == zone]
            no_face = int((~_truthy(z["face_found"])).sum())
            closed = int((_truthy(z["face_found"]) & _truthy(z["eyes_closed"])).sum())
            rows.append(
                {
                    "session": name,
                    "zone": zone,
                    "hold_frames": len(z),
                    "unknown_no_face": no_face,
                    "unknown_eyes_closed": closed,
                    "excluded_share": round((no_face + closed) / len(z), 4) if len(z) else None,
                }
            )
    return rows


def calibration_check(sessions: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """How far the calibration offset is from the session's own at_camera hold frames.

    Distance per feature in units of the at_camera hold standard deviation; reported as the
    mean over features, the largest single feature, and the same for the calibration's first
    second (eyes may still be moving to the lens) and the rest.
    """
    rows = []
    for name, df in sessions.items():
        known: Any = df[known_mask(df)]
        cal: Any = known[known["phase"] == "calibration_camera"]
        cam: Any = known[(known["phase"] == "hold") & (known["label"] == "at_camera")]
        sd = cam[FCOLS].std()

        def dist(part: Any, cam: Any = cam, sd: Any = sd) -> Any:
            return ((part[FCOLS].mean() - cam[FCOLS].mean()) / sd).abs()

        d, first, rest = dist(cal), dist(cal[cal["t_s"] < 1.0]), dist(cal[cal["t_s"] >= 1.0])
        rows.append(
            {
                "session": name,
                "calibration_frames": int((df["phase"] == "calibration_camera").sum()),
                "calibration_known": len(cal),
                "frames_first_1s": int((cal["t_s"] < 1.0).sum()),
                "mean_abs_z": round(float(d.mean()), 3),
                "max_abs_z": round(float(d.max()), 3),
                "max_feature": str(d.idxmax()),
                "mean_abs_z_first_1s": round(float(first.mean()), 3)
                if (cal["t_s"] < 1.0).any()
                else None,
                "mean_abs_z_after_1s": round(float(rest.mean()), 3),
            }
        )
    return rows


def prepared(df: pd.DataFrame, calibration: bool) -> Any:
    """Known hold frames with features (optionally minus the camera-calibration mean)."""
    known: Any = df[known_mask(df)]
    hold: Any = known[known["phase"] == "hold"].copy()
    if calibration:
        cal = known[known["phase"] == "calibration_camera"]
        if cal.empty:
            raise ValueError(f"session {df['session'].iloc[0]} has no known calibration frames")
        hold[FCOLS] = hold[FCOLS] - cal[FCOLS].mean()
    return hold


def _f1(y_true: Any, y_pred: Any, labels: list[str], average: str | None) -> Any:
    """F1 with absent classes scored 0 (sklearn's stubs type these keywords too narrowly)."""
    kwargs: dict[str, Any] = {"labels": labels, "average": average, "zero_division": 0}
    return f1_score(y_true, y_pred, **kwargs)


def make_model(kind: str, random_state: int) -> Any:
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=random_state)
    raise ValueError(kind)


def labels_for(df: Any, n_zones: int, merge: dict[str, str]) -> Any:
    y = df["label"].astype(str)
    return y.replace(merge) if n_zones == 3 else y


def primary_confusion(
    sessions: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Confusion counts (true x predicted) of the primary model on the main folds, 4 and 3 zones."""
    prim = cfg["pass"]["primary"]
    merge = cfg["merge_for_fallback"]
    rows: list[dict[str, Any]] = []
    for train, test in cfg["pass"]["main_folds"]:
        if train not in sessions or test not in sessions:
            continue
        tr = prepared(sessions[train], bool(prim["calibration"]))
        te = prepared(sessions[test], bool(prim["calibration"]))
        for n_zones in (4, 3):
            y_tr, y_te = labels_for(tr, n_zones, merge), labels_for(te, n_zones, merge)
            labels = sorted(set(y_tr) | set(y_te))
            model = make_model(str(prim["model"]), int(cfg["random_state"]))
            model.fit(tr[FCOLS].to_numpy(), y_tr.to_numpy())
            cm = confusion_matrix(y_te, model.predict(te[FCOLS].to_numpy()), labels=labels)
            for i, true in enumerate(labels):
                for j, pred in enumerate(labels):
                    rows.append(
                        {
                            "train": train,
                            "test": test,
                            "model": prim["model"],
                            "calibration": prim["calibration"],
                            "zones": n_zones,
                            "true": true,
                            "pred": pred,
                            "count": int(cm[i, j]),
                        }
                    )
    return rows


def evaluate(sessions: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    merge = cfg["merge_for_fallback"]
    folds: list[tuple[str, list[str], str]] = [
        ("main", [tr], te) for tr, te in cfg["pass"]["main_folds"]
    ]
    folds += [("robustness", list(tr), te) for tr, te in cfg["robustness_folds"]]
    rows: list[dict[str, Any]] = []
    for kind, train, test in folds:
        if test not in sessions or any(t not in sessions for t in train):
            continue
        for calibration in (False, True):
            tr = pd.concat([prepared(sessions[t], calibration) for t in train])
            te = prepared(sessions[test], calibration)
            for n_zones in (4, 3):
                y_tr = labels_for(tr, n_zones, merge)
                y_te = labels_for(te, n_zones, merge)
                labels = sorted(set(y_tr) | set(y_te))
                for model_name in MODELS:
                    model = make_model(model_name, int(cfg["random_state"]))
                    model.fit(tr[FCOLS].to_numpy(), y_tr.to_numpy())
                    pred = model.predict(te[FCOLS].to_numpy())
                    per = _f1(y_te, pred, labels, None)
                    row: dict[str, Any] = {
                        "fold": kind,
                        "train": "+".join(train),
                        "test": test,
                        "model": model_name,
                        "zones": n_zones,
                        "calibration": calibration,
                        "n_train": len(tr),
                        "n_test": len(te),
                        "macro_f1": round(
                            float(_f1(y_te, pred, labels, "macro")),
                            4,
                        ),
                    }
                    for zone, v in zip(labels, np.atleast_1d(per), strict=True):
                        row[f"f1_{zone}"] = round(float(v), 4)
                    rows.append(row)
    return rows


def decide(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["pass"]
    prim = p["primary"]
    threshold = float(p["min_macro_f1"])

    def mean_main(zones: int) -> float | None:
        vals = [
            r["macro_f1"]
            for r in rows
            if r["fold"] == "main"
            and r["model"] == prim["model"]
            and r["calibration"] == prim["calibration"]
            and r["zones"] == zones
        ]
        return round(sum(vals) / len(vals), 4) if len(vals) == len(p["main_folds"]) else None

    f4, f3 = mean_main(4), mean_main(3)
    if f4 is None or f3 is None:
        outcome = "incomplete"  # sessions A and B are both needed
    elif f4 >= threshold:
        outcome = "pass"
    elif f3 >= threshold:
        outcome = "pass_fallback"
    else:
        outcome = "fail"
    return {
        "primary_model": prim["model"],
        "primary_calibration": prim["calibration"],
        "threshold": threshold,
        "macro_f1_4_zones": f4,
        "macro_f1_3_zones": f3,
        "outcome": outcome,
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=protocol.CONFIG)
    ap.add_argument("--data-dir", type=Path, default=protocol.DATA_DIR)
    ap.add_argument("--out", type=Path, default=protocol.RESULTS)
    args = ap.parse_args(argv)
    cfg = protocol.load_config(args.config)
    sessions = load_sessions(args.data_dir, args.out / "features_manifest.csv")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    excluded = excluded_table(sessions)
    rows = evaluate(sessions, cfg)
    decision = {"evaluated_at": stamp, **decide(rows, cfg)}
    _write(args.out / f"{stamp}_excluded.csv", excluded)
    _write(args.out / f"{stamp}_excluded_by_zone.csv", excluded_by_zone(sessions))
    _write(args.out / f"{stamp}_calibration.csv", calibration_check(sessions))
    confusion = primary_confusion(sessions, cfg)
    if confusion:
        _write(args.out / f"{stamp}_confusion.csv", confusion)
    if rows:
        _write(args.out / f"{stamp}_eval.csv", rows)
    _write(args.out / f"{stamp}_decision.csv", [decision])
    for e in excluded:
        print(
            f"session {e['session']}: {e['hold_frames']} hold frames, "
            f"excluded {e['excluded_share']}"
        )
    for r in rows:
        print(
            f"{r['fold']:10s} {r['train']:>3s}->{r['test']} {r['model']:3s} "
            f"{r['zones']} zones cal={r['calibration']!s:5s} macro-F1 {r['macro_f1']}"
        )
    print(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
