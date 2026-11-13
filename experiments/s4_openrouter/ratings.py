"""Unblind the owner's Arabic wording ratings for S4 runs and summarise them per model.

    uv run python experiments/s4_openrouter/ratings.py 20261113-143329 20261113-211500

For each run, reads <run>_ar_rating_sheet.csv (owner's ratings, not modified) and
<run>_ar_rating_key.csv, and writes:
    <run>_ar_ratings_unblinded.csv   prompt_id, model_code, model, rating_1_to_5, notes
    <run>_ar_ratings_summary.csv     model, n_replies, n_rated, mean_rating (1-5 ratings only)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core


def unblind(out: Path, run_id: str) -> tuple[Path, Path]:
    with (out / f"{run_id}_ar_rating_key.csv").open(newline="", encoding="utf-8") as f:
        model_of = {r["model_code"]: r["model"] for r in csv.DictReader(f)}
    with (out / f"{run_id}_ar_rating_sheet.csv").open(newline="", encoding="utf-8-sig") as f:
        rows: list[dict[str, Any]] = [
            {
                "prompt_id": r["prompt_id"],
                "model_code": r["model_code"],
                "model": model_of[r["model_code"]],
                "rating_1_to_5": r["rating_1_to_5"],
                "notes": r["notes"],
            }
            for r in csv.DictReader(f)
        ]
    un_path = out / f"{run_id}_ar_ratings_unblinded.csv"
    with un_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = core.rating_summary(rows)
    sum_path = out / f"{run_id}_ar_ratings_summary.csv"
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(
            f"{run_id} {s['model']:45s} rated {s['n_rated']}/{s['n_replies']} "
            f"mean {s['mean_rating']}"
        )
    return un_path, sum_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--out", type=Path, default=core.REPO / "experiments" / "results" / "s4")
    args = ap.parse_args(argv)
    for run_id in args.run_ids:
        unblind(args.out, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
