"""Record each shortlisted free model's provider data policy from its public OpenRouter page.

    uv run python experiments/s4_openrouter/policies.py

Writes experiments/results/s4/<date>_policies.csv: provider, whether it may train on prompts,
whether it retains prompts (and for how many days, if stated), whether it requires user ids,
and its terms and privacy URLs. No API key is used. The page format is not a documented API, so
a model whose policy cannot be parsed is written with empty fields, never guessed.
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core


def fetch(model: str) -> str:
    url = f"https://openrouter.ai/{model}/providers"
    req = urllib.request.Request(url, headers={"User-Agent": "presenceai-s4"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (fixed https URL)
        return resp.read().decode("utf-8")


def main() -> int:
    cfg = core.load_config()
    out = core.REPO / "experiments" / "results" / "s4"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    names = ["checked_at", *(f.name for f in fields(core.Policy))]
    path = out / f"{stamp}_policies.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        for model in cfg["models"]:
            pol = core.parse_policy(fetch(model), model)
            row = asdict(pol) if pol else {"model": model}
            w.writerow({"checked_at": stamp, **row})
            print(model, "->", "parsed" if pol else "NOT FOUND", asdict(pol) if pol else "")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
