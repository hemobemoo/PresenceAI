"""The owner's hand check of C2 (P2-e, ADR 0021): a blinded, stratified sample of the test part.

    uv run python experiments/c2_synth/hand_check.py draw   # writes data/c2/hand_check/sheet.yaml
    uv run python experiments/c2_synth/hand_check.py score  # after the owner fills the sheet

- **draw:** 10 accepted transcripts per group (4 defect groups + controls = 50), from the
  **test part only**, at random with a fixed seed. The sheet shows only a neutral key and the
  text, in shuffled order; the planted group and hook go to `key.json`, so the owner records
  what is actually present without seeing what was planted.
- The owner records, for each transcript: central message stated (yes/no), conclusion
  (yes/no), supporting material (yes/no), unexplained technical terms (yes/no), hook (yes/no).
- **score:** compares the owner's labels with the planted design. The planted-label error rate
  in the sample is the label-noise estimate for E4; the per-item labels are also the reference
  for E4's second report (the checked subset against the owner's labels). Writes counts only to
  experiments/results/c2/hand_check.json.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "data" / "c2" / "generated.jsonl"
OUT_DIR = ROOT / "data" / "c2" / "hand_check"
RESULTS = ROOT / "experiments" / "results" / "c2" / "hand_check.json"
SEED = 20261123
PER_GROUP = 10
LABELS = ("central_message", "conclusion", "supporting_material", "unexplained_terms", "hook")
# What each group's design plants ("hook" comes from the prompt's own flag).
PLANTED = {
    "control": {"central_message": True, "conclusion": True, "supporting_material": True,
                "unexplained_terms": False},
    "no_central_message": {"central_message": False, "conclusion": True,
                           "supporting_material": True, "unexplained_terms": False},
    "no_conclusion": {"central_message": True, "conclusion": False, "supporting_material": True,
                      "unexplained_terms": False},
    "no_supporting_material": {"central_message": True, "conclusion": True,
                               "supporting_material": False, "unexplained_terms": False},
    "undefined_jargon": {"central_message": True, "conclusion": True, "supporting_material": True,
                         "unexplained_terms": True},
}  # fmt: skip


Sheet = list[dict[str, Any]]


def draw(rows: list[dict[str, Any]], seed: int = SEED) -> tuple[Sheet, dict[str, Any]]:
    """(sheet items, key): 10 per group from the test part, blinded and shuffled."""
    rng = random.Random(seed)  # noqa: S311 - reproducible sample, not security
    chosen: list[dict[str, Any]] = []
    for group in PLANTED:
        pool = sorted((r for r in rows if r["part"] == "test" and r["group"] == group),
                      key=lambda r: r["id"])  # fmt: skip
        if len(pool) < PER_GROUP:
            raise ValueError(f"{group}: only {len(pool)} accepted test transcripts")
        chosen += rng.sample(pool, PER_GROUP)
    rng.shuffle(chosen)
    sheet, key = [], {}
    for n, r in enumerate(chosen, start=1):
        k = f"h{n:02d}"
        sheet.append({"key": k, "text": r["text"], **dict.fromkeys(LABELS)})
        key[k] = {"id": r["id"], "group": r["group"], "hook": r["hook"]}
    return sheet, key


def planted(entry: dict[str, Any]) -> dict[str, bool]:
    return {**PLANTED[entry["group"]], "hook": bool(entry["hook"])}


def label(value: Any) -> bool:
    """The owner's yes/no: a YAML boolean or the strings "yes"/"no" (quoted or not).
    Anything else is refused; bool("no") would be True."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if isinstance(value, str) else ""
    if text in ("yes", "y", "true"):
        return True
    if text in ("no", "n", "false"):
        return False
    raise ValueError(f"not a yes/no label: {value!r}")


def score(sheet: list[dict[str, Any]], key: dict[str, Any]) -> dict[str, Any]:
    missing = [s["key"] for s in sheet if any(s.get(lab) is None for lab in LABELS)]
    if missing:
        raise ValueError(f"labels missing for {missing}")
    wrong_items, by_label, by_group = 0, Counter(), Counter()
    for s in sheet:
        want = planted(key[s["key"]])
        diff = [lab for lab in LABELS if label(s[lab]) != want[lab]]
        by_label.update(diff)
        if diff:
            wrong_items += 1
            by_group[key[s["key"]]["group"]] += 1
    n = len(sheet)
    return {
        "items": n,
        "items_with_any_planted_label_wrong": wrong_items,
        "item_error_rate": round(wrong_items / n, 4),
        "label_error_rate": {lab: round(by_label[lab] / n, 4) for lab in LABELS},
        "items_wrong_by_group": dict(by_group),
    }


def main(argv: list[str]) -> int:
    action = argv[1] if len(argv) > 1 else "draw"
    sheet_path, key_path = OUT_DIR / "sheet.yaml", OUT_DIR / "key.json"
    if action == "draw":
        if sheet_path.exists():
            raise SystemExit(f"{sheet_path} exists; the sample is drawn once")
        rows = [json.loads(x) for x in GENERATED.read_text("utf-8").splitlines() if x.strip()]
        sheet, key = draw(rows)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        sheet_path.write_text(yaml.safe_dump(sheet, allow_unicode=True, sort_keys=False, width=100),
                              encoding="utf-8")  # fmt: skip
        key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(sheet)} items to {sheet_path.relative_to(ROOT)} (key kept apart)")
        return 0
    sheet = yaml.safe_load(sheet_path.read_text("utf-8"))
    key = json.loads(key_path.read_text("utf-8"))
    result = score(sheet, key)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
