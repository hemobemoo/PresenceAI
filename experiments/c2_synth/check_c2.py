"""Automatic checks on every generated C2 transcript (P2-e, ADR 0021). Local, counts only.

    uv run python experiments/c2_synth/check_c2.py data/c2/generated.jsonl

Format checks (all groups): 250-450 words; no markdown, heading, bullet, speaker label or stage
direction; no leftover thinking block. Group check that needs no language model: the
no_supporting_material group contains no digits. Whether each planted defect is really there
(and absent from the controls) is checked by a separate yes/no pass of the model with plain
questions (`generate_c2.py judge`, after the pilot), and by the owner's stratified hand-checked
sample. Writes experiments/results/c2/<file>_checks.json with counts and failing ids only.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "resources" / "c2" / "prompts.jsonl"
MIN_WORDS, MAX_WORDS = 250, 450
_FORMAT = re.compile(
    r"(^\s*#)|(^\s*[-*•]\s)|(^\s*\d+[.)]\s)|(\*\*)|(<think>)|(</think>)|(^\s*\[[^\]]+\])"
    r"|(^\s*\([^)]*\)\s*$)|(^\s*(speaker|presenter|student)\s*\d*\s*:)",
    re.IGNORECASE | re.MULTILINE,
)


def problems(text: str, group: str) -> list[str]:
    out = []
    n = len(text.split())
    if not MIN_WORDS <= n <= MAX_WORDS:
        out.append(f"words {n}")
    if _FORMAT.search(text):
        out.append("format")
    if group == "no_supporting_material" and re.search(r"\d", text):
        out.append("digits")
    return out


def check(path: Path) -> dict[str, object]:
    prompts = {p["id"]: p for p in map(json.loads, PROMPTS.read_text("utf-8").splitlines()) if p}
    rows = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]
    failing: dict[str, list[str]] = {}
    for r in rows:
        found = problems(r["text"], str(prompts[r["id"]]["group"]))
        if found:
            failing[r["id"]] = found
    missing = sorted(set(prompts) - {r["id"] for r in rows})
    return {
        "generated": len(rows),
        "missing": len(missing),
        "failing": len(failing),
        "by_problem": dict(Counter(p.split()[0] for v in failing.values() for p in v)),
        "failing_ids": failing,
    }


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else ROOT / "data" / "c2" / "generated.jsonl"
    result = check(path)
    out = ROOT / "experiments" / "results" / "c2" / f"{path.stem}_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print({k: v for k, v in result.items() if k != "failing_ids"})
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
