"""Automatic checks on every generated C2 transcript (P2-e, ADR 0021). Local, counts only.

    uv run python experiments/c2_synth/check_c2.py data/c2/generated.jsonl

A second, local pass over the accepted transcripts (the notebook already applied the quality
loop: token cap, length, format and prompt echo; generate_c2.py; no model yes/no check).
Format checks (all groups): 300-550 words; no markdown, heading, bullet, speaker label or stage
direction; no leftover thinking block; no "main idea" / "central idea" echo. Group check that
needs no language model: the no_supporting_material group contains no digits. Like the
notebook, it uses no cue lexicon and no detector. If attempts.jsonl and rejected.jsonl sit next
to the file, it also reports attempts and rejections per group and the reasons. Writes
experiments/results/c2/<file>_checks.json with counts and failing ids only.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "resources" / "c2" / "prompts.jsonl"
MIN_WORDS, MAX_WORDS = 300, 550
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
    if re.search(r"(main|central) idea", text, re.IGNORECASE):
        out.append("echo")
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
    loop: dict[str, object] = {}
    attempts_file, rejected_file = (
        path.with_name("attempts.jsonl"),
        path.with_name("rejected.jsonl"),
    )
    if attempts_file.is_file():
        att = [json.loads(x) for x in attempts_file.read_text("utf-8").splitlines() if x.strip()]
        rej = []
        if rejected_file.is_file():
            rej = [
                json.loads(x) for x in rejected_file.read_text("utf-8").splitlines() if x.strip()
            ]
        failed = [a for a in att if a["problems"]]
        loop = {
            "attempts_by_group": dict(Counter(a["group"] for a in att)),
            "failed_attempts_by_group": dict(Counter(a["group"] for a in failed)),
            "rejected_by_group": dict(Counter(r["group"] for r in rej)),
            "failed_attempt_reasons": dict(
                Counter(p.split()[0] for a in failed for p in a["problems"])
            ),
            "rejected_ids": sorted(r["id"] for r in rej),
        }
    return {
        **loop,
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
