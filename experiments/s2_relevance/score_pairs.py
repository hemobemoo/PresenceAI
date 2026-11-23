"""Score the owner's S2 relevance calibration pairs with the e5 relevance measure (ADR 0022 §5).

    uv run python experiments/s2_relevance/score_pairs.py data/s2_relevance/pairs.txt

The pairs are owner-written calibration data: git-ignored under `data/s2_relevance/` and kept
out of every test (ADR 0022 §5). The file is blocks of `id:`, `question:`, `answer:` and
`label:` lines (good, evasive or off-topic), separated by blank lines. Uses the same function
as the app (`text/followup.py: relevance`, question = query, answer = passage). Writes
experiments/results/s2_relevance/scores.json with ids, labels and scores only, never the text.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "results" / "s2_relevance" / "scores.json"
LABELS = ("good", "evasive", "off-topic")
FIELDS = ("id", "question", "answer", "label")


@dataclass(frozen=True)
class Pair:
    id: int
    question: str
    answer: str
    label: str


def parse(text: str) -> list[Pair]:
    pairs: list[Pair] = []
    for block in (b for b in text.replace("\r\n", "\n").split("\n\n") if b.strip()):
        fields: dict[str, str] = {}
        for line in block.strip().splitlines():
            key, sep, value = line.partition(":")
            if not sep or key.strip() not in FIELDS or key.strip() in fields:
                raise ValueError(f"bad line in pair block: {line[:40]!r}")
            fields[key.strip()] = value.strip()
        if set(fields) != set(FIELDS):
            raise ValueError(f"pair block is missing {sorted(set(FIELDS) - set(fields))}")
        if fields["label"] not in LABELS:
            raise ValueError(f"unknown label {fields['label']!r}")
        if not fields["question"] or not fields["answer"]:
            raise ValueError(f"pair {fields['id']} has an empty question or answer")
        pairs.append(Pair(int(fields["id"]), fields["question"], fields["answer"], fields["label"]))
    ids = [p.id for p in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pair id")
    return pairs


def summary(scores: list[float]) -> dict[str, float | int]:
    return {
        "n": len(scores),
        "min": min(scores),
        "median": statistics.median(scores),
        "max": max(scores),
        "mean": statistics.fmean(scores),
    }


def main(path: Path) -> None:
    from presenceai.knowledge.embed import E5Embedder
    from presenceai.text.followup import relevance

    raw = path.read_bytes()
    pairs = parse(raw.decode("utf-8"))
    emb = E5Embedder()
    rows = []
    for p in pairs:
        score = relevance(p.question, p.answer, emb)
        if score is None:  # parse() refuses empty answers, so this cannot happen
            raise ValueError(f"pair {p.id}: no relevance for an empty answer")
        rows.append({"id": p.id, "label": p.label, "relevance": round(score, 4)})
    by_label = {
        lab: summary([r["relevance"] for r in rows if r["label"] == lab])
        for lab in LABELS
        if any(r["label"] == lab for r in rows)
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": "owner-written calibration pairs (data/s2_relevance/pairs.txt)",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "embedder": emb.name,
                "by_label": by_label,
                "pairs": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for lab, s in by_label.items():
        print(f"{lab:>9}: n={s['n']} min={s['min']:.3f} median={s['median']:.3f} "
              f"max={s['max']:.3f}")  # fmt: skip
    for r in sorted(rows, key=lambda r: -r["relevance"]):
        print(f"  {r['relevance']:.3f}  #{r['id']:<2} {r['label']}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
