"""T1 model pick on dev (ADR 0029 §5) and the pass rule (ADR 0026 §4, ADR 0029 §10). Pure.

The pass rule reads point estimates from E3 metrics dicts shaped like `e3_metrics.json`
(per condition: ami_test, podcastfillers_test, librispeech_test_clean).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def pick(cands: Sequence[Mapping[str, Any]], tie_within: float) -> dict[str, Any]:
    """cands: {epoch, f1: {set: F1}, wer: {set: WER %}}, the same sets for every candidate.
    Highest mean F1; candidates within `tie_within` of it tie and go to the lowest mean WER;
    identical WER as well -> the earlier epoch."""
    if not cands:
        raise ValueError("no candidates to pick from")
    sets = sorted(cands[0]["f1"])

    def mean(d: Mapping[str, float | None]) -> float:
        if any(d.get(s) is None for s in sets):
            raise ValueError(f"a candidate has no value for one of {sets}")
        return sum(float(d[s]) for s in sets) / len(sets)  # type: ignore[arg-type]

    scored = [
        {"epoch": c["epoch"], "score": mean(c["f1"]), "mean_wer": mean(c["wer"])} for c in cands
    ]
    best = max(s["score"] for s in scored)
    tied = [s for s in scored if best - s["score"] <= tie_within + 1e-12]
    chosen = min(tied, key=lambda s: (s["mean_wer"], s["epoch"]))
    return {
        "picked_epoch": chosen["epoch"],
        "scores": sorted(scored, key=lambda s: s["epoch"]),
        "tied_epochs": sorted(s["epoch"] for s in tied),
        "sets": sets,
        "tie_within": tie_within,
    }


DECIDING = ("ami_test", "podcastfillers_test")


def pass_rule(t1: Mapping[str, Any], base: Mapping[str, Any], prompt: Mapping[str, Any]):
    """ADR 0026 §4 on point estimates. Returns {conditions: [...], passed: bool}."""
    conds = []
    for s in DECIDING:
        conds.append(_cond(f"f1_vs_base_plus_0.05_{s}", t1[s]["f1"], base[s]["f1"] + 0.05, ">="))
    for s in DECIDING:
        conds.append(_cond(f"f1_above_base_initial_prompt_{s}", t1[s]["f1"], prompt[s]["f1"], ">"))
    for s in DECIDING:
        conds.append(
            _cond(
                f"precision_vs_base_minus_0.02_{s}",
                t1[s]["precision"],
                base[s]["precision"] - 0.02,
                ">=",
            )
        )
    conds.append(
        _cond(
            "repetitions_kept_ami_test",
            t1["ami_test"]["repetitions_kept"]["recall"],
            base["ami_test"]["repetitions_kept"]["recall"],
            ">=",
        )
    )
    conds.append(
        _cond(
            "librispeech_wer_vs_base_plus_1.0",
            t1["librispeech_test_clean"]["wer"],
            base["librispeech_test_clean"]["wer"] + 1.0,
            "<=",
        )
    )
    return {"conditions": conds, "passed": all(c["met"] for c in conds)}


def _cond(name: str, value: float, line: float, op: str) -> dict[str, Any]:
    met = {">=": value >= line, ">": value > line, "<=": value <= line}[op]
    return {"name": name, "value": value, "line": line, "op": op, "met": bool(met)}
