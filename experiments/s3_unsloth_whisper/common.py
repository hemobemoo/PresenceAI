"""Spike S3 shared logic: config, text normalisation, WER, filler counts, pass rule (ADR 0005)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = REPO / "experiments" / "configs" / "s3_unsloth_whisper.yaml"
RESULTS = REPO / "experiments" / "results" / "s3"
MODELS_DIR = REPO / "models" / "s3"  # git-ignored (models/*)
DATA_DIR = REPO / "data" / "s3"  # git-ignored (data/*)

_PUNCT = re.compile(r"[^\w\s']")
_SPACE = re.compile(r"\s+")


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data  # pyright: ignore[reportUnknownVariableType]


def normalise(text: str) -> str:
    """Lower case, punctuation and hyphens to spaces, apostrophes kept ("don't"), fillers kept."""
    text = _PUNCT.sub(" ", text.casefold().replace("’", "'"))
    words = [w.strip("'") for w in _SPACE.split(text)]
    return " ".join(w for w in words if w)


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def corpus_wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Word error rate in percent over all clips: total edits / total reference words x 100."""
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps differ in length")
    edits = words = 0
    for r, h in zip(refs, hyps, strict=True):
        rw, hw = normalise(r).split(), normalise(h).split()
        edits += edit_distance(rw, hw)
        words += len(rw)
    if words == 0:
        raise ValueError("references have no words")
    return 100.0 * edits / words


def count_fillers(text: str, fillers: Sequence[str]) -> int:
    words = normalise(text).split()
    wanted = {f.casefold() for f in fillers}
    return sum(w in wanted for w in words)


@dataclass(frozen=True)
class LossCheck:
    eval_loss_step0: float
    eval_loss_final: float
    train_loss_first: float
    train_loss_last: float

    @property
    def passed(self) -> bool:
        return (
            self.eval_loss_final < self.eval_loss_step0
            and self.train_loss_last < self.train_loss_first
        )


def loss_check(rows: Sequence[dict[str, Any]], window: int) -> LossCheck:
    """From train_log.csv rows: validation loss at step 0 vs last, train loss first vs last N."""

    def num(v: Any) -> float | None:
        return None if v in ("", None) else float(v)

    evals = [(int(r["step"]), num(r["eval_loss"])) for r in rows if num(r["eval_loss"]) is not None]
    train = [num(r["loss"]) for r in sorted(rows, key=lambda r: int(r["step"]))]
    train = [x for x in train if x is not None]
    if not evals or evals[0][0] != 0:
        raise ValueError("train_log has no validation loss at step 0")
    if len(train) < 2 * window:
        raise ValueError(f"need at least {2 * window} logged training losses, got {len(train)}")
    evals.sort()
    return LossCheck(
        eval_loss_step0=float(evals[0][1]),  # pyright: ignore[reportArgumentType]
        eval_loss_final=float(evals[-1][1]),  # pyright: ignore[reportArgumentType]
        train_loss_first=sum(train[:window]) / window,
        train_loss_last=sum(train[-window:]) / window,
    )


def decide(
    loss: LossCheck,
    loads: bool,
    ft_hyps: Sequence[str],
    base_wer: float,
    ft_wer: float,
    max_increase: float,
) -> dict[str, Any]:
    """ADR 0005: pass only if the loss falls, the model loads with non-empty transcripts for all
    test clips, and the fine-tuned WER is at most `max_increase` points above the base WER."""
    nonempty = loads and len(ft_hyps) > 0 and all(h.strip() for h in ft_hyps)
    wer_ok = ft_wer - base_wer <= max_increase
    return {
        "loss_falls": loss.passed,
        "eval_loss_step0": round(loss.eval_loss_step0, 4),
        "eval_loss_final": round(loss.eval_loss_final, 4),
        "train_loss_first": round(loss.train_loss_first, 4),
        "train_loss_last": round(loss.train_loss_last, 4),
        "ct2_loads_and_all_nonempty": nonempty,
        "base_wer": round(base_wer, 2),
        "ft_wer": round(ft_wer, 2),
        "wer_increase_points": round(ft_wer - base_wer, 2),
        "max_wer_increase_points": max_increase,
        "wer_ok": wer_ok,
        "outcome": "pass" if loss.passed and nonempty and wer_ok else "fail",
    }
