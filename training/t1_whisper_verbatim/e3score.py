"""E3 scoring (FINAL_10 E3, ADR 0012, ADR 0026 §4, ADR 0028). Pure, standard library only, so the
same code runs in tests and on Colab.

- **Fillers by alignment** (DisfluencySpeech test, AMI test): reference and hypothesis words are
  aligned by edit distance; an aligned pair of fillers is a true positive, a hypothesis filler
  aligned to anything else (or inserted) a false positive, a reference filler aligned to anything
  else (or deleted) a false negative. Fillers follow the T1/E3 rule (`t1text.is_filler`) on both
  sides; the filler type ("uh" vs "um") does not have to match.
- **Fillers by time** (PodcastFillers test): a hypothesis "uh"/"um" word matches an annotated
  Uh/Um event if their intervals overlap once the event is widened by the tolerance; one-to-one.
  Precision is a lower bound (the annotation may miss fillers).
- **Repetitions kept** (AMI test): a reference repetition (from the disfluency layer) is kept if
  every one of its words is aligned to the same word in the hypothesis.
- **WER** (LibriSpeech test-clean): lower case, punctuation removed, fillers kept.
- **Bootstrap** 95 % intervals, resampling whole units (clip, meeting, episode, speaker).
"""

from __future__ import annotations

import random
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t1text

_PUNCT = re.compile(r"[^\w\s']")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def prf(c: Counts) -> dict[str, float | None]:
    p = c.tp / (c.tp + c.fp) if c.tp + c.fp else None
    r = c.tp / (c.tp + c.fn) if c.tp + c.fn else None
    f = 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else None
    return {"precision": p, "recall": r, "f1": f}


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment (unit costs); pairs of (ref index, hyp index), None for a gap."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1
            )
    out: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


def filler_counts(
    ref_text: str, hyp_text: str, hesitations: Iterable[str], backchannels: Iterable[str]
) -> Counts:
    hes, back = list(hesitations), list(backchannels)
    ref, hyp = t1text.words(ref_text), t1text.words(hyp_text)

    def fil(w: str) -> bool:
        return t1text.is_filler(w, hes, back)

    tp = fp = fn = 0
    for i, j in align(ref, hyp):
        rf = i is not None and fil(ref[i])
        hf = j is not None and fil(hyp[j])
        tp += rf and hf
        fp += hf and not rf
        fn += rf and not hf
    return Counts(tp, fp, fn)


@dataclass(frozen=True)
class Event:
    start: float
    end: float


def hyp_filler_events(
    words: Iterable[tuple[str, float, float]], family: Iterable[str]
) -> list[Event]:
    """Hypothesis words in the scored filler family, as timed events."""
    fam = list(family)
    return [Event(s, e) for w, s, e in words if t1text.is_filler(w, fam, ())]


def time_match(hyp: Sequence[Event], ref: Sequence[Event], tol: float) -> Counts:
    """One-to-one: each hypothesis event takes the earliest unmatched reference event it overlaps
    once the reference is widened by `tol` on both sides."""
    refs = sorted(ref, key=lambda e: (e.start, e.end))
    used = [False] * len(refs)
    tp = 0
    for h in sorted(hyp, key=lambda e: (e.start, e.end)):
        for k, r in enumerate(refs):
            if not used[k] and h.start <= r.end + tol and h.end >= r.start - tol:
                used[k] = True
                tp += 1
                break
    return Counts(tp, len(hyp) - tp, len(refs) - tp)


def repetitions_kept(
    ref_words: Sequence[str], hyp_words: Sequence[str], events: Iterable[Sequence[int]]
) -> tuple[int, int]:
    """(kept, total) reference repetition events; indices point into ref_words."""
    norm_r = [t1text.words(w)[0] if t1text.words(w) else "" for w in ref_words]
    norm_h = [t1text.words(w)[0] if t1text.words(w) else "" for w in hyp_words]
    to_hyp = {i: j for i, j in align(norm_r, norm_h) if i is not None}
    kept = total = 0
    for ev in events:
        total += 1
        ok = all(
            to_hyp.get(i) is not None and norm_h[to_hyp[i]] == norm_r[i]  # type: ignore[index]
            for i in ev
        )
        kept += ok
    return kept, total


def normalise_wer(text: str, cfg: Mapping[str, bool]) -> list[str]:
    """ADR 0028: lower case, punctuation removed (apostrophes kept), fillers kept."""
    if cfg.get("remove_fillers"):
        raise ValueError("ADR 0028: fillers are never removed for WER")
    t = text.casefold() if cfg.get("lower_case", True) else text
    if cfg.get("remove_punctuation", True):
        t = _PUNCT.sub(" ", t.replace("’", "'"))
    return [w.strip("'") for w in _SPACE.split(t) if w.strip("'")]


def word_errors(ref: Sequence[str], hyp: Sequence[str]) -> int:
    return sum(1 for i, j in align(ref, hyp) if i is None or j is None or ref[i] != hyp[j])


def bootstrap(
    units: Mapping[str, tuple[float, float]] | Mapping[str, Counts],
    metric: Callable[[object], float | None],
    resamples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """Percentile 95 % interval of `metric` over unit resamples (whole units with replacement).
    Values are summed per resample: Counts add; (errors, words) tuples add element-wise."""
    keys = sorted(units)
    if not keys:
        return None, None
    rng = random.Random(seed)  # noqa: S311 (reproducible resampling, not security)
    vals: list[float] = []
    for _ in range(resamples):
        pick = [units[keys[rng.randrange(len(keys))]] for _ in keys]
        total: object
        if isinstance(pick[0], Counts):
            total = sum(pick[1:], start=pick[0])  # type: ignore[arg-type]
        else:
            total = tuple(sum(x) for x in zip(*pick, strict=True))  # type: ignore[arg-type]
        v = metric(total)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[round(0.975 * (len(vals) - 1))]
    return lo, hi


def f1_of(c: object) -> float | None:
    assert isinstance(c, Counts)  # noqa: S101 (internal contract)
    return prf(c)["f1"]


def wer_of(t: object) -> float | None:
    errors, words = t  # type: ignore[misc]
    return 100.0 * errors / words if words else None
