"""C1 split into development and test parts (ADR 0011, ADR 0014 §5).

By script, so all renders of one script share a part. Each voice's scripts are split
`dev_fraction` / rest with one seeded `random.Random`, voices in sorted order. Every script is
rendered across all rate bands, so voice stratification also covers every band; `check_counts`
proves it on the measured rates and stops on any empty cell.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Sequence

from presenceai.evalconfig import C1Settings, E2Split
from presenceai.schemas.c1 import ClipTruth, Part, Script
from presenceai.synth.c1 import rate_band

CountKey = tuple[Part, str, str]  # (part, "band" | "voice", value)


def split_scripts(scripts: Sequence[Script], cfg: E2Split) -> dict[str, Part]:
    rng = random.Random(cfg.seed)  # noqa: S311 (a reproducible split, not security)
    parts: dict[str, Part] = {}
    for voice in sorted({s.voice for s in scripts}):
        ids = sorted(s.id for s in scripts if s.voice == voice)
        dev = set(rng.sample(ids, round(len(ids) * cfg.dev_fraction)))
        parts |= {i: ("dev" if i in dev else "test") for i in ids}
    return parts


def assign(truths: Iterable[ClipTruth], parts: dict[str, Part]) -> list[tuple[ClipTruth, Part]]:
    return [(t, parts[t.script_id]) for t in truths]


def counts(table: Iterable[tuple[ClipTruth, Part]], cfg: C1Settings) -> Counter[CountKey]:
    c: Counter[CountKey] = Counter()
    for t, part in table:
        c[(part, "band", rate_band(t.wpm, cfg.rate_bands) or "outside")] += 1
        c[(part, "voice", t.voice)] += 1
    return c


def check_counts(c: Counter[CountKey], cfg: C1Settings) -> None:
    bands = [f"{lo:g}-{hi:g}" for lo, hi in cfg.rate_bands]
    empty = [
        f"{part} {kind} {value}"
        for part in ("dev", "test")
        for kind, values in (("band", bands), ("voice", list(cfg.voices)))
        for value in values
        if c[(part, kind, value)] == 0  # type: ignore[index]
    ]
    if empty:
        raise ValueError("empty cell(s) in the C1 split: " + ", ".join(empty))


def format_counts(c: Counter[CountKey]) -> str:
    lines = ["part  kind   value       clips"]
    for (part, kind, value), n in sorted(c.items()):
        lines.append(f"{part:<5} {kind:<6} {value:<11} {n}")
    return "\n".join(lines)
