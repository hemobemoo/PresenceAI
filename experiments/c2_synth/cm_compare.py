"""Central-message methods compared on the owner's C2 dev labels (ADR 0025; spaCy only).

    uv run python experiments/c2_synth/cm_compare.py   # -> experiments/results/c2/cm_compare.json

Methods A (textbook thesis features, `cm_cues_v2.yaml`), B (position only) and C0 (lemma
centrality), with the grids, objective and folds pre-registered in ADR 0025 item 11: 5-fold
cross-validation grouped by topic, parameters tuned on four folds and scored on the fifth,
counts summed over the held-out folds. Reference rows: "always yes" and the current detector
(thesis cue v1). The e5 methods (C, D) are not here: they are a real-model run (ADR 0025).

Counts only: no transcript text, no sentence, no key reaches the results file. Reads the owner's
sheet (`data/c2/dev_labels/`), which the code uses but never prints.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import random
import statistics
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "experiments" / "results" / "c2" / "cm_compare.json"
CUES = HERE / "cm_cues_v2.yaml"
FOLD_SEED = 20261126  # ADR 0025 item 11
N_FOLDS = 5
OUTCOMES = ("hit", "wrong_sentence", "fn", "fp", "tn")
CONTENT = frozenset({"NOUN", "PROPN", "VERB", "ADJ"})
NUMBER_WORDS = frozenset({"two", "three", "four", "five"})
WINDOWS = (10, 15, 20, 25, 30)


def _load_dev_labels() -> Any:
    name = "c2_dev_labels"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "dev_labels.py")
    if spec is None or spec.loader is None:
        raise ImportError("dev_labels.py not found")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dl = _load_dev_labels()


@dataclass(frozen=True)
class Sent:
    text: str
    first_word: int
    toks: tuple[Any, ...]
    plain: str  # normalised words, space-padded for whole-phrase matching


@dataclass(frozen=True)
class Talk:
    key: str
    topic: int
    owner_stated: bool
    owner_sentence: str | None
    sents: tuple[Sent, ...]
    n_words: int


def talk_from_text(
    key: str, topic: int, text: str, stated: bool, sentence: str | None, *, parser: Any
) -> Talk:
    from presenceai.settings import get_settings
    from presenceai.text.document import build, plain

    doc = build(
        dl.text_transcript(text),
        parser,
        get_settings().speech.pauses,
        opening_pct=20,
        closing_pct=15,
    )  # fmt: skip  (shares unused here)
    sents = tuple(Sent(s.text, s.first_word, s.tokens, f" {plain(s.text)} ") for s in doc.sentences)
    return Talk(key, topic, stated, sentence, sents, doc.n_words)


def candidates(t: Talk, window: float) -> list[int]:
    """Sentences starting in the opening `window` % of the words; never a question."""
    end = max(1, round(t.n_words * window / 100))
    return [
        i for i, s in enumerate(t.sents) if s.first_word < end and not s.text.rstrip().endswith("?")
    ]


@cache
def cue_lists() -> dict[str, tuple[str, ...]]:
    from presenceai.text.document import cues, plain

    data = yaml.safe_load(CUES.read_text("utf-8"))
    out = {k: tuple(f" {plain(p)} " for p in v) for k, v in data.items()}
    out["v1"] = tuple(f" {plain(p)} " for p in cues()["thesis"])
    return out


def _phrase(s: Sent, family: str) -> bool:
    return any(p in s.plain for p in cue_lists()[family])


def _preview(s: Sent) -> bool:
    toks = s.toks
    for a, b in itertools.pairwise(toks):
        number = a.text.lower() in NUMBER_WORDS or a.text in {"2", "3", "4", "5"}
        if number and b.pos == "NOUN" and b.text.lower().endswith("s"):
            return True
    return ":" in s.text or (s.text.count(",") >= 2 and " and " in s.plain)


def _family(s: Sent, family: str) -> bool:
    return _preview(s) if family == "preview" else _phrase(s, family)


def method_a(t: Talk, p: dict[str, Any]) -> int | None:
    hits = [i for i in candidates(t, p["window"])
            if any(_family(t.sents[i], f) for f in p["families"])]  # fmt: skip
    if not hits:
        return None
    return hits[0] if p["pick"] == "first" else hits[-1]


def method_b(t: Talk, p: dict[str, Any]) -> int | None:
    c = candidates(t, p["window"])
    if not c:
        return None
    if p["rule"] == "last":
        return c[-1]
    return c[p["rule"] - 1] if len(c) >= p["rule"] else None


def _content(s: Sent) -> frozenset[str]:
    return frozenset(x.lemma.casefold() for x in s.toks if x.pos in CONTENT and not x.is_stop)


def centrality(t: Talk) -> list[float]:
    sets = [_content(s) for s in t.sents]
    out = []
    for i, own in enumerate(sets):
        if not own:
            out.append(0.0)
            continue
        others = frozenset().union(*(x for j, x in enumerate(sets) if j != i))
        out.append(len(own & others) / len(own))
    return out


def method_c0(t: Talk, p: dict[str, Any]) -> int | None:
    c = candidates(t, p["window"])
    if not c:
        return None
    scores = centrality(t)
    best = max(c, key=lambda i: (scores[i], -i))
    return best if scores[best] - statistics.median(scores) >= p["margin"] else None


METHODS: dict[str, Callable[[Talk, dict[str, Any]], int | None]] = {
    "A": method_a, "B": method_b, "C0": method_c0,
}  # fmt: skip
FAMILIES = (("purpose",), ("purpose", "preview"), ("purpose", "preview", "claim"))
GRIDS: dict[str, list[dict[str, Any]]] = {
    "A": [
        {"window": w, "families": f, "pick": k}
        for w in WINDOWS
        for f in FAMILIES
        for k in ("first", "last")
    ],
    "B": [{"window": w, "rule": r} for w in WINDOWS for r in (1, 2, 3, "last")],
    "C0": [{"window": w, "margin": round(m * 0.05, 2)} for w in WINDOWS for m in range(9)],
}  # fmt: skip  (order = ADR 0025 item 11; ties go to the earlier entry)


def outcome(t: Talk, i: int | None) -> str:
    if i is None:
        return "fn" if t.owner_stated else "tn"
    if not t.owner_stated:
        return "fp"
    same = t.owner_sentence is not None and dl._same(t.sents[i].text, t.owner_sentence)
    return "hit" if same else "wrong_sentence"


def evaluate(talks: list[Talk], method: str, p: dict[str, Any]) -> Counter[str]:
    return Counter(outcome(t, METHODS[method](t, p)) for t in talks)


def _objective(c: Counter[str]) -> tuple[int, int]:
    return (c["hit"] + c["tn"], -c["fp"])


def tune(talks: list[Talk], method: str) -> dict[str, Any]:
    """The grid entry with the most correct items; ties: fewer false positives, then earlier."""
    best, best_key = GRIDS[method][0], None
    for p in GRIDS[method]:
        k = _objective(evaluate(talks, method, p))
        if best_key is None or k > best_key:
            best, best_key = p, k
    return best


def folds(talks: list[Talk], seed: int = FOLD_SEED) -> list[list[Talk]]:
    topics = sorted({t.topic for t in talks})
    random.Random(seed).shuffle(topics)  # noqa: S311 - reproducible folds, not security
    fold_of = {topic: i % N_FOLDS for i, topic in enumerate(topics)}
    return [[t for t in talks if fold_of[t.topic] == f] for f in range(N_FOLDS)]


def _row(c: Counter[str]) -> dict[str, Any]:
    return {"headline": c["hit"], "false_positives": c["fp"],
            "held_out": {k: c[k] for k in OUTCOMES}}  # fmt: skip


def _jsonable(p: dict[str, Any]) -> dict[str, Any]:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in p.items()}


def cross_validate(talks: list[Talk], method: str) -> dict[str, Any]:
    total: Counter[str] = Counter()
    per_fold = []
    parts = folds(talks)
    for k, test in enumerate(parts):
        train = [t for j, part in enumerate(parts) if j != k for t in part]
        p = tune(train, method)
        c = evaluate(test, method, p)
        total += c
        per_fold.append({"params": _jsonable(p), "items": len(test), "headline": c["hit"],
                         "false_positives": c["fp"]})  # fmt: skip
    return {**_row(total), "per_fold": per_fold,
            "fit_on_all_dev_reported_not_adopted": _jsonable(tune(talks, method))}  # fmt: skip


def compare(talks: list[Talk]) -> dict[str, Any]:
    always = Counter("fp" if not t.owner_stated else "fn" for t in talks)  # no sentence chosen
    v1 = evaluate(talks, "A", {"window": 20, "families": ("v1",), "pick": "first"})
    return {
        "owner": {
            "yes": sum(t.owner_stated for t in talks),
            "no": sum(not t.owner_stated for t in talks),
        },
        "folds": N_FOLDS,
        "fold_seed": FOLD_SEED,
        "methods": {m: cross_validate(talks, m) for m in METHODS},
        "reference": {
            "always_yes": {**_row(always), "note": "says stated, names no sentence"},
            "current_v1": {**_row(v1), "note": "no tuning, all items"},
        },
        "note": "headline = stated with the owner's sentence (strict match: a lower bound)",
    }


def load_talks(parser: Any) -> list[Talk]:
    out_dir = dl.OUT_DIR
    sheet = yaml.safe_load((out_dir / "sheet.yaml").read_text("utf-8"))
    key = json.loads((out_dir / "key.json").read_text("utf-8"))
    st = dl.status(sheet, key)
    if st["flags"] or st["done"] < len(sheet):
        raise SystemExit("the dev sheet must be finished and unflagged")
    talks = []
    for s in sheet:
        stated = dl.yes_no(s["main_point_stated"]) is True
        topic = int(key[s["key"]]["id"].rsplit("_", 1)[1])
        talks.append(talk_from_text(s["key"], topic, s["text"], stated,
                                    s["main_point_sentence"] if stated else None,
                                    parser=parser))  # fmt: skip
    return talks


def main() -> int:
    from presenceai.text.nlp import spacy_parser

    result = compare(load_talks(spacy_parser()))
    RESULTS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, r in {**result["methods"], **result["reference"]}.items():
        print(f"{name:>10}: headline {r['headline']:>2}/{result['owner']['yes']}, "
              f"false positives {r['false_positives']:>2}/{result['owner']['no']}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
