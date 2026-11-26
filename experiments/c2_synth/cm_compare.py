"""Central-message methods compared on the owner's C2 dev labels (ADR 0025).

    uv run python experiments/c2_synth/cm_compare.py         # spaCy only (A, B, C0)
    uv run python experiments/c2_synth/cm_compare.py --e5    # also C, D: a real-model run (e5)

Methods A (textbook thesis features, `cm_cues_v2.yaml`), B (position only), C0 (lemma
centrality), C (e5 centrality) and D (A with C), with grids, folds and objectives
pre-registered in ADR 0025 items 11–12: 5-fold cross-validation grouped by topic, parameters
tuned on four folds and scored on the fifth, counts summed over the held-out folds.

- Ranking (item 12): the average of right-sentence rate (hits / owner yes) and correct
  "not stated" rate (true negatives / owner no). A, B and C0 are tuned with both the item 11
  objective ("correct") and the item 12 one ("average"); C and D with "average" only.
- Bars (item 12): minimum bar for any method, e5 rule against the best spaCy-only average.
- e5 vectors ("query: " prefix) are computed once and cached in git-ignored
  `data/c2/cm_e5_cache.npz`; the wall time is printed and written.

Counts only: no transcript text, no sentence, no key reaches the results files. Reads the
owner's sheet (`data/c2/dev_labels/`), which the code uses but never prints.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import random
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "experiments" / "results" / "c2" / "cm_compare.json"  # step 3 record
RESULTS_V2 = ROOT / "experiments" / "results" / "c2" / "cm_compare_v2.json"  # item 12
E5_CACHE = ROOT / "data" / "c2" / "cm_e5_cache.npz"
CUES = HERE / "cm_cues_v2.yaml"
FOLD_SEED = 20261126  # ADR 0025 item 11
N_FOLDS = 5
OUTCOMES = ("hit", "wrong_sentence", "fn", "fp", "tn")
CONTENT = frozenset({"NOUN", "PROPN", "VERB", "ADJ"})
NUMBER_WORDS = frozenset({"two", "three", "four", "five"})
WINDOWS = (10, 15, 20, 25, 30)
MIN_HITS, MAX_FP = 45, 7  # ADR 0025 item 12 (owner): minimum bar, out of 90 and 28
E5_GAIN = 0.10  # ADR 0025 item 12: e5 average >= best spaCy-only average + 0.10


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
    vecs: Any = field(default=None, compare=False)  # e5 sentence vectors (n_sents, dim)


def talk_from_text(
    key: str, topic: int, text: str, stated: bool, sentence: str | None, *, parser: Any
) -> Talk:
    from presenceai.settings import get_settings
    from presenceai.text.document import build, plain

    doc = build(
        dl.text_transcript(text),
        parser,
        get_settings().speech.pauses,
        opening_pct=20,  # unused here: windows come from the grids
        closing_pct=15,
    )
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
    for a, b in itertools.pairwise(s.toks):
        number = a.text.lower() in NUMBER_WORDS or a.text in {"2", "3", "4", "5"}
        if number and b.pos == "NOUN" and b.text.lower().endswith("s"):
            return True
    return ":" in s.text or (s.text.count(",") >= 2 and " and " in s.plain)


def _family(s: Sent, family: str) -> bool:
    return _preview(s) if family == "preview" else _phrase(s, family)


def method_a(t: Talk, p: dict[str, Any]) -> int | None:
    hits = [
        i for i in candidates(t, p["window"]) if any(_family(t.sents[i], f) for f in p["families"])
    ]
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


def _central(t: Talk, scores: list[float], p: dict[str, Any]) -> int | None:
    c = candidates(t, p["window"])
    if not c:
        return None
    best = max(c, key=lambda i: (scores[i], -i))
    return best if scores[best] - statistics.median(scores) >= p["margin"] - 1e-12 else None


def method_c0(t: Talk, p: dict[str, Any]) -> int | None:
    return _central(t, centrality(t), p)


def e5_scores(t: Talk) -> list[float]:
    """Cosine of each sentence vector with the normalised mean vector of the talk."""
    v = np.asarray(t.vecs, dtype=np.float64)
    centre = v.mean(axis=0)
    centre /= np.linalg.norm(centre) or 1.0
    return [float(x) for x in v @ centre]


def method_c(t: Talk, p: dict[str, Any]) -> int | None:
    if t.vecs is None:
        raise ValueError("method C needs e5 vectors (run with --e5)")
    return _central(t, e5_scores(t), p)


def method_d(t: Talk, p: dict[str, Any]) -> int | None:
    a, c = _cached("A", t, p["a"]), _cached("C", t, p["c"])
    if p["mode"] == "or":
        return a if a is not None else c
    return a if a is not None and c is not None else None


METHODS: dict[str, Callable[[Talk, dict[str, Any]], int | None]] = {
    "A": method_a, "B": method_b, "C0": method_c0, "C": method_c, "D": method_d,
}  # fmt: skip
SPACY = ("A", "B", "C0")
E5 = ("C", "D")
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
    "C": [{"window": w, "margin": round(m * 0.005, 3)} for w in WINDOWS for m in range(13)],
}  # order = ADR 0025 items 11-12; ties go to the earlier entry
GRIDS["D"] = [{"mode": mode, "a": a, "c": c} for mode in ("or", "and")
              for a in GRIDS["A"] for c in GRIDS["C"]]  # fmt: skip

_MEMO: dict[tuple[str, int, str], tuple[Talk, int | None]] = {}  # keeps each talk alive


def _freeze(p: dict[str, Any]) -> str:
    return json.dumps(p, sort_keys=True, default=list)


def _cached(method: str, t: Talk, p: dict[str, Any]) -> int | None:
    k = (method, id(t), _freeze(p))
    if k not in _MEMO:
        _MEMO[k] = (t, METHODS[method](t, p))
    return _MEMO[k][1]


def outcome(t: Talk, i: int | None) -> str:
    if i is None:
        return "fn" if t.owner_stated else "tn"
    if not t.owner_stated:
        return "fp"
    same = t.owner_sentence is not None and dl._same(t.sents[i].text, t.owner_sentence)
    return "hit" if same else "wrong_sentence"


def evaluate(talks: list[Talk], method: str, p: dict[str, Any]) -> Counter[str]:
    return Counter(outcome(t, _cached(method, t, p)) for t in talks)


def average(c: Counter[str], yes: int, no: int) -> float:
    """ADR 0025 item 12: mean of hits / owner yes and true negatives / owner no."""
    return ((c["hit"] / yes if yes else 0.0) + (c["tn"] / no if no else 0.0)) / 2


def _key(c: Counter[str], objective: str, yes: int, no: int) -> tuple[float, int]:
    value = c["hit"] + c["tn"] if objective == "correct" else average(c, yes, no)
    return (value, -c["fp"])


def tune(talks: list[Talk], method: str, objective: str = "average") -> dict[str, Any]:
    """The best grid entry by `objective`; ties: fewer false positives, then the earlier one."""
    yes = sum(t.owner_stated for t in talks)
    no = len(talks) - yes
    best, best_key = GRIDS[method][0], None
    for p in GRIDS[method]:
        k = _key(evaluate(talks, method, p), objective, yes, no)
        if best_key is None or k > best_key:
            best, best_key = p, k
    return best


def folds(talks: list[Talk], seed: int = FOLD_SEED) -> list[list[Talk]]:
    topics = sorted({t.topic for t in talks})
    random.Random(seed).shuffle(topics)  # noqa: S311 - reproducible folds, not security
    fold_of = {topic: i % N_FOLDS for i, topic in enumerate(topics)}
    return [[t for t in talks if fold_of[t.topic] == f] for f in range(N_FOLDS)]


def _row(c: Counter[str], yes: int, no: int) -> dict[str, Any]:
    return {
        "headline": c["hit"],
        "false_positives": c["fp"],
        "average": round(average(c, yes, no), 4),
        "meets_minimum_bar": c["hit"] >= MIN_HITS and c["fp"] <= MAX_FP,
        "held_out": {k: c[k] for k in OUTCOMES},
    }


def _jsonable(p: Any) -> Any:
    if isinstance(p, dict):
        return {k: _jsonable(v) for k, v in p.items()}
    return list(p) if isinstance(p, tuple) else p


def cross_validate(talks: list[Talk], method: str, objective: str = "average") -> dict[str, Any]:
    yes = sum(t.owner_stated for t in talks)
    no = len(talks) - yes
    total: Counter[str] = Counter()
    per_fold = []
    parts = folds(talks)
    for k, test in enumerate(parts):
        train = [t for j, part in enumerate(parts) if j != k for t in part]
        p = tune(train, method, objective)
        c = evaluate(test, method, p)
        total += c
        per_fold.append({"params": _jsonable(p), "items": len(test), "headline": c["hit"],
                         "false_positives": c["fp"]})  # fmt: skip
    return {
        **_row(total, yes, no),
        "objective": objective,
        "per_fold": per_fold,
        "fit_on_all_dev_reported_not_adopted": _jsonable(tune(talks, method, objective)),
    }


def compare(talks: list[Talk], *, objective: str = "correct") -> dict[str, Any]:
    """The step 3 table (item 11 objective by default)."""
    yes = sum(t.owner_stated for t in talks)
    no = len(talks) - yes
    always = Counter("fp" if not t.owner_stated else "fn" for t in talks)  # no sentence chosen
    v1 = evaluate(talks, "A", {"window": 20, "families": ("v1",), "pick": "first"})
    return {
        "owner": {"yes": yes, "no": no},
        "folds": N_FOLDS,
        "fold_seed": FOLD_SEED,
        "methods": {m: cross_validate(talks, m, objective) for m in SPACY},
        "reference": {
            "always_yes": {**_row(always, yes, no), "note": "says stated, names no sentence"},
            "current_v1": {**_row(v1, yes, no), "note": "no tuning, all items"},
        },
        "note": "headline = stated with the owner's sentence (strict match: a lower bound)",
    }


def compare_v2(talks: list[Talk], *, e5: bool) -> dict[str, Any]:
    """ADR 0025 item 12: spaCy-only methods under both objectives, e5 methods under "average",
    the minimum bar and the e5 rule."""
    base = compare(talks, objective="correct")
    rows: dict[str, Any] = {f"{m} (correct)": base["methods"][m] for m in SPACY}
    rows |= {f"{m} (average)": cross_validate(talks, m, "average") for m in SPACY}
    best_name, best = max(
        [*((n, r["average"]) for n, r in rows.items()), ("current_v1",
         base["reference"]["current_v1"]["average"])],
        key=lambda x: x[1],
    )  # fmt: skip
    result: dict[str, Any] = {
        "owner": base["owner"], "folds": N_FOLDS, "fold_seed": FOLD_SEED,
        "spacy_only": rows, "reference": base["reference"],
        "best_spacy_only": {"row": best_name, "average": best},
        "minimum_bar": {"hits_at_least": MIN_HITS, "false_positives_at_most": MAX_FP},
        "note": "average = mean of hits/yes and true negatives/no; headline strict (lower bound)",
    }  # fmt: skip
    if e5:
        e5_rows = {f"{m} (average)": cross_validate(talks, m, "average") for m in E5}
        need = round(best + E5_GAIN, 4)
        for r in e5_rows.values():
            r["passes_e5_rule"] = r["average"] >= need - 1e-9 and r["false_positives"] <= MAX_FP
        result["e5"] = e5_rows
        result["e5_rule"] = {"average_at_least": need, "false_positives_at_most": MAX_FP}
    all_rows = {**rows, **result.get("e5", {})}
    result["methods_meeting_minimum_bar"] = sorted(
        n for n, r in all_rows.items() if r["meets_minimum_bar"]
    )
    return result


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


def with_vectors(talks: list[Talk], embed: Callable[[list[str]], Any]) -> tuple[list[Talk], float]:
    """Attach e5 vectors, from the cache when its sentences match; (talks, seconds embedding)."""
    texts = [s.text for t in talks for s in t.sents]
    digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
    seconds = 0.0
    vecs = None
    if E5_CACHE.is_file():
        data = np.load(E5_CACHE)
        vecs = data["vecs"] if str(data["digest"]) == digest else None
    if vecs is None:
        t0 = time.perf_counter()
        vecs = np.asarray(embed(texts), dtype=np.float32)
        seconds = time.perf_counter() - t0
        E5_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(E5_CACHE, vecs=vecs, digest=np.array(digest))
    out, i = [], 0
    for t in talks:
        n = len(t.sents)
        out.append(Talk(t.key, t.topic, t.owner_stated, t.owner_sentence, t.sents, t.n_words,
                        vecs[i : i + n]))  # fmt: skip
        i += n
    return out, seconds


def main(argv: list[str]) -> int:
    from presenceai.text.nlp import spacy_parser

    talks = load_talks(spacy_parser())
    e5 = "--e5" in argv
    meta: dict[str, Any] = {}
    if e5:
        from presenceai.knowledge.embed import E5Embedder

        t0 = time.perf_counter()
        model = E5Embedder()
        load_s = time.perf_counter() - t0
        talks, embed_s = with_vectors(talks, lambda xs: model.embed(xs, "query"))
        del model  # unload before scoring
        meta = {"e5_load_s": round(load_s, 1), "e5_embed_s": round(embed_s, 1),
                "sentences": sum(len(t.sents) for t in talks),
                "cache_used": embed_s == 0.0}  # fmt: skip
        print(f"e5: load {load_s:.1f} s, embed {embed_s:.1f} s, "
              f"{meta['sentences']} sentences (cache used: {meta['cache_used']})")  # fmt: skip
    result = compare_v2(talks, e5=e5)
    result["run"] = meta
    RESULTS_V2.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows = {**result["spacy_only"], **result.get("e5", {}), **result["reference"]}
    for name, r in rows.items():
        print(f"{name:>14}: avg {r['average']:.3f}  headline {r['headline']:>2}/"
              f"{result['owner']['yes']}  fp {r['false_positives']:>2}/{result['owner']['no']}"
              f"  bar {'yes' if r['meets_minimum_bar'] else 'no'}")  # fmt: skip
    print("best spaCy-only:", result["best_spacy_only"])
    if e5:
        print("e5 rule:", result["e5_rule"],
              {n: r["passes_e5_rule"] for n, r in result["e5"].items()})  # fmt: skip
    print("meeting the minimum bar:", result["methods_meeting_minimum_bar"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
