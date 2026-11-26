"""Central-message method comparison on the owner's dev labels (ADR 0025 items 2, 3, 11)."""

from __future__ import annotations

import importlib.util
import json
import sys
from typing import Any

import pytest

from presenceai.evalconfig import REPO_ROOT


def _load(name: str) -> Any:
    path = REPO_ROOT / "experiments" / "c2_synth" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"c2_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"c2_{name}"] = module
    spec.loader.exec_module(module)
    return module


cm = _load("cm_compare")

FILL = " ".join(f"Garden {i} grows tomatoes and garden soil stays rich." for i in range(12))
HOOK = "Have you ever eaten a fresh tomato?"
MAIN = "My main point is that a garden saves money for every family."


@pytest.fixture
def talk(fake_parser: Any) -> Any:
    def make(text: str, *, stated: bool = True, sentence: str | None = MAIN, topic: int = 0,
             key: str = "d001") -> Any:  # fmt: skip
        return cm.talk_from_text(key, topic, text, stated, sentence if stated else None,
                                 parser=fake_parser)  # fmt: skip

    return make


def test_folds_group_by_topic_and_are_seeded(talk):
    talks = [talk(f"{MAIN} {FILL}", topic=t % 24, key=f"d{t:03d}") for t in range(48)]
    folds = cm.folds(talks)
    assert len(folds) == 5
    topics = [{x.topic for x in f} for f in folds]
    assert all(not (a & b) for i, a in enumerate(topics) for b in topics[i + 1 :])
    assert sum(len(f) for f in folds) == 48 and cm.folds(talks) == folds


def test_outcome_categories(talk):
    t = talk(f"{HOOK} {MAIN} {FILL}")
    no = talk(f"{HOOK} {FILL}", stated=False)
    assert cm.outcome(t, t.sents.index(next(s for s in t.sents if s.text == MAIN))) == "hit"
    assert cm.outcome(t, 0) == "wrong_sentence"
    assert cm.outcome(t, None) == "fn"
    assert cm.outcome(no, 0) == "fp" and cm.outcome(no, None) == "tn"


def test_b_takes_the_nth_or_last_candidate_and_skips_questions(talk):
    t = talk(f"{HOOK} {MAIN} {FILL}")
    first = cm.method_b(t, {"window": 30, "rule": 1})
    assert t.sents[first].text == MAIN  # the hook is a question, never a candidate
    last = cm.method_b(t, {"window": 30, "rule": "last"})
    assert last is not None and last > first


def test_a_finds_purpose_preview_and_claim_sentences(talk):
    t = talk(f"{HOOK} {MAIN} {FILL}")
    i = cm.method_a(t, {"window": 30, "families": ("purpose",), "pick": "first"})
    assert t.sents[i].text == MAIN
    preview = "Gardens help in three ways: food, money, and health."
    t2 = talk(f"{HOOK} {preview} {FILL}", sentence=preview)
    assert cm.method_a(t2, {"window": 30, "families": ("purpose",), "pick": "first"}) is None
    j = cm.method_a(t2, {"window": 30, "families": ("purpose", "preview"), "pick": "first"})
    assert t2.sents[j].text == preview
    claim = "Every family should grow a garden."
    t3 = talk(f"{claim} {FILL}", sentence=claim)
    k = cm.method_a(t3, {"window": 30, "families": ("purpose", "preview", "claim"),
                         "pick": "first"})  # fmt: skip
    assert t3.sents[k].text == claim


def test_c0_picks_the_most_central_sentence_and_honours_the_margin(talk):
    central = "The garden soil grows tomatoes."
    t = talk(f"Nobody expected rain yesterday. {central} {FILL}", sentence=central)
    i = cm.method_c0(t, {"window": 30, "margin": 0.0})
    assert t.sents[i].text == central
    assert cm.method_c0(t, {"window": 30, "margin": 1.01}) is None


def test_cross_validation_scores_every_item_once_held_out(talk):
    talks = [talk(f"{HOOK} {MAIN} {FILL}", topic=t % 10, key=f"d{t:03d}") for t in range(20)]
    talks += [talk(f"{HOOK} {FILL}", stated=False, topic=t % 10, key=f"n{t:03d}")
              for t in range(10)]  # fmt: skip
    r = cm.cross_validate(talks, "B")
    c = r["held_out"]
    assert sum(c[k] for k in cm.OUTCOMES) == 30
    assert len(r["per_fold"]) == 5 and all("params" in f for f in r["per_fold"])
    assert r["headline"] == c["hit"] and r["false_positives"] == c["fp"]


def test_compare_writes_counts_only_with_reference_rows(talk):
    talks = [talk(f"{HOOK} {MAIN} {FILL}", topic=t % 10, key=f"d{t:03d}") for t in range(10)]
    talks += [talk(f"{HOOK} {FILL}", stated=False, topic=t % 10, key=f"n{t:03d}")
              for t in range(5)]  # fmt: skip
    r = cm.compare(talks)
    assert set(r["methods"]) == {"A", "B", "C0"}
    assert set(r["reference"]) == {"always_yes", "current_v1"}
    assert r["reference"]["always_yes"]["headline"] == 0
    assert r["reference"]["always_yes"]["false_positives"] == 5
    assert r["reference"]["current_v1"]["headline"] == 10
    assert r["owner"] == {"yes": 10, "no": 5}
    dumped = json.dumps(r)
    assert "garden" not in dumped.lower() and "tomato" not in dumped.lower()


def test_average_is_the_mean_of_the_two_rates():
    from collections import Counter

    c = Counter(hit=18, tn=23, fp=5)
    assert round(cm.average(c, 90, 28), 3) == 0.511  # A in ADR 0025 item 12


def _vectors(t: Any, main_index: int) -> Any:
    import numpy as np

    v = np.tile(np.array([1.0, 0.0, 0.0]), (len(t.sents), 1))
    v[:, 1] = [0.5 if i % 2 else -0.5 for i in range(len(t.sents))]  # off-centre, both ways ...
    v[main_index] = [1.0, 0.0, 0.0]  # ... except the main point
    v[0] = [0.0, 0.0, 1.0]  # the opening question is far away
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return cm.Talk(t.key, t.topic, t.owner_stated, t.owner_sentence, t.sents, t.n_words, v)


def test_c_picks_the_sentence_closest_to_the_talk_and_needs_vectors(talk):
    t = talk(f"{HOOK} {MAIN} {FILL}")
    with pytest.raises(ValueError, match="e5"):
        cm.method_c(t, {"window": 30, "margin": 0.0})
    tv = _vectors(t, 1)
    i = cm.method_c(tv, {"window": 30, "margin": 0.0})
    assert tv.sents[i].text == MAIN
    assert cm.method_c(tv, {"window": 30, "margin": 1.0}) is None  # margin not reached


def test_d_or_falls_back_to_c_and_and_needs_both(talk):
    plain_main = "A garden saves money for every family."
    t = _vectors(talk(f"{HOOK} {plain_main} {FILL}", sentence=plain_main), 1)
    a = {"window": 30, "families": ("purpose",), "pick": "first"}
    c = {"window": 30, "margin": 0.0}
    assert cm.method_a(t, a) is None
    assert t.sents[cm.method_d(t, {"mode": "or", "a": a, "c": c})].text == plain_main
    assert cm.method_d(t, {"mode": "and", "a": a, "c": c}) is None


def test_compare_v2_applies_the_bars_and_the_e5_rule(talk):
    talks = [_vectors(talk(f"{HOOK} {MAIN} {FILL}", topic=t % 10, key=f"d{t:03d}"), 1)
             for t in range(10)]  # fmt: skip
    talks += [_vectors(talk(f"{HOOK} {FILL}", stated=False, sentence=None, topic=t % 10,
                            key=f"n{t:03d}"), 1) for t in range(5)]  # fmt: skip
    r = cm.compare_v2(talks, e5=True)
    assert set(r["spacy_only"]) == {f"{m} ({o})" for m in cm.SPACY for o in ("correct", "average")}
    assert set(r["e5"]) == {"C (average)", "D (average)"}
    need = r["e5_rule"]["average_at_least"]
    assert need == round(r["best_spacy_only"]["average"] + 0.10, 4)
    for row in r["e5"].values():
        ok = row["average"] >= need - 1e-9 and row["false_positives"] <= 7
        assert row["passes_e5_rule"] is ok
    # tiny fixture: 10 yes cannot reach 45 hits, so no method meets the minimum bar
    assert r["methods_meeting_minimum_bar"] == []
    assert "garden" not in json.dumps(r).lower()


def test_vectors_are_cached_and_reused(talk, tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setattr(cm, "E5_CACHE", tmp_path / "cache.npz")
    talks = [talk(f"{HOOK} {MAIN} {FILL}")]
    calls = []

    def embed(texts: list[str]) -> Any:
        calls.append(len(texts))
        return np.ones((len(texts), 3), dtype=np.float32)

    out, _ = cm.with_vectors(talks, embed)
    assert out[0].vecs.shape == (len(talks[0].sents), 3)
    _, seconds = cm.with_vectors(talks, embed)
    assert calls == [len(talks[0].sents)] and seconds == 0.0
