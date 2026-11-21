"""C2 design (P2-e, ADR 0021): prompts, balance, split, independence from the cue lists, checks."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from typing import Any

import yaml

from presenceai.evalconfig import REPO_ROOT
from presenceai.text.document import cues, find


def _load(name: str) -> Any:
    path = REPO_ROOT / "experiments" / "c2_synth" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"c2_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"c2_{name}"] = module
    spec.loader.exec_module(module)
    return module


build = _load("build_prompts")
checks = _load("check_c2")
TOPICS = list(yaml.safe_load((REPO_ROOT / "resources/c2/topics.yaml").read_text("utf-8"))["topics"])
ROWS = build.build(TOPICS)


def test_five_groups_of_80_with_hook_crossed():
    assert len(ROWS) == 400
    assert Counter(r["group"] for r in ROWS) == dict.fromkeys(build.GROUPS, 80)
    assert Counter((r["group"], r["hook"]) for r in ROWS) == {
        (g, h): 40 for g in build.GROUPS for h in (True, False)
    }


def test_split_30_70_by_topic_with_hook_balanced():
    parts = Counter((r["group"], r["part"], r["hook"]) for r in ROWS)
    for g in build.GROUPS:
        assert parts[(g, "dev", True)] == parts[(g, "dev", False)] == 12
        assert parts[(g, "test", True)] == parts[(g, "test", False)] == 28
    by_topic = {}
    for r in ROWS:
        by_topic.setdefault(r["topic_index"], set()).add(r["part"])
    assert all(len(p) == 1 for p in by_topic.values())  # a topic never spans both parts


def test_prompts_are_not_built_from_the_cue_lists():
    for r in ROWS:
        instructions = r["prompt"].replace(r["topic"], "")
        for kind in cues():
            assert not find(instructions, kind), (r["id"], kind, find(instructions, kind))


def test_committed_prompts_are_the_built_ones():
    lines = (REPO_ROOT / "resources/c2/prompts.jsonl").read_text("utf-8").splitlines()
    assert [json.loads(x) for x in lines if x] == ROWS
    assert build.build(TOPICS) == ROWS  # seeded: reproducible


def test_format_checks():
    ok = " ".join(["word"] * 420)
    assert checks.problems(ok, "control") == []
    assert checks.problems("too short", "control") == ["words 2"]
    assert "format" in checks.problems("# Title\n" + ok, "control")
    assert "format" in checks.problems("- point\n" + ok, "control")
    assert "format" in checks.problems("<think>x</think>\n" + ok, "control")
    assert checks.problems(ok + " 45 percent", "no_supporting_material") == ["digits"]
    assert checks.problems(ok + " 45 percent", "control") == []


gen = _load("generate_c2")


def test_generator_quality_checks_use_no_cue_lexicons():
    source = (REPO_ROOT / "experiments/c2_synth/generate_c2.py").read_text("utf-8")
    # no PresenceAI imports (no cue lexicons, no detectors) and no reference to the cue file
    assert "from presenceai" not in source and "import presenceai" not in source
    assert "text_cues" not in source
    ok = " ".join(["word"] * 420)
    assert gen.text_problems(ok) == []
    assert gen.MAX_NEW_TOKENS == 800  # a safety net; a capped transcript is always rejected
    assert not hasattr(gen, "QUESTIONS")  # the model's yes/no check is not a filter (pilot 2b)
    assert gen.text_problems("The main idea of my talk. " + ok) == ["echo"]
    assert gen.text_problems("Our central idea is simple. " + ok) == ["echo"]
    assert gen.text_problems("**Title**\n" + ok) == ["format"]
    assert gen.text_problems(" ".join(["word"] * 560)) == ["words 560"]
    assert gen.text_problems(" ".join(["word"] * 260)) == []  # floor 250 (owner, pilot 3)
    assert gen.text_problems(" ".join(["word"] * 240)) == ["words 240"]


hand = _load("hand_check")


def _rows() -> list[dict[str, object]]:
    return [{**r, "text": f"text of {r['id']}"} for r in ROWS]


def test_hand_check_sample_is_stratified_test_only_and_blinded():
    sheet, key = hand.draw(_rows())
    assert len(sheet) == 50
    assert Counter(key[s["key"]]["group"] for s in sheet) == dict.fromkeys(build.GROUPS, 10)
    parts = {r["id"]: r["part"] for r in ROWS}
    assert all(parts[k["id"]] == "test" for k in key.values())
    for s in sheet:  # the sheet shows no planted information
        assert set(s) == {"key", "text", *hand.LABELS}
        assert all(s[lab] is None for lab in hand.LABELS)
    assert hand.draw(_rows()) == (sheet, key)  # seeded


def test_hand_check_score_is_the_planted_label_error_rate():
    sheet, key = hand.draw(_rows())
    for s in sheet:  # the owner agrees with the design everywhere ...
        s.update(hand.planted(key[s["key"]]))
    assert hand.score(sheet, key)["item_error_rate"] == 0
    s0 = sheet[0]  # ... except one item, where the planted conclusion state is wrong
    s0["conclusion"] = not s0["conclusion"]
    result = hand.score(sheet, key)
    assert result["items_with_any_planted_label_wrong"] == 1
    assert result["label_error_rate"]["conclusion"] == 0.02
    sheet[1]["hook"] = None
    try:
        hand.score(sheet, key)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("an unfilled sheet must not be scored")
