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
    ok = " ".join(["word"] * 320)
    assert checks.problems(ok, "control") == []
    assert checks.problems("too short", "control") == ["words 2"]
    assert "format" in checks.problems("# Title\n" + ok, "control")
    assert "format" in checks.problems("- point\n" + ok, "control")
    assert "format" in checks.problems("<think>x</think>\n" + ok, "control")
    assert checks.problems(ok + " 45 percent", "no_supporting_material") == ["digits"]
    assert checks.problems(ok + " 45 percent", "control") == []
