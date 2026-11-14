"""Tests for experiments/s3_unsloth_whisper: normalisation, WER, fillers, pass rule, job build."""

import ast
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "s3_unsloth_whisper"))
common: Any = importlib.import_module("common")
push: Any = importlib.import_module("push")
CFG = common.load_config()
SHA = re.compile(r"[0-9a-f]{40}")


def test_config_pins_notebook_dataset_model_and_seeds():
    assert SHA.fullmatch(CFG["source_notebook"]["commit"])
    assert SHA.fullmatch(CFG["dataset"]["revision"])
    assert SHA.fullmatch(CFG["base_model"]["revision"])
    assert set(CFG["seeds"]) == {"data", "lora", "trainer"}
    assert CFG["dataset"]["text_column"] == "transcript_a"
    assert (CFG["dataset"]["n_train"], CFG["dataset"]["n_test"]) == (200, 10)
    assert CFG["pass"]["max_wer_increase_points"] == 10.0


def test_normalise_keeps_fillers_and_apostrophes():
    assert common.normalise("Well, McNeil. Neil-, uh, I don't KNOW!") == (
        "well mcneil neil uh i don't know"
    )


def test_corpus_wer():
    assert common.corpus_wer(["a b c d"], ["a b c d"]) == 0
    assert common.corpus_wer(["a b c d"], ["a x c"]) == pytest.approx(50.0)  # 1 sub + 1 del
    assert common.corpus_wer(["a b", "c d e f"], ["a b", ""]) == pytest.approx(100 * 4 / 6)
    with pytest.raises(ValueError):
        common.corpus_wer(["a"], [])


def test_count_fillers():
    assert common.count_fillers("Uh, I um, well, uh-huh, hmm.", CFG["fillers"]) == 4


def _log(eval0: float, eval_end: float, first: float, last: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"step": 0, "loss": "", "eval_loss": eval0}]
    for s in range(1, 101):
        rows.append({"step": s, "loss": first if s <= 50 else last, "eval_loss": ""})
    rows.append({"step": 100, "loss": "", "eval_loss": eval_end})
    return rows


def test_loss_check():
    ok = common.loss_check(_log(2.0, 1.0, 3.0, 1.5), 10)
    assert ok.passed and ok.eval_loss_step0 == 2.0 and ok.eval_loss_final == 1.0
    assert not common.loss_check(_log(1.0, 1.2, 3.0, 1.5), 10).passed  # validation loss rose
    assert not common.loss_check(_log(2.0, 1.0, 1.5, 3.0), 10).passed  # training loss rose


def test_loss_check_needs_step0_and_enough_steps():
    rows = _log(2.0, 1.0, 3.0, 1.5)[1:]
    with pytest.raises(ValueError):
        common.loss_check(rows, 10)
    with pytest.raises(ValueError):
        common.loss_check(_log(2.0, 1.0, 3.0, 1.5)[:5], 10)


@pytest.mark.parametrize(
    ("loss_ok", "loads", "hyps", "base", "ft", "outcome"),
    [
        (True, True, ["a", "b"], 20.0, 29.9, "pass"),
        (True, True, ["a", "b"], 20.0, 30.0, "pass"),  # exactly 10 points: allowed
        (True, True, ["a", "b"], 20.0, 30.1, "fail"),
        (True, True, ["a", " "], 20.0, 20.0, "fail"),  # an empty transcript
        (True, False, [], 20.0, 20.0, "fail"),  # model did not load
        (False, True, ["a", "b"], 20.0, 15.0, "fail"),  # loss did not fall
    ],
)
def test_decide(loss_ok: bool, loads: bool, hyps: list[str], base: float, ft: float, outcome: str):
    loss = common.LossCheck(2.0, 1.0 if loss_ok else 3.0, 3.0, 1.0)
    d = common.decide(loss, loads, hyps, base, ft, CFG["pass"]["max_wer_increase_points"])
    assert d["outcome"] == outcome


def test_kaggle_job_is_private_gpu_and_carries_config(tmp_path: Path):
    out = push.build(CFG, "someone", tmp_path)
    meta = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == f"someone/{CFG['kaggle']['slug']}"
    assert meta["is_private"] is True and meta["enable_gpu"] is True
    assert meta["kernel_type"] == "script" and meta["code_file"] == "train_kernel.py"
    script = (out / "train_kernel.py").read_text(encoding="utf-8")
    assert "__CONFIG_JSON__" not in script
    ast.parse(script)  # still valid Python after injection
    start = script.index('CONFIG = json.loads(r"""') + len('CONFIG = json.loads(r"""')
    injected = json.loads(script[start : script.index('""")', start)])
    assert injected == json.loads(json.dumps(CFG))
    assert not re.search(r"token|api_key|secret", json.dumps(injected), re.IGNORECASE)
