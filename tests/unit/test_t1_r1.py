"""T1-r1 (PR-4, ADR 0029): chunk planner, dev pick, pass rule, paired bootstrap, build, ingest,
compare. No model, GPU, network or corpus data: fakes only."""

from __future__ import annotations

import ast
import base64
import importlib.util
import itertools
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
T1 = REPO / "training" / "t1_whisper_verbatim"
sys.path.insert(0, str(T1))
sys.path.insert(0, str(REPO / "training"))


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


chunks: Any = _load("chunks", T1 / "chunks.py")
t1pick: Any = _load("t1pick", T1 / "t1pick.py")
sc: Any = _load("e3score", T1 / "e3score.py")
ct1: Any = _load("colab_t1", T1 / "colab_t1.py")

TR = yaml.safe_load((REPO / "configs" / "training.yaml").read_text(encoding="utf-8"))["t1"]
EV = yaml.safe_load((REPO / "configs" / "evaluation.yaml").read_text(encoding="utf-8"))


# --- ADR 0029 §1-2, §5, §9: the owner's values ----------------------------------------------


def test_r1_settings_are_the_owners_q1_values():
    r1 = TR["r1"]
    assert r1["lora"] == {
        "r": 64,
        "alpha": 64,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
    }
    o = r1["optimiser"]
    assert (o["learning_rate"], o["lr_scheduler"], o["warmup_steps"]) == (1e-4, "linear", 50)
    assert r1["batch"] == {"per_device": 8, "gradient_accumulation": 2}
    assert (r1["precision"], r1["epochs"], r1["seed"]) == ("fp16", 3, 20261128)
    assert r1["data"] == ["disfluencyspeech_train"]  # r1: DisfluencySpeech only


def test_pick_rule_and_publish_repo_are_the_owners_q2_q3():
    p = TR["pick"]
    assert p["sets"] == ["disfluencyspeech_validation", "ami_dev"]  # dev only, never test
    assert (p["score"], p["tie_within"], p["tie_break"]) == (
        "mean_filler_f1",
        0.005,
        "lower_mean_wer",
    )
    assert TR["publish"]["repo"] == "Heemo-Beemo/presenceai-whisper-small-verbatim"
    assert TR["publish"]["token_env"] == "HF_TOKEN"  # noqa: S105 (a variable name)


def test_adr_0029_is_proposed_and_names_the_rules():
    text = (REPO / "docs" / "adr" / "0029-t1-r1.md").read_text(encoding="utf-8")
    assert "Status: **proposed**" in text
    assert "never by re-running r1 after its test numbers are seen" in text
    assert "the line is not moved" in text


# --- chunk planner (ADR 0028 §12, ADR 0029 §8) ----------------------------------------------


def _check_cover(cs, duration, max_s=30.0):
    assert cs[0][0] == 0.0 and cs[-1][1] == pytest.approx(duration)
    for (_, e), (s, _) in itertools.pairwise(cs):
        assert s == pytest.approx(e)  # contiguous: no audio dropped
    assert all(e - s <= max_s + 1e-9 and e > s for s, e in cs)


def test_chunks_join_regions_up_to_30_s_and_cut_in_the_middle_of_the_pause():
    regs = [(1.0, 10.0), (12.0, 20.0), (40.0, 50.0)]
    cs = chunks.plan_chunks(regs, 60.0)
    _check_cover(cs, 60.0)
    assert len(cs) == 2
    assert cs[0][1] == pytest.approx(30.0)  # middle of the 20-40 s pause
    assert cs[0][0] <= 1.0 and cs[1][0] <= 40.0


def test_chunks_never_exceed_30_s_and_a_long_pause_gets_its_own_chunk():
    regs = [(0.0, 25.0), (100.0, 125.0)]
    cs = chunks.plan_chunks(regs, 125.0)
    _check_cover(cs, 125.0)
    assert cs[0] == (0.0, 30.0)  # 5 s of slack, all to the right (none needed on the left)
    middle = [c for c in cs if c[0] >= 30.0 and c[1] <= 95.0]
    assert middle and all(e - s <= 30.0 for s, e in middle)  # the pause part nobody could take


def test_a_region_over_30_s_is_cut_every_30_s():
    cs = chunks.plan_chunks([(0.0, 70.0)], 70.0)
    _check_cover(cs, 70.0)
    assert cs == [(0.0, 30.0), (30.0, 60.0), (60.0, 70.0)]


def test_no_speech_still_covers_the_recording():
    assert chunks.plan_chunks([], 65.0) == [(0.0, 30.0), (30.0, 60.0), (60.0, 65.0)]
    assert chunks.plan_chunks([], 0.0) == []


def test_every_region_lies_inside_one_chunk():
    regs = [(0.5, 3.0), (3.2, 9.0), (9.1, 29.0), (31.0, 33.0), (33.5, 61.0), (70.0, 71.0)]
    cs = chunks.plan_chunks(regs, 80.0)
    _check_cover(cs, 80.0)
    for s, e in regs:
        if e - s <= 30.0:
            assert any(a <= s and e <= b + 1e-9 for a, b in cs), (s, e)


# --- dev pick (ADR 0029 §5) -----------------------------------------------------------------


def _cand(epoch, f_ds, f_ami, w_ds, w_ami):
    return {
        "epoch": epoch,
        "f1": {"disfluencyspeech_validation": f_ds, "ami_dev": f_ami},
        "wer": {"disfluencyspeech_validation": w_ds, "ami_dev": w_ami},
    }


def test_pick_takes_the_highest_mean_f1():
    r = t1pick.pick([_cand(1, 0.6, 0.6, 9, 9), _cand(2, 0.9, 0.5, 9, 9)], 0.005)
    assert r["picked_epoch"] == 2 and r["tied_epochs"] == [2]


def test_pick_tie_goes_to_lower_mean_wer_over_both_sets_then_earlier_epoch():
    # means 0.700 and 0.704: a tie; epoch 2 has the lower mean WER (DS-val alone would say 1)
    cands = [_cand(1, 0.7, 0.7, 5.0, 20.0), _cand(2, 0.708, 0.7, 6.0, 15.0)]
    assert t1pick.pick(cands, 0.005)["picked_epoch"] == 2
    same = [_cand(1, 0.7, 0.7, 10, 10), _cand(3, 0.7, 0.7, 10, 10)]
    assert t1pick.pick(same, 0.005)["picked_epoch"] == 1
    far = [_cand(1, 0.7, 0.7, 1, 1), _cand(2, 0.71, 0.71, 50, 50)]  # 0.01 apart: no tie
    assert t1pick.pick(far, 0.005)["picked_epoch"] == 2


def test_pick_refuses_a_missing_value():
    with pytest.raises(ValueError):
        t1pick.pick([_cand(1, None, 0.5, 1, 1)], 0.005)


# --- pass rule (ADR 0026 §4) and paired bootstrap -------------------------------------------


def _m(ami_f1, ami_p, pf_f1, pf_p, rep, wer):
    return {
        "ami_test": {"f1": ami_f1, "precision": ami_p, "repetitions_kept": {"recall": rep}},
        "podcastfillers_test": {"f1": pf_f1, "precision": pf_p},
        "librispeech_test_clean": {"wer": wer},
    }


B5_BASE = _m(0.079, 0.84, 0.031, 0.78, 0.171, 3.89)
B5_PROMPT = _m(0.815, 0.91, 0.038, 0.81, 0.468, 4.06)


def test_pass_rule_passes_only_if_every_condition_holds():
    good = _m(0.85, 0.90, 0.20, 0.80, 0.50, 4.2)
    assert t1pick.pass_rule(good, B5_BASE, B5_PROMPT)["passed"]
    at_line = _m(0.815, 0.90, 0.20, 0.80, 0.50, 4.2)  # equal to base + prompt is not "above"
    r = t1pick.pass_rule(at_line, B5_BASE, B5_PROMPT)
    assert not r["passed"]
    failed = [c["name"] for c in r["conditions"] if not c["met"]]
    assert failed == ["f1_above_base_initial_prompt_ami_test"]
    worse_wer = _m(0.85, 0.90, 0.20, 0.80, 0.50, 4.9)
    assert not t1pick.pass_rule(worse_wer, B5_BASE, B5_PROMPT)["passed"]
    low_p = _m(0.85, 0.81, 0.20, 0.80, 0.50, 4.2)  # 0.84 - 0.02 = 0.82 > 0.81
    assert not t1pick.pass_rule(low_p, B5_BASE, B5_PROMPT)["passed"]


def test_paired_bootstrap_resamples_the_same_units():
    a = {"m1": sc.Counts(5, 1, 1), "m2": sc.Counts(3, 0, 2)}
    assert sc.paired_bootstrap(a, a, sc.f1_of, 200, 1) == (0.0, 0.0)
    b = {"m1": sc.Counts(1, 1, 5), "m2": sc.Counts(0, 0, 5)}
    lo, hi = sc.paired_bootstrap(a, b, sc.f1_of, 200, 1)
    assert lo is not None and hi is not None and 0 < lo <= hi
    with pytest.raises(ValueError):
        sc.paired_bootstrap(a, {"m1": sc.Counts()}, sc.f1_of, 10, 1)


# --- build, ingest, compare -----------------------------------------------------------------


def _between(body: str, start: str) -> str:
    return body.split(start, 1)[1].split('""")', 1)[0]


def test_build_embeds_the_e3_script_and_orders_the_stages(tmp_path):
    nb = json.loads(ct1.build("t", tmp_path).read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    body = (tmp_path / "t1_run.py").read_text(encoding="utf-8")
    ast.parse(body)
    src = "".join(cells)
    assert "'T4' in gpu" in src and "MyDrive/presenceai_p4/t1r1_t/t1r1" in src
    setup = next(c for c in cells if "pip install" in c)
    assert "unsloth==2026.11.12 unsloth_zoo==2026.11.12" in setup
    assert setup.index("unsloth==") < setup.index("faster-whisper==")
    assert "userdata.get('HF_TOKEN')" in src and "DELETE_RUNTIME = False" in cells[-1]
    mods = json.loads(_between(body, 'MODULES = json.loads(r"""'))
    assert set(mods) == {"e3_run.py", "t1train.py", "t1merge.py", "chunks.py", "t1pick.py"}
    e3body = base64.b64decode(mods["e3_run.py"]).decode("utf-8")
    ast.parse(e3body)
    assert "def convert(src, dst)" in e3body and 'f"units/{cond}.json"' in e3body
    cfg = json.loads(_between(body, 'CONFIG = json.loads(r"""'))
    assert cfg["training"]["t1"] == TR and cfg["e3"]["per_window"] == EV["e3"]["per_window"]
    vad = yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))
    assert cfg["app_vad"] == vad["speech"]["vad"]  # the app's Silero settings (ADR 0028 §12)
    run = body.split("def main():", 1)[1]  # the calls, in the order main makes them
    order = [
        run.index(s)
        for s in (
            "e3.stage_ds_check()",
            "e3.stage_model(keep_source=True)",
            "stage_train()",
            "stage_dev_data()",
            "stage_pick()",
            "stage_publish(epoch)",
            "stage_load_check(epoch)",
            "e3.stage_data()",  # test data only after the pick is written
            'e3.stage_asr("t1"',
            "stage_per_window(t1_model)",
        )
    ]
    assert order == sorted(order)
    assert ".private:" in body  # nothing is uploaded to a public repo
    assert "print(tok" not in body and "log(tok" not in body


def test_ingest_keeps_text_out_of_git(tmp_path, monkeypatch):
    ci = sys.modules["colab_ingest"]
    root = tmp_path / "build" / "t1r1"
    files = {
        "hyps/t1.jsonl": b"{}",
        "hyps/pick_epoch_1.jsonl": b"{}",
        "items/ami_dev.json": b"[]",
        "e3_metrics.json": b"{}",
        "units/t1.json": b"{}",
        "pick.json": b"{}",
    }
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    (root / "run_info.json").write_text(
        json.dumps({"status": "ok", "sha256": ci.run_info_table(root)})
    )
    z = tmp_path / "t1r1_t.zip"
    shutil.make_archive(str(z.with_suffix("")), "zip", tmp_path / "build", "t1r1")
    monkeypatch.setattr(ct1, "RESULTS", tmp_path / "res")
    monkeypatch.setattr(ct1, "DATA", tmp_path / "data")
    assert ct1.ingest("t", z)["status"] == "ok"
    assert (tmp_path / "data" / "t" / "hyps" / "pick_epoch_1.jsonl").is_file()
    assert (tmp_path / "data" / "t" / "items" / "ami_dev.json").is_file()
    assert (tmp_path / "res" / "r1_t" / "units" / "t1.json").is_file()


def _write_run(tmp_path, monkeypatch, t1_items_same=True):
    """A tiny b5 and T1 run on disk (fake numbers, two meetings, one episode)."""
    b5r, b5d = tmp_path / "b5res", tmp_path / "b5data"
    res, data = tmp_path / "res" / "r1_t", tmp_path / "data" / "t"
    ami = [
        {"id": "a", "meeting": "M1", "deciding": True, "text": "um yes", "repetitions": []},
        {"id": "b", "meeting": "M2", "deciding": True, "text": "uh no", "repetitions": []},
    ]
    events = {"E": [[1.0, 1.2]]}
    for d in (b5d / "items", data / "items"):
        d.mkdir(parents=True)
        (d / "ami_test.json").write_text(json.dumps(ami))
        (d / "podcastfillers_test_events.json").write_text(json.dumps(events))
    if not t1_items_same:
        (data / "items" / "ami_test.json").write_text(json.dumps(ami[:1]))
    (b5d / "hyps").mkdir()
    rows = {
        "base": [("a", "yes"), ("b", "no")],
        "base_initial_prompt": [("a", "um yes"), ("b", "uh no")],
    }
    for cond, hs in rows.items():
        lines = [json.dumps({"dataset": "ami", "id": i, "text": t, "words": []}) for i, t in hs]
        lines.append(json.dumps({"dataset": "podcastfillers", "id": "E", "text": "", "words": []}))
        (b5d / "hyps" / f"{cond}.jsonl").write_text("\n".join(lines) + "\n")
    b5r.mkdir()
    met = {
        "base": {**_m(0.0, 0.0, 0.0, 0.0, 0.2, 4.0)},
        "base_initial_prompt": {**_m(1.0, 1.0, 0.0, 0.0, 0.3, 4.1)},
    }
    met["base"]["ami_test"] |= {"tp": 0, "fp": 0, "fn": 2}
    met["base"]["podcastfillers_test"] |= {"tp": 0, "fp": 0, "fn": 1}
    met["base_initial_prompt"]["ami_test"] |= {"tp": 2, "fp": 0, "fn": 0}
    met["base_initial_prompt"]["podcastfillers_test"] |= {"tp": 0, "fp": 0, "fn": 1}
    (b5r / "e3_metrics.json").write_text(json.dumps(met))
    (res / "units").mkdir(parents=True)
    (res / "e3_metrics.json").write_text(json.dumps({"t1": _m(1.0, 1.0, 1.0, 1.0, 0.4, 4.2)}))
    units = {
        "ami_test": {"M1": [1, 0, 0], "M2": [1, 0, 0]},
        "podcastfillers_test": {"E": [1, 0, 0]},
    }
    (res / "units" / "t1.json").write_text(json.dumps(units))
    for name, val in (
        ("B5_RESULTS", b5r),
        ("B5_DATA", b5d),
        ("RESULTS", tmp_path / "res"),
        ("DATA", tmp_path / "data"),
    ):
        monkeypatch.setattr(ct1, name, val)
    return res


def test_compare_recomputes_base_and_applies_the_pass_rule(tmp_path, monkeypatch):
    res = _write_run(tmp_path, monkeypatch)
    out = ct1.compare("t")
    names = {c["name"]: c["met"] for c in out["pass_rule"]["conditions"]}
    assert names["f1_above_base_initial_prompt_ami_test"] is False  # 1.0 is not above 1.0
    assert out["pass_rule"]["passed"] is False
    assert "t1_minus_base_ami_test" in out["paired_bootstrap_95"]
    assert (res / "pass_rule.json").is_file()


def test_compare_refuses_items_that_differ_from_b5(tmp_path, monkeypatch):
    _write_run(tmp_path, monkeypatch, t1_items_same=False)
    with pytest.raises(ValueError, match="like for like"):
        ct1.compare("t")


# --- merge (ADR 0029 §3) --------------------------------------------------------------------

t1merge: Any = _load("t1merge", T1 / "t1merge.py")


def test_merge_adds_scaled_lora_to_the_targets_and_keeps_every_other_tensor():
    import numpy as np

    rng = np.random.default_rng(0)
    q = "model.encoder.layers.0.self_attn.q_proj"
    base = {
        f"{q}.weight": rng.standard_normal((4, 3)).astype(np.float32),
        f"{q}.bias": np.zeros(4, np.float32),
        "model.encoder.layers.0.self_attn.k_proj.weight": np.ones((4, 3), np.float32),
    }
    a, b = (
        rng.standard_normal((2, 3)).astype(np.float32),
        rng.standard_normal((4, 2)).astype(np.float32),
    )
    adapter = {f"base_model.model.{q}.lora_A.weight": a, f"base_model.model.{q}.lora_B.weight": b}
    targets = ["q_proj", "v_proj"]
    cfg = {"r": 2, "lora_alpha": 4, "target_modules": ["v_proj", "q_proj"]}
    scaling = t1merge.check_config(cfg, 2, 4, targets)
    merged, changed = t1merge.merge(base, adapter, scaling, ["q_proj", "v_proj"])
    assert scaling == 2.0 and changed == [f"{q}.weight"]
    assert set(merged) == set(base)  # the base's names, so the app's converter sees the app's model
    np.testing.assert_allclose(merged[f"{q}.weight"], base[f"{q}.weight"] + 2.0 * b @ a, rtol=1e-6)
    assert merged[f"{q}.bias"] is base[f"{q}.bias"]
    assert all(v.dtype == np.float32 for v in merged.values())


def test_merge_refuses_anything_but_the_runs_plain_lora():
    import numpy as np

    with pytest.raises(ValueError, match="r/alpha"):
        t1merge.check_config({"r": 8, "lora_alpha": 64}, 64, 64, ["q_proj", "v_proj"])
    with pytest.raises(ValueError, match="use_dora"):
        t1merge.check_config({"r": 64, "lora_alpha": 64, "use_dora": True}, 64, 64, ["q_proj"])
    k = "model.decoder.layers.0.fc1"
    base = {f"{k}.weight": np.zeros((2, 2), np.float32)}
    ad = {
        f"{k}.lora_A.weight": np.zeros((1, 2), np.float32),
        f"{k}.lora_B.weight": np.zeros((2, 1), np.float32),
    }
    with pytest.raises(ValueError, match="not a target module"):
        t1merge.merge(base, ad, 1.0, ["q_proj", "v_proj"])
    with pytest.raises(ValueError, match="no LoRA weights"):
        t1merge.merge(base, {}, 1.0, ["q_proj"])


# --- Kaggle (owner 2026-11-28, ADR 0029 §11) ------------------------------------------------


def test_kaggle_build_runs_the_same_script_and_keeps_everything_in_the_output(tmp_path):
    import hashlib

    ci = sys.modules["colab_ingest"]
    nb = json.loads(ct1.build("t", tmp_path / "k", platform="kaggle").read_text(encoding="utf-8"))
    ct1.build("t", tmp_path / "c")
    kbody = (tmp_path / "k" / "t1_run.py").read_bytes()
    assert kbody == (tmp_path / "c" / "t1_run.py").read_bytes()  # same script, same SHA-256
    sha = hashlib.sha256(kbody).hexdigest()
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    src = "".join(cells)
    assert "google.colab" not in src and "drive.mount" not in src
    assert "UserSecretsClient().get_secret('HF_TOKEN')" in src
    assert cells[0].index("CUDA_VISIBLE_DEVICES'] = '0'") >= 0  # one GPU, set first
    assert "P4_OUT'] = '/kaggle/working/t1r1'" in cells[0]
    assert "P4_WORKD'] = '/kaggle/working/work'" in cells[0]  # outside the zipped folder
    assert "P4_CACHE'] = '/kaggle/working/presenceai_data'" in cells[0]
    assert f"== {sha!r}" in cells[1]  # resume only results of this same build
    assert "'/kaggle/input/**/presenceai_data'" in cells[1]
    assert f"!timeout {ct1.TIMEOUT_S} python /kaggle/working/t1_run.py" in src
    assert "make_archive('/kaggle/working/t1r1_t', 'zip', '/kaggle/working', 't1r1')" in cells[-1]
    assert ci.KAGGLE_WORKING == "/kaggle/working"
    with pytest.raises(ValueError):
        ct1.build("t", tmp_path / "x", platform="laptop")


def test_cloud_guard_accepts_colab_or_kaggle_only(monkeypatch):
    fetch = _load("fetch_public_data", REPO / "scripts" / "fetch_public_data.py")
    with pytest.raises(fetch.NotOnColabError):
        fetch.require_cloud({})
    with pytest.raises(fetch.NotOnColabError):  # the variable alone is not enough
        fetch.require_cloud({"KAGGLE_KERNEL_RUN_TYPE": "Batch"}) if not Path(
            "/kaggle/working"
        ).is_dir() else fetch.require_cloud({})
    monkeypatch.setattr(fetch.Path, "is_dir", lambda self: True)
    fetch.require_cloud({"KAGGLE_KERNEL_RUN_TYPE": "Batch"})
    fetch.require_cloud({"COLAB_RELEASE_TAG": "x"})
