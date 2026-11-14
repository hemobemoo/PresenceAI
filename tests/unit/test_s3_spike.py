"""Tests for experiments/s3_unsloth_whisper: normalisation, WER, fillers, pass rule, Colab build
and ingest."""

import ast
import hashlib
import importlib
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "s3_unsloth_whisper"))
common: Any = importlib.import_module("common")
colab: Any = importlib.import_module("colab")
CFG = common.load_config()
SHA = re.compile(r"[0-9a-f]{40}")
NL = "\n"


def test_config_pins_notebook_dataset_model_and_seeds():
    assert SHA.fullmatch(CFG["source_notebook"]["commit"])
    assert SHA.fullmatch(CFG["dataset"]["revision"])
    assert SHA.fullmatch(CFG["base_model"]["revision"])
    assert set(CFG["seeds"]) == {"data", "lora", "trainer"}
    assert CFG["dataset"]["text_column"] == "transcript_a"
    assert (CFG["dataset"]["n_train"], CFG["dataset"]["n_test"]) == (200, 10)
    assert CFG["pass"]["max_wer_increase_points"] == 10.0
    assert CFG["runtime"]["platform"] == "colab" and "kaggle" not in CFG


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


def _injected(script: str, name: str) -> Any:
    opening = f'{name} = json.loads(r"""'
    start = script.index(opening) + len(opening)
    return json.loads(script[start : script.index('""")', start)])


def test_colab_notebook_carries_config_and_no_kaggle_or_secrets(tmp_path: Path):
    nb_path = colab.build(CFG, "test-run", tmp_path)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    first_line, script = cells[1].split(NL, 1)
    assert first_line == "%%writefile /content/train_s3.py"
    assert script == (tmp_path / "train_s3.py").read_text(encoding="utf-8")
    assert "__CONFIG_JSON__" not in script and "__BUILD_JSON__" not in script
    ast.parse(script)  # still valid Python after injection
    assert _injected(script, "CONFIG") == json.loads(json.dumps(CFG))
    assert _injected(script, "BUILD")["run_id"] == "test-run"
    assert 'OUT = "/content/s3"' in script and "/kaggle" not in script
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in script
    assert "S3 DONE: status=" in script and "__doc__" not in script
    every = NL.join(cells)
    assert "drive.mount" not in every and "userdata" not in every
    assert not re.search(r"token|api_key|secret", json.dumps(CFG), re.IGNORECASE)
    assert f"!timeout {CFG['runtime']['timeout_s']} python /content/train_s3.py" in every
    assert "T4" in cells[0] and "runtime.unassign()" in cells[-1]


def test_build_is_lf_and_the_integrity_cell_matches_it(tmp_path: Path):
    colab.build(CFG, "test-run", tmp_path)
    data = (tmp_path / "train_s3.py").read_bytes()
    assert b"\r" not in data
    nb = json.loads((tmp_path / "S3_test-run.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert hashlib.sha256(data).hexdigest() in cells[2]


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The built Colab script imported as a module (main() does not run on import)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # restored after the test
    colab.build(CFG, "test-run", tmp_path / "build")
    spec = importlib.util.spec_from_file_location("built_s3", tmp_path / "build" / "train_s3.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fill_missing_config_copies_only_what_the_merge_left_out(built: Any, tmp_path: Path):
    base, merged = tmp_path / "base", tmp_path / "merged"
    base.mkdir()
    merged.mkdir()
    (base / "config.json").write_text('{"model_type": "whisper"}', encoding="utf-8")
    (base / "generation_config.json").write_text("{}", encoding="utf-8")
    (merged / "generation_config.json").write_text('{"kept": true}', encoding="utf-8")
    fixes = built.fill_missing_config(str(merged), str(base))
    assert fixes == ["config.json copied from the base model into the merged model"]
    assert json.loads((merged / "config.json").read_text(encoding="utf-8"))["model_type"]
    assert json.loads((merged / "generation_config.json").read_text(encoding="utf-8")) == {
        "kept": True
    }
    assert built.fill_missing_config(str(merged), str(base)) == []


def test_export_copies_only_existing_tokenizer_files_and_fills_the_base(built: Any, tmp_path: Path):
    base, ft_out, base_out = tmp_path / "base", tmp_path / "ct2_ft", tmp_path / "ct2_base"
    for d in (base, ft_out, base_out):
        d.mkdir()
    (base / "preprocessor_config.json").write_text("{}", encoding="utf-8")  # no tokenizer.json
    assert built.copy_files_arg(str(base)) == " --copy_files preprocessor_config.json"
    assert built.copy_files_arg(str(base_out)) == ""
    (ft_out / "tokenizer.json").write_text('{"tok": 1}', encoding="utf-8")
    (base_out / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    fixes = built.fill_missing_files(str(ft_out), str(base_out), built.TOKENIZER_FILES)
    assert fixes == [f"tokenizer.json copied from ct2_ft into {base_out}"]
    assert (base_out / "tokenizer.json").read_text(encoding="utf-8") == '{"tok": 1}'
    assert built.fill_missing_files(str(ft_out), str(base_out), built.TOKENIZER_FILES) == []


def test_built_checksums_feed_ingest_verify(built: Any, tmp_path: Path):
    s3 = tmp_path / "s3"
    (s3 / "ct2_ft").mkdir(parents=True)
    (s3 / "ct2_ft" / "model.bin").write_bytes(b"x")
    (s3 / "train_log.csv").write_text("step", encoding="utf-8")
    table = built.checksums(str(s3))
    assert set(table) == {"ct2_ft/model.bin", "train_log.csv"}
    (s3 / "run_info.json").write_text(json.dumps({"sha256": table}), encoding="utf-8")
    assert colab.verify(s3)["sha256"] == table


def test_load_wav_reads_16k_mono_pcm_and_rejects_other_formats(tmp_path: Path):
    import wave

    check_local: Any = importlib.import_module("check_local")
    ok, bad = tmp_path / "ok.wav", tmp_path / "bad.wav"
    for path, rate in ((ok, 16000), (bad, 8000)):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"\x00\x00\xff\x7f\x00\x80")  # 0, 32767, -32768
    audio = check_local.load_wav(ok)
    assert audio.dtype.name == "float32"
    assert audio.tolist() == pytest.approx([0.0, 32767 / 32768, -1.0])
    with pytest.raises(ValueError, match="16 kHz mono"):
        check_local.load_wav(bad)


def _fake_run(root: Path, tamper: bool = False) -> Path:
    s3 = root / "s3"
    files = {
        "train_log.csv": b"step,loss",
        "ct2_base/model.bin": b"base",
        "ct2_ft/model.bin": b"ft",
        "clips/test_00.wav": b"RIFF",
        "clips/refs.csv": b"clip,duration_s,reference",
    }
    table = {}
    for rel, data in files.items():
        (s3 / rel).parent.mkdir(parents=True, exist_ok=True)
        (s3 / rel).write_bytes(data)
        table[rel] = {"bytes": len(data), "sha256": colab.sha256(s3 / rel)}
    (s3 / "run_info.json").write_text(json.dumps({"status": "ok", "sha256": table}), "utf-8")
    if tamper:
        (s3 / "ct2_ft/model.bin").write_bytes(b"FT")
    zp = root / "s3_run.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for f in s3.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(root).as_posix())
    return zp


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("DATA_DIR", "MODELS_DIR", "RESULTS"):
        monkeypatch.setattr(common, name, tmp_path / name.lower())
    return tmp_path


def test_ingest_moves_verified_files(dirs: Path):
    out = colab.ingest("r1", _fake_run(dirs / "dl"))
    assert out == {"run_id": "r1", "status": "ok", "files_verified": 5, "model_files": 2}
    assert (common.MODELS_DIR / "r1" / "ft" / "model.bin").read_bytes() == b"ft"
    assert (common.DATA_DIR / "r1" / "clips" / "refs.csv").exists()
    assert (common.RESULTS / "r1_run_info.json").exists()
    assert (common.RESULTS / "r1_models_manifest.csv").exists()
    assert not (common.DATA_DIR / "r1" / "raw").exists()


def test_ingest_refuses_a_checksum_mismatch(dirs: Path):
    with pytest.raises(ValueError, match=r"checksum mismatch: ct2_ft/model\.bin"):
        colab.ingest("r2", _fake_run(dirs / "dl", tamper=True))
    assert not (common.MODELS_DIR / "r2").exists() and not common.RESULTS.exists()


def test_safe_extract_rejects_paths_outside(tmp_path: Path):
    zp = tmp_path / "bad.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="unsafe path"):
        colab.safe_extract(zp, tmp_path / "out")
