"""experiments/c1_kokoro: notebook build, the word check, generation with a fake TTS, ingest.

Kokoro is never installed here (ADR 0014 §1); the generator's own code runs with FakeTTS."""

import base64
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fake_tts import FakeTTS

from presenceai.evalconfig import REPO_ROOT, load_evaluation
from presenceai.schemas.c1 import ClipTruth
from presenceai.synth import scripts as sc

HERE = REPO_ROOT / "experiments" / "c1_kokoro"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


colab = _load("c1_colab", HERE / "colab.py")
gen = _load("c1_generate", HERE / "colab" / "generate_c1.py")
CFG = load_evaluation().c1


def test_build_writes_a_checked_notebook(tmp_path: Path):
    nb_path = colab.build("t1", "timing", tmp_path)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    body = (tmp_path / "generate_c1.py").read_text(encoding="utf-8")
    cells = ["".join(c["source"]) for c in nb["cells"]]
    assert any(c.startswith("%%writefile /content/generate_c1.py") for c in cells)
    assert any(hashlib.sha256(body.encode()).hexdigest() in c for c in cells)
    assert any("C1_MODE=timing" in c for c in cells)
    assert any("sys.version_info < (3, 13)" in c for c in cells)  # kokoro 0.9.4 needs < 3.13
    assert 'PACKAGE_B64 = "__PACKAGE_B64__"' not in body  # the package was injected
    assert 'r"""__CONFIG_JSON__"""' not in body
    compile(body, "generate_c1.py", "exec")


def test_the_embedded_package_holds_exactly_the_c1_code():
    data = base64.b64decode(colab.package_b64())
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert set(z.namelist()) == set(colab.PACKAGE_FILES)
    assert colab.package_b64() == colab.package_b64()  # stable bytes, stable SHA-256


def test_the_generator_reads_no_secret_and_pins_every_download():
    text = (HERE / "colab" / "generate_c1.py").read_text(encoding="utf-8")
    for bad in ("HF_TOKEN", '".env', "dotenv", "MEDIA_KEY", "OPENROUTER", "drive.mount", "getpass"):
        assert bad not in text
    assert 'revision=gen["revision"]' in text
    assert "EspeakFallback = GuardFallback" in text


# --- the word check and the espeak guard ------------------------------------------------------


@dataclass
class Tok:
    text: str
    _: Any = None


@dataclass
class Rating:
    rating: int | None


@dataclass
class FakeG2P:
    unknown: set[str] = field(default_factory=set)

    def __call__(self, text: str) -> tuple[str, list[Tok]]:
        import re

        toks = []
        for w in re.findall(r"[A-Za-z]+|[,.?!]", text):
            if w.lower() in self.unknown:
                gen.GuardFallback(british=False)(Tok(w))  # what misaki would do
                toks.append(Tok(w, Rating(None)))
            else:
                toks.append(Tok(w, Rating(4)))
        return "", toks


def test_word_check_lists_words_the_word_lists_miss():
    gen.GuardFallback.calls.clear()
    gen.GuardFallback.strict = False
    scripts = sc.load_scripts(REPO_ROOT / CFG.scripts)
    check = gen.check_words(
        scripts, CFG, lambda code: FakeG2P({"morning"} if code == "a" else set())
    )
    assert check["missing"]["a"] == ["morning"]
    assert check["missing"]["b"] == []
    assert check["words"]["a"]["morning"] is None and check["words"]["b"]["um"] == 4


def test_guard_stops_synthesis_when_strict():
    gen.GuardFallback.strict = True
    try:
        with pytest.raises(RuntimeError, match="espeak-ng would be needed"):
            gen.GuardFallback(british=True)(Tok("zzyzx"))
    finally:
        gen.GuardFallback.strict = False


# --- generation with a fake TTS, then ingest ------------------------------------------------


def test_timing_selection_covers_five_voices_and_five_rates():
    scripts = sc.load_scripts(REPO_ROOT / CFG.scripts)
    picks = gen.select(scripts, CFG, "timing")
    assert len(picks) == CFG.timing_run_clips
    assert len({scripts[i].voice for i, _ in picks}) == 5
    assert sorted(r for _, r in picks) == [0, 1, 2, 3, 4]
    assert len(gen.select(scripts, CFG, "full")) == 300


def test_generate_and_ingest_round_trip(tmp_path: Path):
    out = tmp_path / "c1"
    result = gen.generate(FakeTTS(), "timing", out)
    assert len(result["clips"]) + len(result["dropped"]) == 5
    assert result["projected_all_clips_s"] is not None
    for c in result["clips"]:
        truth = ClipTruth.model_validate_json((out / "truth" / f"{c['clip_id']}.json").read_text())
        assert CFG.wpm_range[0] <= truth.wpm <= CFG.wpm_range[1]
        assert (out / "clips" / f"{c['clip_id']}.flac").exists()
    (out / "lexicon_check.json").write_text("{}", encoding="utf-8")
    gen.finish({"status": "ok", "build": {}}, out)
    zip_path = Path(shutil.make_archive(str(tmp_path / "c1_t1"), "zip", tmp_path, "c1"))
    summary = colab.ingest(
        "t1", zip_path, data=tmp_path / "data", results=tmp_path / "results",
        lexicon=tmp_path / "resources" / "lexicon_check.json",
    )  # fmt: skip
    assert summary["status"] == "ok" and summary["clips"] == len(result["clips"])
    assert (tmp_path / "results" / "t1_run_info.json").exists()
    assert (tmp_path / "results" / "t1_clips_manifest.csv").exists()
    assert (tmp_path / "resources" / "lexicon_check.json").exists()


def test_ingest_rejects_a_changed_file(tmp_path: Path):
    out = tmp_path / "c1"
    out.mkdir()
    (out / "a.txt").write_text("one", encoding="utf-8")
    gen.finish({"status": "ok", "build": {}}, out)
    (out / "a.txt").write_text("two", encoding="utf-8")
    zip_path = Path(shutil.make_archive(str(tmp_path / "c1_t2"), "zip", tmp_path, "c1"))
    with pytest.raises(ValueError, match="checksum"):
        colab.ingest("t2", zip_path, data=tmp_path / "data", results=tmp_path / "results",
                     lexicon=tmp_path / "lex.json")  # fmt: skip
