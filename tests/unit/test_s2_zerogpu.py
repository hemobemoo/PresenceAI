"""Tests for experiments/s2_zerogpu: config pins, pass rule, Space build (ADR 0007)."""

import ast
import importlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "s2_zerogpu"))
for name in ("common", "prepare"):  # the S3 tests also import a module called `common`
    sys.modules.pop(name, None)
common: Any = importlib.import_module("common")
prepare: Any = importlib.import_module("prepare")
sys.modules.pop("common", None)
sys.modules.pop("prepare", None)
CFG = common.load_config()
SHA = re.compile(r"[0-9a-f]{40}")


def test_config_pins_model_dataset_and_cuda12_stack():
    assert SHA.fullmatch(CFG["model"]["source_revision"])
    assert CFG["model"]["source"] == "openai/whisper-large-v3-turbo"
    assert SHA.fullmatch(CFG["clip"]["revision"])
    reqs = CFG["space"]["requirements"]
    assert "ctranslate2==4.9.0" in reqs and "faster-whisper==1.2.1" in reqs
    torch_pin = next(r for r in reqs if r.startswith("torch=="))
    major, minor = (int(x) for x in torch_pin.split("==")[1].split(".")[:2])
    assert (2, 8) <= (major, minor) <= (2, 13)  # versions ZeroGPU lists
    assert CFG["space"]["gpu_duration_s"] == 60
    assert CFG["pass"] == {"max_median_warm_gpu_s": 60.0, "max_any_call_gpu_s": 60.0}
    assert (CFG["runs"]["cold"], CFG["runs"]["warm"]) == (1, 3)


def _calls(warm: list[float], cold: float = 50.0, texts: tuple[str, ...] = ()) -> list[Any]:
    out = [common.Call("cold", cold, "text")]
    for i, s in enumerate(warm):
        out.append(common.Call("warm", s, texts[i] if texts else "text"))
    return out


@pytest.mark.parametrize(
    ("calls", "outcome"),
    [
        (_calls([30, 40, 59.9]), "pass"),
        (_calls([30, 61, 62]), "fail"),  # median over 60 (and calls over the reservation)
        (_calls([30, 40, 50], cold=60.0), "pass"),  # exactly the reservation is allowed
        (_calls([30, 40, 50], cold=60.5), "fail"),  # cold call over the reservation
        (_calls([30, 40, 50], texts=("a", " ", "c")), "fail"),  # an empty transcript
        ([*_calls([30, 40, 50]), common.Call("warm", math.nan, "", ok=False)], "fail"),
    ],
)
def test_decide(calls: list[Any], outcome: str):
    assert common.decide(calls, 60.0, 60.0)["outcome"] == outcome


def test_decide_median_is_strictly_below_60():
    assert common.decide(_calls([59, 60, 60]), 60.0, 60.0)["outcome"] == "fail"


def test_space_build_refuses_without_a_pinned_model_revision(tmp_path: Path):
    cfg = json.loads(json.dumps(CFG))
    cfg["model"]["repo_revision"] = None
    with pytest.raises(ValueError, match="repo_revision"):
        prepare.build_space(cfg, tmp_path / "space")


def test_space_build_carries_the_shared_loader_and_no_ids_or_secrets(tmp_path: Path):
    cfg = json.loads(json.dumps(CFG))
    cfg["model"]["repo_revision"] = "0" * 40
    out = prepare.build_space(cfg, tmp_path / "space")
    files = {p.name for p in out.iterdir()}
    assert files == {"app.py", "audio.py", "requirements.txt", "README.md", "s2_config.json"}
    loader = REPO / "src" / "presenceai" / "speech" / "audio.py"
    assert (out / "audio.py").read_bytes() == loader.read_bytes()
    conf = json.loads((out / "s2_config.json").read_text(encoding="utf-8"))
    assert conf["model_repo_revision"] == "0" * 40 and conf["gpu_duration_s"] == 60
    assert (out / "requirements.txt").read_text(encoding="utf-8").splitlines() == cfg["space"][
        "requirements"
    ]
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "sdk: gradio" in card and 'python_version: "3.12"' in card
    everything = "\n".join(p.read_text(encoding="utf-8") for p in out.iterdir())
    assert not re.search(r"hf_[A-Za-z0-9]{20,}|api_key|password", everything)
    assert "repo_id" not in conf and "model_repo" not in conf  # the id is a Space variable


def test_space_app_reserves_60s_reads_ids_from_env_and_uses_the_loader():
    src = (REPO / "experiments" / "s2_zerogpu" / "space" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imports = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    first = imports[0]
    assert isinstance(first, ast.Import) and first.names[0].name == "spaces"  # before torch
    assert 'os.environ["S2_MODEL_REPO"]' in src and 'CFG["gpu_duration_s"]' in src
    assert "audio.transcribe(" in src and "print(" not in src
