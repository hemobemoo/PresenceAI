"""E3 base run (PR-3): build the Colab notebook and ingest its zip (ADR 0028).

    uv run python training/t1_whisper_verbatim/colab_e3.py build  --run-id <id>
    uv run python training/t1_whisper_verbatim/colab_e3.py ingest --run-id <id> [--zip <path>]

build writes data/e3/colab/e3_run.py (config, build info and the modules it needs injected:
t1text, ami_words, e3score, splitzip, fetch_public_data and the app's text/normalize.py) and
data/e3/colab/E3BASE_<id>.ipynb. Needs a **T4 GPU**. Results (never datasets) go to the owner's
Drive (MyDrive/presenceai_p4/e3base_<id>/), so a disconnect loses nothing and Run all resumes;
the optional Colab Secret `HF_TOKEN` (read token) is used only for Hugging Face reads.

The model is the app's own: openai/whisper-small at the revision in configs/models.yaml, every
source file checked against models/MANIFEST.json, converted with the converter versions in
configs/models.yaml, and every converted file must match the app's SHA-256.

ingest reads data/colab_results/e3base_<id>.zip by default and checks every file: hypotheses
and item lists (they hold corpus text and research-only annotation times) go to the git-ignored
data/e3/<id>/; metrics, probes, pins, checks and licence texts to experiments/results/e3/base_<id>/.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import colab_data  # noqa: E402
import colab_ingest as ci  # noqa: E402

REPO = ci.REPO
TEMPLATE = HERE / "colab" / "e3_run.py"
BUILD_DIR = REPO / "data" / "e3" / "colab"
RESULTS = REPO / "experiments" / "results" / "e3"
DATA = REPO / "data" / "e3"
MODELS_YAML = REPO / "configs" / "models.yaml"
MANIFEST = REPO / "models" / "MANIFEST.json"
MODULES = {
    "t1text.py": HERE / "t1text.py",
    "ami_words.py": HERE / "ami_words.py",
    "e3score.py": HERE / "e3score.py",
    "splitzip.py": HERE / "splitzip.py",
    "fetch_public_data.py": REPO / "scripts" / "fetch_public_data.py",
    "normalize.py": REPO / "src" / "presenceai" / "text" / "normalize.py",
}
TIMEOUT_S = 10 * 3600
# The app's versions (uv.lock); CUDA 12 libraries for CTranslate2 4.9 on Colab's GPU image.
SETUP = """# The app's ASR versions (uv.lock) and the CUDA 12 libraries CTranslate2 4.9 needs.
!pip install -q faster-whisper==1.2.1 ctranslate2==4.9.0 nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
import os, importlib.util
libs = []
for pkg in ("nvidia.cublas", "nvidia.cudnn"):
    spec = importlib.util.find_spec(pkg)
    if spec and spec.submodule_search_locations:
        libs.append(os.path.join(list(spec.submodule_search_locations)[0], "lib"))
os.environ["LD_LIBRARY_PATH"] = ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])
import ctranslate2
print("CUDA devices seen by CTranslate2:", ctranslate2.get_cuda_device_count())
assert ctranslate2.get_cuda_device_count() > 0, "no GPU for CTranslate2: choose a T4 runtime"
"""


def model_config() -> dict[str, Any]:
    models = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = next(m for m in man["models"] if m["name"] == "whisper_small")
    spec = models["asr"]["whisper_small"]
    if entry["source"]["revision"] != spec["source"]["revision"]:
        raise ValueError("models/MANIFEST.json and configs/models.yaml disagree on the revision")
    return {
        "source": spec["source"],
        "quantization": spec["quantization"],
        "source_files": entry["source_files"],
        "files": entry["files"],
        "converter": models["converter"],
        "copy_files": list(models["converter"]["copy_files"]),
    }


def uv_version() -> str:
    """The laptop's uv version; the Colab converter environment is made with the same uv."""
    out = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)  # noqa: S607
    m = re.match(r"uv (\d+\.\d+\.\d+)", out.stdout.strip())
    if not m:
        raise ValueError(f"cannot read the uv version from {out.stdout!r}")
    return m.group(1)


def load_config() -> dict[str, Any]:
    cfg = colab_data.load_config()
    cfg["model"] = model_config()
    cfg["uv_version"] = uv_version()
    return cfg


def script(cfg: dict[str, Any], build_info: dict[str, Any]) -> str:
    text = TEMPLATE.read_text(encoding="utf-8")
    mods = {n: base64.b64encode(p.read_bytes()).decode("ascii") for n, p in MODULES.items()}
    for key in ("__CONFIG_JSON__", "__BUILD_JSON__", "__MODULES_JSON__"):
        if key not in text:
            raise ValueError(f"template has no {key} placeholder")
    return (
        text.replace("__CONFIG_JSON__", json.dumps(cfg, ensure_ascii=True))
        .replace("__BUILD_JSON__", json.dumps(build_info, ensure_ascii=True))
        .replace("__MODULES_JSON__", json.dumps(mods))
    )


def build(run_id: str, out: Path = BUILD_DIR) -> Path:
    body = script(load_config(), {"run_id": run_id, **ci.repo_commit()})
    nb = ci.notebook(
        title=f"PresenceAI E3 base run `{run_id}` (ADR 0028)",
        steps=(
            "1. Runtime > Change runtime type > **T4 GPU**.\n"
            "2. Secrets (key icon): `HF_TOKEN` = your Hugging Face **read** token, with *Notebook "
            "access* on (already there from the data run).\n"
            "3. Runtime > Run all; allow Google Drive when asked.\n"
            f"4. Results go to `MyDrive/presenceai_p4/e3base_{run_id}/` as each stage finishes. "
            "**If Colab disconnects, reconnect and Run all again**: finished stages and finished "
            "items are skipped (the model and data are fetched again, which takes a few "
            "minutes).\n"
            f"5. Save `e3base_{run_id}.zip` into `data/colab_results/` when the browser asks, or "
            "take it from the Drive folder.\n"
            "6. Only after the zip is on the laptop: set `DELETE_RUNTIME = True` in the last cell "
            "and run it by hand.\n\n"
            "Stage 0 is the DisfluencySpeech builder check (pass: at most 50 of 5,000 rows "
            "differ in words, ADR 0027 §6); if it fails, the run stops there."
        ),
        script_name="e3_run.py",
        body=body,
        out_dir="e3base",
        zip_stem=f"e3base_{run_id}",
        timeout_s=TIMEOUT_S,
        gpu="T4",
        drive_folder=f"presenceai_p4/e3base_{run_id}",
        out_env="P4_OUT",
        secrets=("HF_TOKEN",),
        setup=SETUP,
        cache_folder="presenceai_data",  # owner 2026-11-28, ADR 0028 §9
    )
    return ci.write_build(out, "e3_run.py", body, f"E3BASE_{run_id}.ipynb", nb)


def ingest(run_id: str, zip_path: Path | None = None) -> dict[str, Any]:
    res, data = RESULTS / f"base_{run_id}", DATA / run_id
    return ci.ingest(
        zip_path or ci.INBOX / f"e3base_{run_id}.zip",
        "e3base",
        [("hyps/*", data), ("items/*", data), ("*", res)],
        scratch=DATA / f"_ingest_{run_id}",
        info_dest=res,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["build", "ingest"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--zip", type=Path, help="default: data/colab_results/e3base_<id>.zip")
    args = ap.parse_args(argv)
    if args.action == "build":
        print(f"built {build(args.run_id)}")
    else:
        print(json.dumps(ingest(args.run_id, args.zip), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
