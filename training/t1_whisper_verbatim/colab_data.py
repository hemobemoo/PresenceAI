"""P4 data run (PR-2): build the Colab notebook and ingest its zip (ADR 0026 §3, ADR 0027).

    uv run python training/t1_whisper_verbatim/colab_data.py build  --run-id <id>
    uv run python training/t1_whisper_verbatim/colab_data.py ingest --run-id <id> [--zip <path>]

build writes data/t1/colab/data_run.py (config, build info and the modules it needs injected:
t1text, ami_words, prepare_data, fetch_public_data and the app's text/normalize.py) and
data/t1/colab/P4DATA_<id>.ipynb. The notebook needs **no GPU** (a CPU runtime is enough). It
mounts the owner's Google Drive for **results only** (MyDrive/presenceai_p4/p4data_<id>/; never
datasets, ADR 0026 D2), so a disconnect loses nothing and Run all resumes; and it passes the
optional Colab Secret `HF_TOKEN` (the read token) to the script for the Hugging Face download,
never showing it (owner, 2026-11-27).

ingest reads data/colab_results/p4data_<id>.zip by default, checks every file against
run_info.json, then copies: t1/manifest.jsonl -> data/t1/<id>/ (git-ignored), everything else ->
experiments/results/t1/data_<id>/; then deletes the zip and its extracted copy.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import colab_ingest as ci  # noqa: E402

REPO = ci.REPO
TEMPLATE = HERE / "colab" / "data_run.py"
BUILD_DIR = REPO / "data" / "t1" / "colab"
RESULTS = REPO / "experiments" / "results" / "t1"
DATA = REPO / "data" / "t1"
EVAL_CONFIG = REPO / "configs" / "evaluation.yaml"
FILLERS = REPO / "src" / "presenceai" / "lang" / "en" / "fillers.yaml"
MODULES = {
    "t1text.py": HERE / "t1text.py",
    "ami_words.py": HERE / "ami_words.py",
    "prepare_data.py": HERE / "prepare_data.py",
    "fetch_public_data.py": REPO / "scripts" / "fetch_public_data.py",
    "normalize.py": REPO / "src" / "presenceai" / "text" / "normalize.py",
}
TIMEOUT_S = 3 * 3600


def load_config() -> dict[str, Any]:
    ev = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
    fil = yaml.safe_load(FILLERS.read_text(encoding="utf-8"))
    for src in ev["public_data"].values():  # per-file pins live in configs/pins/ (ADR 0027 §6)
        if "sha256_file" in src:
            pins = json.loads((REPO / src.pop("sha256_file")).read_text(encoding="utf-8"))
            src["sha256"] = dict(pins["sha256"])
    return {
        "t1": ev["t1"],
        "e3": ev["e3"],
        "public_data": ev["public_data"],
        "hesitations": list(fil["hesitations"]),
    }


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
        title=f"PresenceAI P4 data run `{run_id}` (ADR 0026, ADR 0027)",
        steps=(
            "1. **Once:** key icon (Secrets) on the left > Add new secret > name `HF_TOKEN`, value "
            "= your Hugging Face **read** token; turn on *Notebook access*. Optional: without it "
            "the public dataset is read anonymously.\n"
            "2. A **CPU runtime is enough** (no GPU). Runtime > Run all; allow Google Drive when "
            "asked.\n"
            "3. Results (not datasets) are written to "
            f"`MyDrive/presenceai_p4/p4data_{run_id}/` as each stage finishes. **If Colab "
            "disconnects, reconnect and Run all again: finished stages are skipped.**\n"
            f"4. Save `p4data_{run_id}.zip` into `data/colab_results/` when the browser asks. If "
            "the download fails, take the same zip from that Drive folder.\n"
            "5. Only after the zip is on the laptop: set `DELETE_RUNTIME = True` in the last cell "
            "and run it by hand.\n\n"
            "Downloads only the approved public sources (AMI official files, DisfluencySpeech at "
            "its pinned revision, LibriSpeech test-clean), reads the AMI licence first, and "
            "deletes the audio after hashing it."
        ),
        script_name="data_run.py",
        body=body,
        out_dir="p4data",
        zip_stem=f"p4data_{run_id}",
        timeout_s=TIMEOUT_S,
        gpu=None,
        drive_folder=f"presenceai_p4/p4data_{run_id}",
        out_env="P4_OUT",
        secrets=("HF_TOKEN",),
    )
    return ci.write_build(out, "data_run.py", body, f"P4DATA_{run_id}.ipynb", nb)


def ingest(run_id: str, zip_path: Path | None = None) -> dict[str, Any]:
    """Default zip: data/colab_results/p4data_<id>.zip. The manifest is git-ignored data; every
    other file (summaries, probe, speaker check, licence, pins, run_info) is a committed result."""
    return ci.ingest(
        zip_path or ci.INBOX / f"p4data_{run_id}.zip",
        "p4data",
        [("t1/manifest.jsonl", DATA / run_id), ("*", RESULTS / f"data_{run_id}")],
        scratch=DATA / f"_ingest_{run_id}",
        info_dest=RESULTS / f"data_{run_id}",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["build", "ingest"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--zip", type=Path, help="default: data/colab_results/p4data_<id>.zip")
    args = ap.parse_args(argv)
    if args.action == "build":
        print(f"built {build(args.run_id)}")
    else:
        print(json.dumps(ingest(args.run_id, args.zip), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
