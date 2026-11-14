"""Spike S2: record the converted model and build the Space folder (ADR 0007).

    uv run python experiments/s2_zerogpu/prepare.py model   # SHA-256 manifest + model card
    uv run python experiments/s2_zerogpu/prepare.py space   # data/s2/space, needs repo_revision

`model` writes experiments/results/s2/model_manifest.csv (every file of both model folders) and
data/s2/model_card/README.md (card for the private model repo).

`space` writes data/s2/space/: app.py, audio.py (a copy of src/presenceai/speech/audio.py, so the
Space follows ADR 0006), requirements.txt, README.md (Space card) and s2_config.json. It refuses to
build until `model.repo_revision` is pinned in the config. Repo ids and user names are never
written: the Space reads the model repo id from its S2_MODEL_REPO variable.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

LOADER = common.REPO / "src" / "presenceai" / "speech" / "audio.py"
SPACE_SRC = common.HERE / "space" / "app.py"
TURBO_DIR = common.MODELS_DIR / "whisper-large-v3-turbo-ct2"
SMALL_DIR = common.REPO / "models" / "s3" / "20261114-c3" / "base"


def model_folders(cfg: dict[str, Any]) -> dict[str, Path]:
    m = cfg["model"]
    return {m["turbo_subfolder"]: TURBO_DIR, m["cpu_subfolder"]: SMALL_DIR}


def model_manifest(cfg: dict[str, Any], out: Path = common.RESULTS) -> Path:
    rows = []
    for sub, folder in model_folders(cfg).items():
        for f in sorted(folder.iterdir()):
            if f.is_file():
                rows.append(
                    {
                        "path": f"{sub}/{f.name}",
                        "bytes": f.stat().st_size,
                        "sha256": common.sha256(f),
                    }
                )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    return path


def model_card(cfg: dict[str, Any]) -> str:
    m = cfg["model"]
    return (
        "---\nlicense: mit\nlibrary_name: ctranslate2\ntags: [whisper, ctranslate2]\n---\n\n"
        "# PresenceAI spike S2 models (private)\n\n"
        f"- `{m['turbo_subfolder']}/`: `{m['source']}` at revision `{m['source_revision']}`, "
        f"converted with ctranslate2 {m['converted_with']['ctranslate2']} and transformers "
        f"{m['converted_with']['transformers']} (`--quantization {m['ct2_quantization']}`).\n"
        f"- `{m['cpu_subfolder']}/`: the base whisper-small CTranslate2 int8 export from spike S3 "
        "run `20261114-c3` (`unsloth/whisper-small`, a mirror of `openai/whisper-small`, MIT).\n\n"
        "SHA-256 of every file: `experiments/results/s2/model_manifest.csv` in the PresenceAI "
        "repository.\n"
    )


def space_card(cfg: dict[str, Any]) -> str:
    s = cfg["space"]
    return (
        "---\ntitle: PresenceAI S2 ZeroGPU\nsdk: gradio\n"
        f'sdk_version: {s["gradio_sdk_version"]}\npython_version: "{s["python_version"]}"\n'
        "app_file: app.py\nlicense: mit\npinned: false\n---\n\n"
        "Private spike S2 Space (ADR 0007): faster-whisper large-v3-turbo on ZeroGPU.\n"
    )


def space_config(cfg: dict[str, Any]) -> dict[str, Any]:
    m, s = cfg["model"], cfg["space"]
    if not m.get("repo_revision"):
        raise ValueError("pin model.repo_revision in the config first (after `hf upload`)")
    return {
        "model_repo_revision": m["repo_revision"],
        "turbo_subfolder": m["turbo_subfolder"],
        "cpu_subfolder": m["cpu_subfolder"],
        "gpu_compute_type": s["gpu_compute_type"],
        "cpu_compute_type": s["cpu_compute_type"],
        "gpu_duration_s": s["gpu_duration_s"],
        "transcribe": cfg["transcribe"],
    }


def build_space(cfg: dict[str, Any], out: Path = common.DATA_DIR / "space") -> Path:
    conf = space_config(cfg)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(SPACE_SRC, out / "app.py")
    shutil.copy2(LOADER, out / "audio.py")
    (out / "requirements.txt").write_text(
        "\n".join(cfg["space"]["requirements"]) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "README.md").write_text(space_card(cfg), encoding="utf-8", newline="\n")
    (out / "s2_config.json").write_text(
        json.dumps(conf, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("what", choices=["model", "space"])
    args = ap.parse_args(argv)
    cfg = common.load_config()
    if args.what == "model":
        print(f"wrote {model_manifest(cfg)}")
        card = common.DATA_DIR / "model_card" / "README.md"
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(model_card(cfg), encoding="utf-8", newline="\n")
        print(f"wrote {card}")
    else:
        print(f"built {build_space(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
