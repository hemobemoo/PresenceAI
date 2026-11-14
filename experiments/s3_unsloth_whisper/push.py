"""Spike S3: build, push, watch and fetch the private Kaggle job (ADR 0005).

    uv run python experiments/s3_unsloth_whisper/push.py build  --owner <kaggle-user>
    uv run python experiments/s3_unsloth_whisper/push.py push   --owner <kaggle-user>
    uv run python experiments/s3_unsloth_whisper/push.py status --owner <kaggle-user>
    uv run python experiments/s3_unsloth_whisper/push.py fetch  --owner <kaggle-user> --run-id <id>

The Kaggle CLI reads its own token from ~/.kaggle; this script never opens it. The Kaggle user
name is passed on the command line and is not written to the repository. The job is a private
script kernel with a hard time limit (`kaggle.timeout_s`); it ends, and its GPU session stops,
when the script finishes.

fetch puts the CTranslate2 models in models/s3/<run>/ and the 10 test clips in data/s3/<run>/
(both git-ignored), and copies train_log.csv and run_info.json to
experiments/results/s3/<run>_train_log.csv and <run>_run_info.json, plus a SHA-256 manifest of
the model files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

TEMPLATE = common.HERE / "kaggle" / "train_kernel.py"
BUILD = common.DATA_DIR / "kernel"
# Fixed command; arguments come from the config and the CLI, never from downloaded content.
KAGGLE = ["uv", "run", "--no-project", "--with", "kaggle==2.2.5", "--with", "six", "kaggle"]


def metadata(cfg: dict[str, Any], owner: str) -> dict[str, Any]:
    k = cfg["kaggle"]
    return {
        "id": f"{owner}/{k['slug']}",
        "title": k["title"],
        "code_file": TEMPLATE.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def build(cfg: dict[str, Any], owner: str, out: Path = BUILD) -> Path:
    """Write kernel-metadata.json and the script with the config injected."""
    if "__CONFIG_JSON__" not in TEMPLATE.read_text(encoding="utf-8"):
        raise ValueError("template has no __CONFIG_JSON__ placeholder")
    out.mkdir(parents=True, exist_ok=True)
    script = TEMPLATE.read_text(encoding="utf-8").replace(
        "__CONFIG_JSON__", json.dumps(cfg, ensure_ascii=True)
    )
    (out / TEMPLATE.name).write_text(script, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(
        json.dumps(metadata(cfg, owner), indent=2), encoding="utf-8"
    )
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(cfg: dict[str, Any], owner: str, run_id: str) -> None:
    raw = common.DATA_DIR / run_id / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 (fixed Kaggle CLI command)
        [*KAGGLE, "kernels", "output", f"{owner}/{cfg['kaggle']['slug']}", "-p", str(raw)],
        check=True,
    )
    src = raw / "s3" if (raw / "s3").is_dir() else raw
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    for name in ("train_log.csv", "run_info.json"):
        if (src / name).exists():
            shutil.copy2(src / name, common.RESULTS / f"{run_id}_{name}")
    models = common.MODELS_DIR / run_id
    rows = []
    for which in ("base", "ft"):
        d = src / f"ct2_{which}"
        if d.is_dir():
            dest = models / which
            shutil.copytree(d, dest, dirs_exist_ok=True)
            for f in sorted(dest.iterdir()):
                rows.append(
                    {"model": which, "file": f.name, "bytes": f.stat().st_size, "sha256": sha256(f)}
                )
    if rows:
        with (common.RESULTS / f"{run_id}_models_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if (src / "clips").is_dir():
        shutil.copytree(src / "clips", common.DATA_DIR / run_id / "clips", dirs_exist_ok=True)
    print(f"fetched run {run_id}: models in {models}, results in {common.RESULTS}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["build", "push", "status", "fetch"])
    ap.add_argument("--owner", required=True, help="Kaggle user name (not stored in the repo)")
    ap.add_argument("--run-id")
    ap.add_argument("--config", type=Path, default=common.CONFIG)
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    ref = f"{args.owner}/{cfg['kaggle']['slug']}"
    if args.action == "build":
        print(f"built {build(cfg, args.owner)}")
    elif args.action == "push":
        out = build(cfg, args.owner)
        k = cfg["kaggle"]
        subprocess.run(  # noqa: S603 (fixed Kaggle CLI command)
            [
                *KAGGLE,
                "kernels",
                "push",
                "-p",
                str(out),
                "--accelerator",
                k["accelerator"],
                "-t",
                str(k["timeout_s"]),
            ],
            check=True,
        )
    elif args.action == "status":
        subprocess.run([*KAGGLE, "kernels", "status", ref], check=True)  # noqa: S603
    else:
        if not args.run_id:
            ap.error("fetch needs --run-id")
        fetch(cfg, args.owner, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
