"""Spike S3 on Google Colab: build the notebook and ingest its results (ADR 0005).

    uv run python experiments/s3_unsloth_whisper/colab.py build  --run-id <id>
    uv run python experiments/s3_unsloth_whisper/colab.py ingest --run-id <id> --zip <s3_<id>.zip>

build writes data/s3/colab/train_s3.py (config and build info injected) and
data/s3/colab/S3_<id>.ipynb. The notebook checks for a T4, writes the script to /content, runs it
with the time limit from the config, zips /content/s3 to /content/s3_<id>.zip and downloads it
through the browser. It mounts nothing and reads no secret.

ingest unpacks the zip into data/s3/<id>/raw, checks every file against the SHA-256 table in
run_info.json, and only if all match moves the CTranslate2 models to models/s3/<id>/{base,ft},
the clips to data/s3/<id>/clips (both git-ignored), and train_log.csv and run_info.json to
experiments/results/s3/<id>_*, with a SHA-256 manifest of the model files.

Spike S3 no longer uses Kaggle (runs 1-2, 2026-11-14, see ADR 0005); this file has no Kaggle code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

TEMPLATE = common.HERE / "colab" / "train_s3.py"
BUILD_DIR = common.DATA_DIR / "colab"


def repo_commit() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603 (fixed git command)
            ["git", *args],  # noqa: S607
            cwd=common.REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        return {"repo_commit": git("rev-parse", "HEAD"), "repo_dirty": bool(git("status", "-s"))}
    except (OSError, subprocess.CalledProcessError):
        return {"repo_commit": None, "repo_dirty": None}


def script(cfg: dict[str, Any], build_info: dict[str, Any]) -> str:
    text = TEMPLATE.read_text(encoding="utf-8")
    for key in ("__CONFIG_JSON__", "__BUILD_JSON__"):
        if key not in text:
            raise ValueError(f"template has no {key} placeholder")
    return text.replace("__CONFIG_JSON__", json.dumps(cfg, ensure_ascii=True)).replace(
        "__BUILD_JSON__", json.dumps(build_info, ensure_ascii=True)
    )


def _code(src: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def notebook(cfg: dict[str, Any], run_id: str, body: str) -> dict[str, Any]:
    rt = cfg["runtime"]
    intro = (
        f"# PresenceAI spike S3, run `{run_id}` (ADR 0005)\n\n"
        "Manual steps: Runtime > Change runtime type > **T4 GPU**; Runtime > Run all; save "
        f"`s3_{run_id}.zip` when the browser asks; then run the last cell to delete the runtime.\n"
        "Nothing is mounted and no secret is used."
    )
    gpu_check = (
        "import subprocess\n"
        "gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],\n"
        "                     capture_output=True, text=True).stdout.strip()\n"
        "print('GPU:', gpu or 'none')\n"
        f"assert {rt['gpu']!r} in gpu, 'Choose Runtime > Change runtime type > T4 GPU'\n"
    )
    integrity = (
        "# The script in Colab must be byte-identical to the build (SHA-256, LF line endings).\n"
        "import hashlib\n"
        "h = hashlib.sha256(open('/content/train_s3.py', 'rb').read()).hexdigest()\n"
        "print(h)\n"
        f"assert h == {hashlib.sha256(body.encode('utf-8')).hexdigest()!r}, 'script differs'\n"
    )
    run = f"!timeout {int(rt['timeout_s'])} python /content/train_s3.py\n"
    download = (
        "import os, shutil\n"
        "from google.colab import files\n"
        "assert os.path.exists('/content/s3/run_info.json'), 'the run wrote no run_info.json'\n"
        f"zip_path = shutil.make_archive('/content/s3_{run_id}', 'zip', '/content', 's3')\n"
        "print(zip_path, os.path.getsize(zip_path), 'bytes')\n"
        "files.download(zip_path)\n"
    )
    end = (
        "# Run only after the zip is saved on the laptop: ends and deletes this runtime.\n"
        "from google.colab import runtime\n"
        "runtime.unassign()\n"
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": rt["gpu"], "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)},
            _code(gpu_check),
            _code("%%writefile /content/train_s3.py\n" + body),
            _code(integrity),
            _code(run),
            _code(download),
            _code(end),
        ],
    }


def build(cfg: dict[str, Any], run_id: str, out: Path = BUILD_DIR) -> Path:
    """Write the injected script and the notebook; return the notebook path."""
    out.mkdir(parents=True, exist_ok=True)
    body = script(cfg, {"run_id": run_id, **repo_commit()})
    # LF on every platform, so the SHA-256 checked in Colab matches this file.
    (out / TEMPLATE.name).write_text(body, encoding="utf-8", newline="\n")
    nb_path = out / f"S3_{run_id}.ipynb"
    nb_path.write_text(
        json.dumps(notebook(cfg, run_id, body), indent=1), encoding="utf-8", newline="\n"
    )
    return nb_path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract only relative member paths that stay inside dest."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            p = PurePosixPath(m.filename)
            if p.is_absolute() or ".." in p.parts or ":" in m.filename or "\\" in m.filename:
                raise ValueError(f"unsafe path in zip: {m.filename!r}")
        z.extractall(dest)


def verify(src: Path) -> dict[str, Any]:
    """Check every file under src against run_info.json's SHA-256 table; return run_info."""
    info = json.loads((src / "run_info.json").read_text(encoding="utf-8"))
    table = info.get("sha256")
    if not isinstance(table, dict) or not table:
        raise ValueError("run_info.json has no sha256 table")
    found = {
        p.relative_to(src).as_posix()
        for p in src.rglob("*")
        if p.is_file() and p.name != "run_info.json"
    }
    problems = [f"missing: {r}" for r in sorted(set(table) - found)]
    problems += [f"not listed: {r}" for r in sorted(found - set(table))]
    for rel in sorted(set(table) & found):
        f = src / rel
        if f.stat().st_size != table[rel]["bytes"] or sha256(f) != table[rel]["sha256"]:
            problems.append(f"checksum mismatch: {rel}")
    if problems:
        raise ValueError("checksum check failed:\n" + "\n".join(problems))
    return info


def ingest(run_id: str, zip_path: Path) -> dict[str, Any]:
    raw = common.DATA_DIR / run_id / "raw"
    if raw.exists():
        raise FileExistsError(f"{raw} exists; remove it or use a new run id")
    safe_extract(zip_path, raw)
    src = raw / "s3"
    info = verify(src)
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    for name in ("train_log.csv", "run_info.json"):
        if (src / name).exists():
            shutil.copy2(src / name, common.RESULTS / f"{run_id}_{name}")
    rows = []
    for which in ("base", "ft"):
        d = src / f"ct2_{which}"
        if d.is_dir():
            dest = common.MODELS_DIR / run_id / which
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(dest))
            for f in sorted(dest.iterdir()):
                rows.append(
                    {"model": which, "file": f.name, "bytes": f.stat().st_size, "sha256": sha256(f)}
                )
    if rows:
        with (common.RESULTS / f"{run_id}_models_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if (src / "clips").is_dir():
        shutil.move(str(src / "clips"), str(common.DATA_DIR / run_id / "clips"))
    shutil.rmtree(raw)
    return {
        "run_id": run_id,
        "status": info.get("status"),
        "files_verified": len(info["sha256"]),
        "model_files": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["build", "ingest"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--zip", type=Path, help="s3_<id>.zip downloaded from Colab (ingest)")
    ap.add_argument("--config", type=Path, default=common.CONFIG)
    args = ap.parse_args(argv)
    if args.action == "build":
        print(f"built {build(common.load_config(args.config), args.run_id)}")
    else:
        if not args.zip:
            ap.error("ingest needs --zip")
        print(json.dumps(ingest(args.run_id, args.zip), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
