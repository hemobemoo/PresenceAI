"""Shared Colab helpers for every P4 run (ADR 0026 §1, ADR 0005 pattern).

- `notebook(...)` builds a Colab notebook that writes one script, checks its SHA-256 against the
  build, runs it with a time limit, zips its output folder and downloads the zip. It mounts
  nothing and reads no secret unless a run says so.
- `ingest(...)` reads the zip the owner downloaded into **`data/colab_results/`** (the default
  inbox, owner 2026-11-27), checks every file against `run_info.json`'s SHA-256 table, copies the
  checked files to their places (`models/` for model files, `experiments/results/<track>/<run>/`
  for metrics and small files, `data/<track>/<run>/` for git-ignored data such as manifests), then
  deletes the zip and its extracted copy. On any mismatch nothing is copied, the extracted copy is
  deleted and the zip is kept for diagnosis.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

REPO = Path(__file__).resolve().parents[1]
INBOX = REPO / "data" / "colab_results"  # git-ignored (data/*); never a rule-9 path


def repo_commit(repo: Path = REPO) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603 (fixed git command)
            ["git", *args],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        return {"repo_commit": git("rev-parse", "HEAD"), "repo_dirty": bool(git("status", "-s"))}
    except (OSError, subprocess.CalledProcessError):
        return {"repo_commit": None, "repo_dirty": None}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code(src: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def notebook(
    *,
    title: str,
    steps: str,
    script_name: str,
    body: str,
    out_dir: str,
    zip_stem: str,
    timeout_s: int,
    gpu: str | None,
    drive_folder: str | None = None,
    out_env: str = "P4_OUT",
    secrets: Sequence[str] = (),
    setup: str | None = None,
) -> dict[str, Any]:
    """A notebook that runs `body` as /content/<script_name> and downloads <zip_stem>.zip.

    drive_folder: results (never datasets, ADR 0026 D2) are written to
    /content/drive/MyDrive/<drive_folder>/<out_dir>, passed to the script as `out_env`, so a
    disconnect loses nothing and a rerun resumes; the zip is also saved there.
    secrets: Colab Secrets names passed to the script as environment variables; their values are
    never printed or written.
    The last cell (delete the runtime) never runs under Run all: it would cut the download off.
    """
    intro = f"# {title}\n\n{steps}\n"
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)}]
    if drive_folder:
        base = f"/content/drive/MyDrive/{drive_folder}"
        res = f"{base}/{out_dir}"
        cells.append(
            _code(
                "# Results (not datasets) go to your Google Drive, so a disconnect loses nothing.\n"
                "import os\n"
                "from google.colab import drive\n"
                "drive.mount('/content/drive')\n"
                f"os.makedirs({res!r}, exist_ok=True)\n"
                f"os.environ[{out_env!r}] = {res!r}\n"
                f"print('results folder:', {res!r})\n"
            )
        )
    else:
        res = f"/content/{out_dir}"
        base = "/content"
    for name in secrets:
        cells.append(
            _code(
                f"# {name} from Colab Secrets (key icon on the left). The value is never shown.\n"
                "import os\n"
                "try:\n"
                "    from google.colab import userdata\n"
                f"    os.environ[{name!r}] = userdata.get({name!r})\n"
                f"    print({name!r}, 'found in Colab Secrets (value not shown)')\n"
                "except Exception as e:\n"
                f"    os.environ.pop({name!r}, None)\n"
                f"    print({name!r}, 'not available:', type(e).__name__, '- going on without')\n"
            )
        )
    if gpu:
        cells.append(
            _code(
                "import subprocess\n"
                "q = ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']\n"
                "gpu = subprocess.run(q, capture_output=True, text=True).stdout.strip()\n"
                "print('GPU:', gpu or 'none')\n"
                f"assert {gpu!r} in gpu, 'Choose Runtime > Change runtime type > {gpu} GPU'\n"
            )
        )
    if setup:
        cells.append(_code(setup))
    cells += [
        _code(f"%%writefile /content/{script_name}\n" + body),
        _code(
            "# The script in Colab must be byte-identical to the build (SHA-256, LF endings).\n"
            "import hashlib\n"
            f"h = hashlib.sha256(open('/content/{script_name}', 'rb').read()).hexdigest()\n"
            "print(h)\n"
            f"assert h == {sha256_bytes(body.encode('utf-8'))!r}, 'script differs from the build'\n"
        ),
        _code(f"!timeout {int(timeout_s)} python /content/{script_name}\n"),
        _code(
            "import os, shutil\n"
            "from google.colab import files\n"
            f"assert os.path.exists('{res}/run_info.json'), 'no run_info.json'\n"
            f"zip_path = shutil.make_archive('/content/{zip_stem}', 'zip',\n"
            f"                               {str(PurePosixPath(res).parent)!r}, {out_dir!r})\n"
            "print(zip_path, os.path.getsize(zip_path), 'bytes')\n"
            + (
                f"shutil.copy(zip_path, {base!r})\n"
                f"print('copy on Google Drive:', {base + '/' + zip_stem + '.zip'!r})\n"
                if drive_folder
                else ""
            )
            + "files.download(zip_path)  # save it into data/colab_results/ in the repository\n"
        ),
        _code(
            "# Run this cell BY HAND, only after the zip is saved on the laptop: it ends and\n"
            "# deletes this runtime. Under Run all it does nothing (it would cut the download).\n"
            "DELETE_RUNTIME = False  # set to True, then run this cell\n"
            "if DELETE_RUNTIME:\n"
            "    from google.colab import runtime\n"
            "    runtime.unassign()\n"
        ),
    ]
    meta: dict[str, Any] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}}
    if gpu:
        meta |= {"accelerator": "GPU", "colab": {"gpuType": gpu, "provenance": []}}
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}


def write_build(out: Path, script_name: str, body: str, nb_name: str, nb: dict[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    # LF on every platform, so the SHA-256 checked in Colab matches this file.
    (out / script_name).write_text(body, encoding="utf-8", newline="\n")
    path = out / nb_name
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
    return path


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


def ingest(
    zip_path: Path,
    out_dir: str,
    placements: Sequence[tuple[str, Path]],
    scratch: Path,
    info_dest: Path,
) -> dict[str, Any]:
    """Verify the zip and copy each file to the first placement whose glob matches its relative
    path; run_info.json goes to `info_dest`. Unmatched files are an error. Deletes the extracted
    copy always, and the zip only after success."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"{zip_path} not found; download the Colab zip into {INBOX}")
    if scratch.exists():
        raise FileExistsError(f"{scratch} exists; remove it or use a new run id")
    try:
        safe_extract(zip_path, scratch)
        src = scratch / out_dir
        info = verify(src)
        plan: list[tuple[Path, Path]] = []
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            rel = f.relative_to(src).as_posix()
            if rel == "run_info.json":
                plan.append((f, info_dest / rel))
                continue
            dest = next((d for pat, d in placements if fnmatch(rel, pat)), None)
            if dest is None:
                raise ValueError(f"no placement for {rel}")
            plan.append((f, dest / rel))
        for _f, dest in plan:
            if dest.exists():
                raise FileExistsError(f"{dest} exists; refusing to overwrite")
        for f, dest in plan:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    zip_path.unlink()
    return {
        "status": info.get("status"),
        "files_verified": len(info["sha256"]),
        "copied": [
            (d.relative_to(REPO) if d.is_relative_to(REPO) else d).as_posix() for _, d in plan
        ],
    }


def run_info_table(root: Path) -> dict[str, dict[str, Any]]:
    """SHA-256 table of every file under root except run_info.json (used inside Colab scripts)."""
    return {
        p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "run_info.json"
    }
