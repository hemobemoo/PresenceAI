"""C1 on Google Colab: build the notebook and ingest its results (ADR 0014).

    uv run python experiments/c1_kokoro/colab.py build  --run-id <id> --mode timing|full
    uv run python experiments/c1_kokoro/colab.py ingest --run-id <id> --zip <c1_<id>.zip>

build writes data/c1/colab/generate_c1.py (config, build info and the PresenceAI C1 code
injected) and data/c1/colab/C1_<id>.ipynb. The notebook checks for a GPU, writes the script,
checks its SHA-256, runs it, zips /content/c1 and downloads the zip through the browser. It
mounts nothing and reads no secret.

ingest unpacks the zip into data/c1/<id>/raw, checks every file against run_info.json's SHA-256
table, and only if all match: moves the clips to data/c1/<id>/clips (git-ignored), the truth
files to experiments/results/c1/<id>/truth/, run_info.json to experiments/results/c1/, writes a
clip manifest (SHA-256 per clip), and copies lexicon_check.json to resources/c1/.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from presenceai.evalconfig import REPO_ROOT, load_evaluation

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "colab" / "generate_c1.py"
DATA = REPO_ROOT / "data" / "c1"
BUILD_DIR = DATA / "colab"
RESULTS = REPO_ROOT / "experiments" / "results" / "c1"
LEXICON = REPO_ROOT / "resources" / "c1" / "lexicon_check.json"
VENV = "/content/venv312"
# kokoro 0.9.4 and misaki 0.9.4 require Python < 3.13 (PyPI metadata); Colab's kernel is 3.13.
# The generator runs in its own venv on the image's Ubuntu Python 3.12, so nothing extra is
# downloaded. Ubuntu's Python has no ensurepip, so the venv gets a copy of the runtime's own pip.
VENV_CELL = '''# The generator's own Python 3.12 environment (kokoro 0.9.4 needs Python < 3.13).
import os, pip, subprocess
cmd = """
set -e
rm -rf VENV
/usr/bin/python3.12 -m venv --without-pip VENV
SP=$(VENV/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
cp -r PIPDIR "$SP/pip"
VENV/bin/python -V
VENV/bin/python -m pip --version
""".replace("PIPDIR", os.path.dirname(pip.__file__))
print(subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout)
'''
TIMEOUT_S = {"timing": 3600, "full": 4 * 3600}
PACKAGE_FILES = (
    "src/presenceai/__init__.py",
    "src/presenceai/evalconfig.py",
    "src/presenceai/schemas/__init__.py",
    "src/presenceai/schemas/assessment.py",
    "src/presenceai/schemas/speech.py",
    "src/presenceai/schemas/c1.py",
    "src/presenceai/synth/__init__.py",
    "src/presenceai/synth/c1.py",
    "src/presenceai/synth/scripts.py",
    "src/presenceai/synth/split.py",
    "configs/evaluation.yaml",
    "resources/c1/scripts.yaml",
)


def repo_commit() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603 (fixed git command)
            ["git", *args],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        return {"repo_commit": git("rev-parse", "HEAD"), "repo_dirty": bool(git("status", "-s"))}
    except (OSError, subprocess.CalledProcessError):
        return {"repo_commit": None, "repo_dirty": None}


def package_b64() -> str:
    buf = io.BytesIO()
    # stored, not deflated: compressed bytes differ between zlib versions (1.3 on Colab, 1.3.2
    # locally), and the notebook checks the script's SHA-256 across machines
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for rel in PACKAGE_FILES:
            data = (REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
            info = zipfile.ZipInfo(rel, date_time=(2026, 1, 1, 0, 0, 0))  # stable bytes
            info.create_system = 3  # Unix on every machine (Windows would write 0)
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def script(build_info: dict[str, Any]) -> str:
    c1 = load_evaluation().c1
    cfg = {"generator": c1.generator, "voices": c1.voices}
    text = TEMPLATE.read_text(encoding="utf-8")
    for key in ("__CONFIG_JSON__", "__BUILD_JSON__", "__PACKAGE_B64__"):
        if key not in text:
            raise ValueError(f"template has no {key} placeholder")
    return (
        text.replace("__CONFIG_JSON__", json.dumps(cfg, ensure_ascii=True))
        .replace("__BUILD_JSON__", json.dumps(build_info, ensure_ascii=True))
        .replace("__PACKAGE_B64__", package_b64())
    )


def _code(src: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def notebook(run_id: str, mode: str, body: str) -> dict[str, Any]:
    intro = (
        f"# PresenceAI C1 generation, run `{run_id}`, mode `{mode}` (ADR 0014)\n\n"
        "Manual steps: Runtime > Change runtime type > **T4 GPU**; Runtime > Run all; save "
        f"`c1_{run_id}.zip` when the browser asks; then run the last cell to delete the "
        "runtime.\nNothing is mounted and no secret is used. Kokoro and its dependencies are "
        "installed only in this runtime."
    )
    gpu_check = (
        "import subprocess\n"
        "gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],\n"
        "                     capture_output=True, text=True).stdout.strip()\n"
        "print('GPU:', gpu or 'none')\n"
        "assert gpu, 'Choose Runtime > Change runtime type > T4 GPU'\n"
    )
    venv = VENV_CELL.replace("VENV", VENV)
    integrity = (
        "# The script in Colab must be byte-identical to the build (SHA-256, LF line endings).\n"
        "import hashlib\n"
        "h = hashlib.sha256(open('/content/generate_c1.py', 'rb').read()).hexdigest()\n"
        "print(h)\n"
        f"assert h == {hashlib.sha256(body.encode('utf-8')).hexdigest()!r}, 'script differs'\n"
    )
    run = f"!C1_MODE={mode} timeout {TIMEOUT_S[mode]} {VENV}/bin/python /content/generate_c1.py\n"
    download = (
        "import os, shutil\n"
        "from google.colab import files\n"
        "assert os.path.exists('/content/c1/run_info.json'), 'the run wrote no run_info.json'\n"
        f"zip_path = shutil.make_archive('/content/c1_{run_id}', 'zip', '/content', 'c1')\n"
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
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)},
            _code(gpu_check),
            _code(venv),
            _code("%%writefile /content/generate_c1.py\n" + body),
            _code(integrity),
            _code(run),
            _code(download),
            _code(end),
        ],
    }


def build(run_id: str, mode: str, out: Path = BUILD_DIR) -> Path:
    if mode not in TIMEOUT_S:
        raise ValueError(f"mode must be one of {sorted(TIMEOUT_S)}")
    out.mkdir(parents=True, exist_ok=True)
    body = script({"run_id": run_id, "mode": mode, **repo_commit()})
    (out / TEMPLATE.name).write_text(body, encoding="utf-8", newline="\n")
    nb_path = out / f"C1_{run_id}.ipynb"
    nb_path.write_text(
        json.dumps(notebook(run_id, mode, body), indent=1), encoding="utf-8", newline="\n"
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


def ingest(run_id: str, zip_path: Path, data: Path = DATA, results: Path = RESULTS,
           lexicon: Path = LEXICON) -> dict[str, Any]:  # fmt: skip
    raw = data / run_id / "raw"
    if raw.exists():
        raise FileExistsError(f"{raw} exists; remove it or use a new run id")
    safe_extract(zip_path, raw)
    src = raw / "c1"
    info = verify(src)
    results.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "run_info.json", results / f"{run_id}_run_info.json")
    if (src / "lexicon_check.json").exists():
        lexicon.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "lexicon_check.json", lexicon)
    rows = []
    if (src / "truth").is_dir():
        dest = results / run_id / "truth"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src / "truth"), str(dest))
    if (src / "clips").is_dir():
        clips = data / run_id / "clips"
        shutil.move(str(src / "clips"), str(clips))
        for f in sorted(clips.iterdir()):
            rows.append({"clip": f.name, "bytes": f.stat().st_size, "sha256": sha256(f)})
    if rows:
        with (results / f"{run_id}_clips_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    shutil.rmtree(raw)
    return {
        "run_id": run_id,
        "status": info.get("status"),
        "error": info.get("error"),
        "files_verified": len(info["sha256"]),
        "clips": len(rows),
    }


def generation_report(run_info: dict[str, Any]) -> str:
    """Clips kept, re-rendered and dropped, by voice and by rate band. A kept clip's band is its
    measured wpm; a dropped clip has none, so it is counted in its target's band."""
    from collections import Counter

    from presenceai.synth.c1 import rate_band

    cfg = load_evaluation().c1
    gen = run_info.get("generation") or {}
    kept: Counter[tuple[str, str]] = Counter()
    rerendered: Counter[tuple[str, str]] = Counter()
    extra: Counter[tuple[str, str]] = Counter()
    dropped: Counter[tuple[str, str]] = Counter()
    for c in gen.get("clips", []):
        band = rate_band(c["wpm"], cfg.rate_bands) or "outside"
        for key in (("voice", c["voice"]), ("band", band)):
            kept[key] += 1
            rerendered[key] += c["attempts"] > 1
            extra[key] += c["attempts"] - 1
    voices = list(cfg.voices)
    for d in gen.get("dropped", []):
        script, render = d["clip_id"].split("_r")
        voice = voices[(int(script[1:]) - 1) // cfg.scripts_per_voice]
        target = cfg.target_wpm[int(render)]
        for key in (("voice", voice), ("band", rate_band(target, cfg.rate_bands) or "outside")):
            dropped[key] += 1
    lines = ["kind   value        kept  re-rendered  extra renders  dropped"]
    for kind, value in sorted(set(kept) | set(dropped)):
        k = (kind, value)
        lines.append(
            f"{kind:<6} {value:<11} {kept[k]:>5}  {rerendered[k]:>11}  {extra[k]:>13}"
            f"  {dropped[k]:>7}"
        )
    lines.append(f"dropped clips: {len(gen.get('dropped', []))}")
    lines += [f"  {d['clip_id']}: {d['reason']}" for d in gen.get("dropped", [])]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["build", "ingest", "report"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--mode", choices=sorted(TIMEOUT_S), default="timing")
    ap.add_argument("--zip", type=Path, help="c1_<id>.zip downloaded from Colab (ingest)")
    args = ap.parse_args(argv)
    if args.action == "build":
        print(f"built {build(args.run_id, args.mode)}")
    elif args.action == "report":
        info = json.loads((RESULTS / f"{args.run_id}_run_info.json").read_text(encoding="utf-8"))
        print(generation_report(info))
    else:
        if not args.zip:
            ap.error("ingest needs --zip")
        print(json.dumps(ingest(args.run_id, args.zip), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
