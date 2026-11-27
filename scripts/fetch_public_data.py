"""Fetch approved public datasets, on Google Colab only (ADR 0026 §1, DATA_SOURCES §1-4).

    python fetch_public_data.py --only ami_annotations,librispeech_test_clean --dest /content/public

- **Refuses to run outside Colab**: datasets are never downloaded to the laptop (owner,
  2026-11-27). Only sources listed under `public_data` in `configs/evaluation.yaml` exist here,
  and each was approved by the owner before it was listed.
- Every file is streamed to disk with its SHA-256; a pinned hash that does not match stops the
  run. An unpinned file reports its hash so it can be pinned (`public_data.*.sha256`).
- `licence_text` reads only the licence member of a zip, so the AMI licence is checked before any
  other AMI file is opened (DATA_SOURCES §4 rule 3).
Requests carry a generic User-Agent and nothing about the user.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

USER_AGENT = "PresenceAI-research-fetch/1.0"
CHUNK = 1 << 20
APPROVED = ("ami_annotations", "ami_headset", "librispeech_test_clean")


class NotOnColabError(RuntimeError):
    pass


def on_colab(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(env.get("COLAB_RELEASE_TAG")) and Path("/content").is_dir()


def require_colab(env: Mapping[str, str] | None = None) -> None:
    if not on_colab(env):
        raise NotOnColabError(
            "fetch_public_data runs on Google Colab only; datasets are never downloaded to the "
            "laptop (ADR 0026 §1)"
        )


def download(url: str, dest: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Stream url to dest; return {url, bytes, sha256}. A pinned hash must match, else dest is
    deleted and ValueError is raised."""
    if not url.startswith(("https://", "file://")):
        raise ValueError(f"refusing non-https url: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 (scheme checked)
    h, n = hashlib.sha256(), 0
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:  # noqa: S310
        while chunk := r.read(CHUNK):
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
    digest = h.hexdigest()
    if expected_sha256 and digest != expected_sha256:
        dest.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {url}: got {digest}, pinned {expected_sha256}")
    return {"url": url, "bytes": n, "sha256": digest, "pinned": bool(expected_sha256)}


_LICENCE_NAME = re.compile(r"^licen[cs]e(\.txt|\.md)?$", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


def licence_text(zip_path: Path) -> tuple[str, str]:
    """(member name, text) of the single licence file in the zip; reads nothing else."""
    with zipfile.ZipFile(zip_path) as z:
        names = [m for m in z.namelist() if _LICENCE_NAME.match(PurePosixPath(m).name)]
        if len(names) != 1:
            raise ValueError(f"expected one licence file in {zip_path.name}, found {names}")
        return names[0], z.read(names[0]).decode("utf-8", errors="replace")


def licence_states(text: str, must_state: str) -> bool:
    """True if the licence names CC BY 4.0 (as 'CC BY 4.0' or 'Creative Commons Attribution 4.0')
    and no NonCommercial / NoDerivatives / ShareAlike term."""
    flat = _SPACE.sub(" ", text).casefold()
    if must_state.casefold() != "cc by 4.0":
        raise ValueError(f"only 'CC BY 4.0' is supported, got {must_state!r}")
    named = "cc by 4.0" in flat or "creative commons attribution 4.0" in flat
    restricted = any(
        t in flat
        for t in (
            "noncommercial",
            "non-commercial",
            "noderivatives",
            "sharealike",
            "cc by-nc",
            "cc by-sa",
            "cc by-nd",
        )
    )
    return named and not restricted


def load_sources(config: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    return data["public_data"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", required=True, help=f"comma-separated, from {', '.join(APPROVED)}")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "evaluation.yaml",
    )
    args = ap.parse_args(argv)
    require_colab()
    sources = load_sources(args.config)
    out = []
    for name in args.only.split(","):
        if name not in APPROVED or name not in sources:
            ap.error(f"not an approved source: {name}")
        if name == "ami_headset":
            ap.error("ami_headset needs meeting and channel; the T1 data run fetches it")
        src = sources[name]
        dest = args.dest / PurePosixPath(src["url"]).name
        out.append({"source": name, **download(src["url"], dest, src.get("sha256"))})
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
