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
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
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


RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_WAITS_S = (15, 30, 60, 120, 240, 480)  # run d2: the AMI server answered 503 after 50 files


def _wait_s(err: Exception, attempt: int) -> float:
    after = getattr(err, "headers", None) and err.headers.get("Retry-After")  # type: ignore[union-attr]
    if after and str(after).isdigit():
        return min(float(after), 900.0)
    return float(RETRY_WAITS_S[attempt])


def _fetch_once(url: str, dest: Path) -> tuple[str, int]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 (scheme checked)
    h, n = hashlib.sha256(), 0
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:  # noqa: S310
        while chunk := r.read(CHUNK):
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def download(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Stream url to dest; return {url, bytes, sha256, retries}. A busy server (429/5xx), a reset
    or a timeout is retried after 15 s, 30 s, ... 480 s (or the server's Retry-After); any other
    HTTP error stops at once. A pinned hash must match, else dest is deleted and ValueError is
    raised."""
    if not url.startswith(("https://", "file://")):
        raise ValueError(f"refusing non-https url: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(len(RETRY_WAITS_S) + 1):
        try:
            digest, n = _fetch_once(url, dest)
            break
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == len(RETRY_WAITS_S):
                raise
            err: Exception = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == len(RETRY_WAITS_S):
                raise
            err = e
        wait = _wait_s(err, attempt)
        print(
            f"retry {attempt + 1} for {PurePosixPath(url).name} in {wait:.0f} s: {err}", flush=True
        )
        sleep(wait)
    else:  # pragma: no cover (the loop always breaks or raises)
        raise RuntimeError("unreachable")
    retries = attempt
    if expected_sha256 and digest != expected_sha256:
        dest.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {url}: got {digest}, pinned {expected_sha256}")
    return {
        "url": url,
        "bytes": n,
        "sha256": digest,
        "pinned": bool(expected_sha256),
        "retries": retries,
    }


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
