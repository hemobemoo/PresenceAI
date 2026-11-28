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
import shutil
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


def on_kaggle(env: Mapping[str, str] | None = None) -> bool:
    """Kaggle Notebooks, the fallback platform (ADR 0026 §1; used for T1-r1, ADR 0029 §11)."""
    env = os.environ if env is None else env
    return bool(env.get("KAGGLE_KERNEL_RUN_TYPE")) and Path("/kaggle/working").is_dir()


def require_cloud(env: Mapping[str, str] | None = None) -> None:
    """Colab or Kaggle; never the laptop (ADR 0026 §1)."""
    if not (on_colab(env) or on_kaggle(env)):
        raise NotOnColabError(
            "this runs on Google Colab or Kaggle only; datasets are never downloaded to the "
            "laptop (ADR 0026 §1)"
        )


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


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


# ---- Drive cache (owner, 2026-11-28; ADR 0028 §9, amends ADR 0026 D2) ----------------------------
# Only PINNED public files are cached, in the owner's private folder on Google Drive. A cached copy
# is used only if its SHA-256 matches the pin; a copy that does not match is deleted and the file is
# downloaded again. The owner's own recordings never come here (they never leave the laptop).


def cache_get(cache_dir: Path | None, name: str, expected_sha256: str | None, dest: Path) -> bool:
    """Copy a verified cached file to dest; False if there is none (or it did not verify)."""
    if cache_dir is None or not expected_sha256:
        return False
    cached = cache_dir / name
    if not cached.is_file():
        return False
    if sha256_of(cached) != expected_sha256:
        print(f"cache: {name} does not match its pin; deleted, downloading again", flush=True)
        cached.unlink()
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    if sha256_of(dest) != expected_sha256:  # the copy itself must verify too
        dest.unlink(missing_ok=True)
        return False
    return True


def cache_put(
    cache_dir: Path | None, name: str, expected_sha256: str | None, src: Path, source: str
) -> bool:
    """Store a verified file in the cache and its manifest; only pinned files are stored."""
    if cache_dir is None or not expected_sha256 or sha256_of(src) != expected_sha256:
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / (name + ".part")
    shutil.copyfile(src, tmp)
    tmp.replace(cache_dir / name)
    mpath = cache_dir / "cache_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    manifest[name] = {"sha256": expected_sha256, "bytes": src.stat().st_size, "source": source}
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return True


def download(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Stream url to dest; return {url, bytes, sha256, retries, from_cache}. A pinned file in
    `cache_dir` that verifies is used without any request; a downloaded pinned file is added to
    the cache. A busy server (429/5xx), a reset or a timeout is retried after 15 s, 30 s, ... 480 s
    (or the server's Retry-After); any other HTTP error stops at once. A pinned hash must match,
    else dest is deleted and ValueError is raised."""
    if not url.startswith(("https://", "file://")):
        raise ValueError(f"refusing non-https url: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if cache_get(cache_dir, dest.name, expected_sha256, dest):
        return {
            "url": url,
            "bytes": dest.stat().st_size,
            "sha256": expected_sha256,
            "pinned": True,
            "retries": 0,
            "from_cache": True,
        }
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
    cache_put(cache_dir, dest.name, expected_sha256, dest, url)  # saved as soon as it verifies
    return {
        "url": url,
        "bytes": n,
        "sha256": digest,
        "pinned": bool(expected_sha256),
        "retries": retries,
        "from_cache": False,
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
