"""Download model files into models/ and record them in models/MANIFEST.json.

This is the only place in the project allowed to download models (AGENTS.md rule 2).
For now it fetches only the MediaPipe Pose Landmarker (lite, full) and Face Landmarker
bundles listed under `vision:` in configs/models.yaml.

    uv run python scripts/download_models.py --pin    # first time: record SHA-256 digests
    uv run python scripts/download_models.py          # later: verify against recorded digests

Integrity: the expected SHA-256 comes from configs/models.yaml (`digest`) or, if that is
null, from models/MANIFEST.json. If neither has one, the download stops unless --pin is
given; --pin trusts the first download and records its digest in the manifest (copy it into
configs/models.yaml to pin it for everyone). A mismatch deletes the file and fails.
Only HTTPS from storage.googleapis.com is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS_YAML = ROOT / "configs" / "models.yaml"
MODELS_DIR = ROOT / "models"
MANIFEST = MODELS_DIR / "MANIFEST.json"
DEFAULT_MODELS = ("pose_landmarker_lite", "pose_landmarker_full", "face_landmarker")
ALLOWED_HOSTS = frozenset({"storage.googleapis.com"})
MAX_BYTES = 50 * 1024 * 1024  # MediaPipe .task bundles are a few MB; refuse anything huge
CHUNK = 1 << 16

Fetch = Callable[[str, Path, int], None]


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    licence: str
    digest: str | None

    @property
    def filename(self) -> str:
        return Path(urlparse(self.url).path).name


def load_specs(path: Path = MODELS_YAML) -> dict[str, ModelSpec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs: dict[str, ModelSpec] = {}
    for name, entry in (data.get("vision") or {}).items():
        if isinstance(entry, dict) and entry.get("url"):
            specs[name] = ModelSpec(
                name, entry["url"], entry.get("licence") or "", entry.get("digest")
            )
    return specs


def read_manifest(path: Path = MANIFEST) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    return {m["name"]: m for m in data.get("models", []) if "name" in m}


def write_manifest(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    models = [entries[k] for k in sorted(entries)]
    path.write_text(json.dumps({"models": models}, indent=2) + "\n", encoding="utf-8")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _check_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise DownloadError(f"URL not allowed (HTTPS from {sorted(ALLOWED_HOSTS)} only): {url}")


def http_fetch(url: str, dest: Path, max_bytes: int) -> None:
    """Stream `url` into `dest`, refusing bodies larger than `max_bytes`."""
    _check_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "presenceai-download-models"})  # noqa: S310
    written = 0
    with urllib.request.urlopen(request, timeout=60) as resp, dest.open("wb") as out:  # noqa: S310
        while chunk := resp.read(CHUNK):
            written += len(chunk)
            if written > max_bytes:
                raise DownloadError(f"{url} is larger than {max_bytes} bytes")
            out.write(chunk)


def fetch_model(
    spec: ModelSpec,
    dest_dir: Path,
    *,
    manifest: dict[str, dict[str, Any]],
    pin: bool,
    fetch: Fetch = http_fetch,
) -> dict[str, Any]:
    """Download (if needed) and verify one model; return its manifest entry."""
    _check_url(spec.url)
    expected = spec.digest or manifest.get(spec.name, {}).get("sha256")
    if expected is None and not pin:
        raise DownloadError(
            f"{spec.name}: no SHA-256 in configs/models.yaml or MANIFEST.json; "
            "re-run with --pin to trust and record the first download"
        )
    target = dest_dir / spec.filename
    if expected is not None and target.exists() and sha256_of(target) == expected:
        print(f"OK    {spec.name}: already present, SHA-256 verified")
    else:
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            fetch(spec.url, tmp, MAX_BYTES)
            actual = sha256_of(tmp)
            if expected is not None and actual != expected:
                raise DownloadError(
                    f"{spec.name}: SHA-256 mismatch (expected {expected}, got {actual}); "
                    "file deleted"
                )
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"OK    {spec.name}: downloaded {target.stat().st_size} bytes")
    digest = sha256_of(target)
    return {
        "name": spec.name,
        "file": f"models/{spec.filename}",
        "url": spec.url,
        "licence": spec.licence,
        "sha256": digest,
        "bytes": target.stat().st_size,
        "pinned_on_first_download": expected is None,
        "retrieved": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--pin", action="store_true", help="trust and record digests that are missing"
    )
    parser.add_argument("--only", nargs="+", default=list(DEFAULT_MODELS), help="model names")
    parser.add_argument("--list", action="store_true", help="show what would be fetched and exit")
    args = parser.parse_args(argv)

    specs = load_specs()
    unknown = [n for n in args.only if n not in specs]
    if unknown:
        print(f"unknown model(s): {unknown}; known: {sorted(specs)}")
        return 2
    if args.list:
        for name in args.only:
            s = specs[name]
            print(f"{name}: {s.url} (digest: {s.digest or 'not pinned'})")
        return 0

    MODELS_DIR.mkdir(exist_ok=True)
    manifest = read_manifest()
    failures = 0
    for name in args.only:
        try:
            manifest[name] = fetch_model(specs[name], MODELS_DIR, manifest=manifest, pin=args.pin)
        except (DownloadError, OSError) as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    write_manifest(MANIFEST, manifest)
    print(f"manifest: {MANIFEST.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
