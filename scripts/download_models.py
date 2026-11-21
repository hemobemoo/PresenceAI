"""Download model files into models/ and record them in models/MANIFEST.json.

This is the only place in the project allowed to download models (AGENTS.md rule 2).

MediaPipe (default): the Pose Landmarker (lite, full) and Face Landmarker bundles listed under
`vision:` in configs/models.yaml.

    uv run python scripts/download_models.py --pin    # first time: record SHA-256 digests
    uv run python scripts/download_models.py          # later: verify against recorded digests

spaCy (only when named): the `nlp:` wheel from spaCy's GitHub release (owner approval
2026-11-21, ADR 0018). The wheel is verified like the MediaPipe bundles, then its model folder
is unpacked into models/<dir>/, which text/nlp.py loads with spacy.load(path). Nothing is
pip-installed.

    uv run python scripts/download_models.py --only spacy_en

Whisper (only when named): the `asr:` entries that have a `source` (owner approval 2026-11-17).

    uv run python scripts/download_models.py --only whisper_small whisper_turbo

Each source file is fetched from huggingface.co at the pinned commit and checked against the
hash Hugging Face lists for that commit (SHA-256 for large files, the git blob SHA-1 for small
ones). The model is then converted to CTranslate2 with `ct2-transformers-converter` in a
throw-away `uv run --no-project` environment (versions under `converter:`, as in S2), offline.
The manifest records the SHA-256 of every source and converted file. Source files are deleted
after a successful conversion unless --keep-sources is given.

Integrity: the expected SHA-256 comes from configs/models.yaml (`digest`) or, if that is
null, from models/MANIFEST.json. If neither has one, the download stops unless --pin is
given; --pin trusts the first download and records its digest in the manifest (copy it into
configs/models.yaml to pin it for everyone). A mismatch deletes the file and fails.
Only HTTPS from storage.googleapis.com (MediaPipe), github.com (spaCy; redirects only to
GitHub's release-asset hosts) and huggingface.co (Whisper) is requested.
Hugging Face serves large files from its own CDN, so a redirect from huggingface.co is followed
only to an HTTPS host under hf.co (HF_REDIRECT_SUFFIX); the hash check above applies either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS_YAML = ROOT / "configs" / "models.yaml"
MODELS_DIR = ROOT / "models"
MANIFEST = MODELS_DIR / "MANIFEST.json"
DEFAULT_MODELS = ("pose_landmarker_lite", "pose_landmarker_full", "face_landmarker")
ALLOWED_HOSTS = frozenset({"storage.googleapis.com", "github.com"})
# GitHub serves release assets from its own storage hosts after a redirect from github.com
REDIRECT_HOSTS = frozenset(
    {"objects.githubusercontent.com", "release-assets.githubusercontent.com"}
)
MAX_BYTES = 50 * 1024 * 1024  # MediaPipe .task bundles are a few MB; refuse anything huge
CHUNK = 1 << 16
HF_HOST = "huggingface.co"
HF_REDIRECT_SUFFIX = ".hf.co"  # Hugging Face's own CDN for large (LFS/Xet) files
HF_MAX_BYTES = 4 * 1024 * 1024 * 1024  # whisper-large-v3-turbo safetensors is ~1.6 GB
HEX40 = frozenset("0123456789abcdef")

Fetch = Callable[[str, Path, int], None]
GetJson = Callable[[str], Any]
Run = Callable[..., Any]


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    licence: str
    digest: str | None
    unpack: str | None = None  # wheel: path inside the wheel of the model folder to unpack
    dir: str | None = None  # wheel: folder under models/ to unpack it into

    @property
    def filename(self) -> str:
        return Path(urlparse(self.url).path).name


def load_specs(path: Path = MODELS_YAML) -> dict[str, ModelSpec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs: dict[str, ModelSpec] = {}
    for section in ("vision", "nlp"):
        for name, entry in (data.get(section) or {}).items():
            if isinstance(entry, dict) and entry.get("url"):
                specs[name] = ModelSpec(
                    name,
                    entry["url"],
                    entry.get("licence") or "",
                    entry.get("digest"),
                    unpack=entry.get("unpack"),
                    dir=entry.get("dir"),
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


def fetch_redirect_allowed(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return parsed.scheme == "https" and (host in ALLOWED_HOSTS or host in REDIRECT_HOSTS)


class _Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        target = urljoin(req.full_url, newurl)
        if not fetch_redirect_allowed(target):
            raise DownloadError(
                f"redirect to a host that is not allowed: {urlparse(target).hostname}"
            )
        return super().redirect_request(req, fp, code, msg, headers, target)


def http_fetch(url: str, dest: Path, max_bytes: int) -> None:
    """Stream `url` into `dest`, refusing bodies larger than `max_bytes`."""
    _check_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "presenceai-download-models"})  # noqa: S310
    opener = urllib.request.build_opener(_Redirects())
    written = 0
    with opener.open(request, timeout=60) as resp, dest.open("wb") as out:
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
    if spec.unpack and spec.dir:
        unpack_wheel(target, spec.unpack, dest_dir / spec.dir)
        print(f"OK    {spec.name}: unpacked into models/{spec.dir}")
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


def unpack_wheel(wheel: Path, member_dir: str, out_dir: Path) -> None:
    """Extract the files under `member_dir/` of a verified wheel into `out_dir` (no pip install).

    Only regular files under that folder are written, and every path is checked to stay inside
    `out_dir`, so a crafted archive cannot write elsewhere.
    """
    prefix = member_dir.rstrip("/") + "/"
    root = out_dir.resolve()
    with zipfile.ZipFile(wheel) as zf:
        members = [m for m in zf.infolist() if m.filename.startswith(prefix) and not m.is_dir()]
        if not members:
            raise DownloadError(f"{wheel.name}: no files under {prefix}")
        for m in members:
            target = (root / m.filename[len(prefix) :]).resolve()
            if not target.is_relative_to(root):
                raise DownloadError(f"{wheel.name}: unsafe path {m.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, target.open("wb") as dst:
                while block := src.read(CHUNK):
                    dst.write(block)


# --- Whisper from huggingface.co ------------------------------------------------------------


@dataclass(frozen=True)
class HfSpec:
    name: str
    repo: str
    revision: str
    quantization: str
    dir: str
    licence: str
    files: tuple[str, ...]
    copy_files: tuple[str, ...]
    packages: tuple[str, ...]
    python: str


def load_hf_specs(path: Path = MODELS_YAML) -> dict[str, HfSpec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    conv = data.get("converter") or {}
    specs: dict[str, HfSpec] = {}
    for name, entry in (data.get("asr") or {}).items():
        src = entry.get("source") if isinstance(entry, dict) else None
        if not src:
            continue
        specs[name] = HfSpec(
            name=name,
            repo=str(src["repo"]),
            revision=str(src["revision"]),
            quantization=str(entry["quantization"]),
            dir=str(entry["dir"]),
            licence=str(entry.get("licence") or ""),
            files=tuple(conv.get("source_files") or ()),
            copy_files=tuple(conv.get("copy_files") or ()),
            packages=tuple(conv.get("packages") or ()),
            python=str(conv.get("python") or "3.12"),
        )
    return specs


def _check_hf_spec(spec: HfSpec) -> None:
    if len(spec.revision) != 40 or not set(spec.revision) <= HEX40:
        raise DownloadError(f"{spec.name}: revision must be a full 40-hex commit id")
    if not spec.files:
        raise DownloadError(f"{spec.name}: no `converter.source_files` in configs/models.yaml")
    if Path(spec.dir).name != spec.dir or spec.dir in {"", ".", ".."}:
        raise DownloadError(f"{spec.name}: `dir` must be a plain folder name")


def _check_hf_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != HF_HOST:
        raise DownloadError(f"URL not allowed (HTTPS from {HF_HOST} only): {url}")


def redirect_allowed(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return parsed.scheme == "https" and (host == HF_HOST or host.endswith(HF_REDIRECT_SUFFIX))


class _HfRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        target = urljoin(req.full_url, newurl)
        if not redirect_allowed(target):
            raise DownloadError(
                f"redirect to a host that is not allowed: {urlparse(target).hostname}"
            )
        return super().redirect_request(req, fp, code, msg, headers, target)


def _hf_open(url: str, timeout: int) -> Any:
    _check_hf_url(url)
    opener = urllib.request.build_opener(_HfRedirects())
    request = urllib.request.Request(url, headers={"User-Agent": "presenceai-download-models"})  # noqa: S310
    return opener.open(request, timeout=timeout)


def hf_get_json(url: str) -> Any:
    with _hf_open(url, 60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def hf_fetch(url: str, dest: Path, max_bytes: int) -> None:
    written = 0
    with _hf_open(url, 120) as resp, dest.open("wb") as out:
        while chunk := resp.read(CHUNK):
            written += len(chunk)
            if written > max_bytes:
                raise DownloadError(f"{url} is larger than {max_bytes} bytes")
            out.write(chunk)


def hf_file_url(repo: str, revision: str, filename: str) -> str:
    return f"https://{HF_HOST}/{quote(repo)}/resolve/{revision}/{quote(filename)}"


def hf_expected_hashes(spec: HfSpec, get_json: GetJson = hf_get_json) -> dict[str, tuple[str, str]]:
    """{file: (algorithm, hex)} at the pinned commit: ('sha256', ...) for LFS files and
    ('git-sha1', ...) for files stored directly in git."""
    url = f"https://{HF_HOST}/api/models/{quote(spec.repo)}/tree/{spec.revision}"
    expected: dict[str, tuple[str, str]] = {}
    for item in get_json(url):
        if item.get("type") != "file":
            continue
        lfs = item.get("lfs")
        if lfs and lfs.get("oid"):
            expected[item["path"]] = ("sha256", str(lfs["oid"]))
        else:
            expected[item["path"]] = ("git-sha1", str(item["oid"]))
    missing = [f for f in spec.files if f not in expected]
    if missing:
        raise DownloadError(f"{spec.name}: not in {spec.repo}@{spec.revision[:7]}: {missing}")
    return expected


def git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    h.update(f"blob {path.stat().st_size}".encode() + b"\x00")
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _matches(path: Path, algorithm: str, expected: str) -> bool:
    actual = sha256_of(path) if algorithm == "sha256" else git_blob_sha1(path)
    return actual == expected


def fetch_hf_sources(
    spec: HfSpec,
    src_dir: Path,
    expected: dict[str, tuple[str, str]],
    *,
    fetch: Fetch = hf_fetch,
) -> list[dict[str, Any]]:
    """Download each source file (skipping verified ones); return its manifest rows."""
    src_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for filename in spec.files:
        algorithm, want = expected[filename]
        target = src_dir / filename
        if not (target.exists() and _matches(target, algorithm, want)):
            tmp = target.with_name(target.name + ".part")
            try:
                fetch(hf_file_url(spec.repo, spec.revision, filename), tmp, HF_MAX_BYTES)
                if not _matches(tmp, algorithm, want):
                    raise DownloadError(
                        f"{spec.name}/{filename}: {algorithm} does not match "
                        f"{spec.repo}@{spec.revision[:7]}; file deleted"
                    )
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)
            size = target.stat().st_size
            print(f"OK    {spec.name}/{filename}: {size} bytes, {algorithm} verified")
        rows.append({"path": filename, "bytes": target.stat().st_size, "sha256": sha256_of(target)})
    return rows


def converter_command(spec: HfSpec, src_dir: Path, out_dir: Path) -> list[str]:
    cmd = ["uv", "run", "--no-project", "--python", spec.python]
    for pkg in spec.packages:
        cmd += ["--with", pkg]
    cmd += ["ct2-transformers-converter", "--model", str(src_dir), "--output_dir", str(out_dir)]
    cmd += ["--quantization", spec.quantization, "--copy_files", *spec.copy_files]
    return cmd


def convert(spec: HfSpec, src_dir: Path, out_dir: Path, run: Run = subprocess.run) -> None:
    """Convert offline: the converter reads only `src_dir` and must not reach the Hub."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    result = run(converter_command(spec, src_dir, out_dir), env=env, check=False)
    if result.returncode != 0 or not (out_dir / "model.bin").is_file():
        shutil.rmtree(out_dir, ignore_errors=True)
        raise DownloadError(f"{spec.name}: conversion failed (exit {result.returncode})")


def dir_rows(folder: Path) -> list[dict[str, Any]]:
    return [
        {"path": f.name, "bytes": f.stat().st_size, "sha256": sha256_of(f)}
        for f in sorted(folder.iterdir())
        if f.is_file()
    ]


def _converted_ok(out_dir: Path, previous: dict[str, Any] | None, spec: HfSpec) -> bool:
    if not previous or not out_dir.is_dir():
        return False
    src = previous.get("source") or {}
    pinned = (src.get("repo"), src.get("revision"), previous.get("quantization"))
    if pinned != (spec.repo, spec.revision, spec.quantization):
        return False
    return previous.get("files") == dir_rows(out_dir)


def fetch_hf_model(
    spec: HfSpec,
    models_dir: Path,
    *,
    manifest: dict[str, dict[str, Any]],
    keep_sources: bool = False,
    fetch: Fetch = hf_fetch,
    get_json: GetJson = hf_get_json,
    run: Run = subprocess.run,
) -> dict[str, Any]:
    _check_hf_spec(spec)
    out_dir = models_dir / spec.dir
    previous = manifest.get(spec.name)
    if previous is not None and _converted_ok(out_dir, previous, spec):
        print(f"OK    {spec.name}: already converted, SHA-256 verified")
        return previous
    expected = hf_expected_hashes(spec, get_json)
    src_dir = models_dir / "hf-sources" / f"{spec.repo.replace('/', '--')}-{spec.revision[:12]}"
    source_rows = fetch_hf_sources(spec, src_dir, expected, fetch=fetch)
    convert(spec, src_dir, out_dir, run=run)
    if not keep_sources:
        shutil.rmtree(src_dir, ignore_errors=True)
    print(f"OK    {spec.name}: converted to models/{spec.dir}")
    return {
        "name": spec.name,
        "dir": f"models/{spec.dir}",
        "licence": spec.licence,
        "source": {"host": HF_HOST, "repo": spec.repo, "revision": spec.revision},
        "source_files": source_rows,
        "quantization": spec.quantization,
        "converter": {"python": spec.python, "packages": list(spec.packages)},
        "files": dir_rows(out_dir),
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
    parser.add_argument(
        "--keep-sources", action="store_true", help="keep Hugging Face files after converting"
    )
    args = parser.parse_args(argv)

    specs = load_specs()
    hf_specs = load_hf_specs()
    unknown = [n for n in args.only if n not in specs and n not in hf_specs]
    if unknown:
        print(f"unknown model(s): {unknown}; known: {sorted(specs) + sorted(hf_specs)}")
        return 2
    if args.list:
        for name in args.only:
            if name in hf_specs:
                h = hf_specs[name]
                print(f"{name}: {h.repo}@{h.revision} -> models/{h.dir} ({h.quantization})")
            else:
                s = specs[name]
                print(f"{name}: {s.url} (digest: {s.digest or 'not pinned'})")
        return 0

    MODELS_DIR.mkdir(exist_ok=True)
    manifest = read_manifest()
    failures = 0
    for name in args.only:
        try:
            if name in hf_specs:
                manifest[name] = fetch_hf_model(
                    hf_specs[name], MODELS_DIR, manifest=manifest, keep_sources=args.keep_sources
                )
                write_manifest(MANIFEST, manifest)  # large downloads: save progress per model
                continue
            manifest[name] = fetch_model(specs[name], MODELS_DIR, manifest=manifest, pin=args.pin)
        except (DownloadError, OSError) as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    write_manifest(MANIFEST, manifest)
    print(f"manifest: {MANIFEST.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
