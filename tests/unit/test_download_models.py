"""Tests for scripts/download_models.py with a fake fetcher (no network)."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "download_models.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_models"] = module
    spec.loader.exec_module(module)
    return module


dm: Any = _load()
URL = "https://storage.googleapis.com/mediapipe-models/x/x/float16/latest/x.task"
DATA = b"fake model bytes"
DIGEST = hashlib.sha256(DATA).hexdigest()


def _spec(digest: str | None = None, url: str = URL) -> Any:
    return dm.ModelSpec(name="x", url=url, licence="Apache-2.0", digest=digest)


class FakeFetch:
    def __init__(self, data: bytes = DATA) -> None:
        self.data = data
        self.calls: list[str] = []

    def __call__(self, url: str, dest: Path, max_bytes: int) -> None:
        self.calls.append(url)
        if len(self.data) > max_bytes:
            raise dm.DownloadError("too large")
        dest.write_bytes(self.data)


def test_specs_from_config_cover_the_three_landmarkers():
    specs = dm.load_specs(REPO / "configs" / "models.yaml")
    assert set(dm.DEFAULT_MODELS) <= set(specs)
    for name in dm.DEFAULT_MODELS:
        assert specs[name].url.startswith("https://storage.googleapis.com/mediapipe-models/")
        assert specs[name].licence == "Apache-2.0"


def test_unpinned_model_needs_pin_flag(tmp_path: Path):
    fetch = FakeFetch()
    with pytest.raises(dm.DownloadError, match="--pin"):
        dm.fetch_model(_spec(), tmp_path, manifest={}, pin=False, fetch=fetch)
    assert not list(tmp_path.iterdir())


def test_pin_records_digest(tmp_path: Path):
    entry = dm.fetch_model(_spec(), tmp_path, manifest={}, pin=True, fetch=FakeFetch())
    assert entry["sha256"] == DIGEST
    assert entry["bytes"] == len(DATA)
    assert entry["pinned_on_first_download"] is True
    assert (tmp_path / "x.task").read_bytes() == DATA


def test_digest_mismatch_fails_and_leaves_nothing(tmp_path: Path):
    with pytest.raises(dm.DownloadError, match="SHA-256"):
        dm.fetch_model(_spec("0" * 64), tmp_path, manifest={}, pin=False, fetch=FakeFetch())
    assert not list(tmp_path.iterdir())


def test_config_digest_verifies(tmp_path: Path):
    entry = dm.fetch_model(_spec(DIGEST), tmp_path, manifest={}, pin=False, fetch=FakeFetch())
    assert entry["pinned_on_first_download"] is False


def test_manifest_digest_is_used_when_config_has_none(tmp_path: Path):
    manifest = {"x": {"sha256": DIGEST}}
    entry = dm.fetch_model(_spec(), tmp_path, manifest=manifest, pin=False, fetch=FakeFetch())
    assert entry["sha256"] == DIGEST


def test_existing_verified_file_is_not_downloaded_again(tmp_path: Path):
    (tmp_path / "x.task").write_bytes(DATA)
    fetch = FakeFetch()
    dm.fetch_model(_spec(DIGEST), tmp_path, manifest={}, pin=False, fetch=fetch)
    assert fetch.calls == []


def test_existing_file_with_wrong_digest_is_replaced(tmp_path: Path):
    (tmp_path / "x.task").write_bytes(b"corrupt")
    fetch = FakeFetch()
    dm.fetch_model(_spec(DIGEST), tmp_path, manifest={}, pin=False, fetch=fetch)
    assert fetch.calls == [URL]
    assert (tmp_path / "x.task").read_bytes() == DATA


@pytest.mark.parametrize(
    "url",
    [
        "http://storage.googleapis.com/mediapipe-models/x.task",
        "https://example.com/x.task",
        "file:///etc/passwd",
    ],
)
def test_only_https_from_allowed_hosts(tmp_path: Path, url: str):
    with pytest.raises(dm.DownloadError, match="not allowed"):
        dm.fetch_model(_spec(url=url), tmp_path, manifest={}, pin=True, fetch=FakeFetch())


def test_write_manifest(tmp_path: Path):
    path = tmp_path / "MANIFEST.json"
    dm.write_manifest(path, {"x": {"name": "x", "sha256": DIGEST}})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"models": [{"name": "x", "sha256": DIGEST}]}
    assert dm.read_manifest(path) == {"x": {"name": "x", "sha256": DIGEST}}


def test_read_manifest_empty_template():
    assert dm.read_manifest(REPO / "models" / "MANIFEST.json") is not None
