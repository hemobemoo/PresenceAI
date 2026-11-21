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


# --- spaCy wheel (ADR 0018) -------------------------------------------------------------------


def test_spacy_spec_is_pinned_and_from_github():
    spec = dm.load_specs(REPO / "configs" / "models.yaml")["spacy_en"]
    assert spec.url.startswith("https://github.com/explosion/spacy-models/releases/download/")
    assert spec.digest and len(spec.digest) == 64 and spec.licence == "MIT"
    assert spec.unpack == "en_core_web_sm/en_core_web_sm-3.8.0"
    assert spec.dir == "en_core_web_sm-3.8.0"
    assert "spacy_en" not in dm.DEFAULT_MODELS  # only when named


def _wheel(path: Path, members: dict[str, bytes]) -> Path:
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_wheel_model_folder_is_unpacked(tmp_path: Path):
    wheel = _wheel(
        tmp_path / "m.whl",
        {"pkg/pkg-1.0/config.cfg": b"cfg", "pkg/pkg-1.0/ner/model": b"w", "pkg/__init__.py": b""},
    )
    out = tmp_path / "models" / "pkg-1.0"
    dm.unpack_wheel(wheel, "pkg/pkg-1.0", out)
    assert (out / "config.cfg").read_bytes() == b"cfg"
    assert (out / "ner" / "model").read_bytes() == b"w"
    assert not (out / "__init__.py").exists()  # only the model folder


def test_unsafe_wheel_paths_are_refused(tmp_path: Path):
    wheel = _wheel(tmp_path / "bad.whl", {"pkg/m/../../../evil.txt": b"x"})
    with pytest.raises(dm.DownloadError, match="unsafe"):
        dm.unpack_wheel(wheel, "pkg/m", tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://release-assets.githubusercontent.com/x", True),
        ("https://objects.githubusercontent.com/x", True),
        ("https://github.com/x", True),
        ("http://objects.githubusercontent.com/x", False),
        ("https://evil.example.com/x", False),
    ],
)
def test_redirects_only_to_github_asset_hosts(url: str, ok: bool):
    assert dm.fetch_redirect_allowed(url) is ok


# --- e5-small embeddings (ADR 0020) ------------------------------------------------------------


def test_e5_spec_is_pinned_at_a_commit():
    spec = dm.load_hf_plain_specs(REPO / "configs" / "models.yaml")["e5_small"]
    assert spec.repo == "intfloat/multilingual-e5-small" and len(spec.revision) == 40
    assert spec.licence == "MIT"
    assert set(spec.files) == {"onnx/model.onnx", "onnx/tokenizer.json"}
    assert {f for f, _ in spec.pins} == set(spec.files)
    assert all(len(h) == 64 for _, h in spec.pins)


def _plain(pins: dict[str, str]) -> Any:
    return dm.HfSpec(
        name="e", repo="org/m", revision="a" * 40, quantization="none", dir="m", licence="MIT",
        files=("onnx/model.onnx",), copy_files=(), packages=(), python="",
        pins=tuple(pins.items()),
    )  # fmt: skip


def test_plain_fetch_checks_hugging_face_hash_and_pin(tmp_path: Path):
    listing = [{"type": "file", "path": "onnx/model.onnx", "oid": "x", "lfs": {"oid": DIGEST}}]
    entry = dm.fetch_hf_plain(_plain({"onnx/model.onnx": DIGEST}), tmp_path,
                              fetch=FakeFetch(), get_json=lambda _: listing)  # fmt: skip
    assert (tmp_path / "m" / "onnx" / "model.onnx").read_bytes() == DATA
    assert entry["files"][0]["sha256"] == DIGEST
    with pytest.raises(dm.DownloadError, match="pin"):
        dm.fetch_hf_plain(_plain({"onnx/model.onnx": "0" * 64}), tmp_path,
                          fetch=FakeFetch(), get_json=lambda _: listing)  # fmt: skip
    with pytest.raises(dm.DownloadError, match="no SHA-256 pinned"):
        dm.fetch_hf_plain(_plain({}), tmp_path, fetch=FakeFetch(), get_json=lambda _: listing)
