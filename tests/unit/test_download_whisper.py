"""scripts/download_models.py, Whisper part: fetched from huggingface.co at a pinned commit,
hash-checked, converted offline. Fake tree, fetch and converter: no network, no torch."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "download_models.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_models_whisper", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_models_whisper"] = module
    spec.loader.exec_module(module)
    return module


dm: Any = _load()
REV = "a" * 40
SMALL = b'{"tiny": true}'
BIG = b"weights" * 100


def _git_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data, usedforsecurity=False).hexdigest()


def _hf_spec(**kw: Any) -> Any:
    base: dict[str, Any] = {
        "name": "whisper_x",
        "repo": "openai/whisper-x",
        "revision": REV,
        "quantization": "int8",
        "dir": "whisper-x-ct2",
        "licence": "MIT",
        "files": ("config.json", "model.safetensors"),
        "copy_files": ("tokenizer.json",),
        "packages": ("ctranslate2==4.9.0",),
        "python": "3.12",
    }
    base.update(kw)
    return dm.HfSpec(**base)


def _tree(big: bytes = BIG, small: bytes = SMALL) -> Any:
    items = [
        {"type": "file", "path": "config.json", "oid": _git_sha1(small)},
        {
            "type": "file",
            "path": "model.safetensors",
            "oid": "0" * 40,
            "lfs": {"oid": hashlib.sha256(big).hexdigest()},
        },
        {"type": "directory", "path": "sub", "oid": "1" * 40},
    ]
    return lambda url: items


class HfFetch:
    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = files or {"config.json": SMALL, "model.safetensors": BIG}
        self.calls: list[str] = []

    def __call__(self, url: str, dest: Path, max_bytes: int) -> None:
        self.calls.append(url)
        dest.write_bytes(self.files[url.rsplit("/", 1)[1]])


class FakeConverter:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.cmds: list[list[str]] = []
        self.envs: list[dict[str, str]] = []

    def __call__(self, cmd: list[str], env: dict[str, str], check: bool) -> Any:
        self.cmds.append(cmd)
        self.envs.append(env)
        out = Path(cmd[cmd.index("--output_dir") + 1])
        if self.returncode == 0:
            out.mkdir(parents=True)
            (out / "model.bin").write_bytes(b"ct2")
            (out / "config.json").write_text("{}", encoding="utf-8")
        return type("R", (), {"returncode": self.returncode})()


def _run(tmp_path: Path, spec: Any = None, **kw: Any) -> Any:
    args: dict[str, Any] = {
        "manifest": {},
        "fetch": HfFetch(),
        "get_json": _tree(),
        "run": FakeConverter(),
    }
    args.update(kw)
    return dm.fetch_hf_model(spec or _hf_spec(), tmp_path, **args)


def test_real_config_pins_both_whisper_models_on_huggingface():
    specs = dm.load_hf_specs(REPO / "configs" / "models.yaml")
    assert set(specs) == {"whisper_small", "whisper_turbo"}
    assert specs["whisper_small"].repo == "openai/whisper-small"
    assert specs["whisper_turbo"].repo == "openai/whisper-large-v3-turbo"
    # S2 converted this exact revision (experiments/configs/s2_zerogpu.yaml)
    assert specs["whisper_turbo"].revision == "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    assert specs["whisper_turbo"].quantization == "float16"
    for s in specs.values():
        dm._check_hf_spec(s)
        assert s.licence == "MIT"
        assert "model.safetensors" in s.files and "pytorch_model.bin" not in s.files
    assert "whisper_small" not in dm.DEFAULT_MODELS  # large downloads only when named


def test_hf_model_is_fetched_verified_converted_and_recorded(tmp_path: Path):
    fetch = HfFetch()
    entry = _run(tmp_path, fetch=fetch)
    assert fetch.calls == [
        f"https://huggingface.co/openai/whisper-x/resolve/{REV}/config.json",
        f"https://huggingface.co/openai/whisper-x/resolve/{REV}/model.safetensors",
    ]
    assert entry["source"] == {
        "host": "huggingface.co",
        "repo": "openai/whisper-x",
        "revision": REV,
    }
    assert {r["path"]: r["sha256"] for r in entry["source_files"]} == {
        "config.json": hashlib.sha256(SMALL).hexdigest(),
        "model.safetensors": hashlib.sha256(BIG).hexdigest(),
    }
    assert {r["path"] for r in entry["files"]} == {"config.json", "model.bin"}
    assert entry["dir"] == "models/whisper-x-ct2"
    assert not (tmp_path / "hf-sources" / f"openai--whisper-x-{REV[:12]}").exists()


def test_keep_sources(tmp_path: Path):
    _run(tmp_path, keep_sources=True)
    assert (tmp_path / "hf-sources" / f"openai--whisper-x-{REV[:12]}" / "config.json").is_file()


def test_converter_runs_offline_on_the_local_folder(tmp_path: Path):
    conv = FakeConverter()
    _run(tmp_path, run=conv)
    (cmd,), (env,) = conv.cmds, conv.envs
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert cmd[:3] == ["uv", "run", "--no-project"]
    assert Path(cmd[cmd.index("--model") + 1]).is_relative_to(tmp_path)  # never a Hub id
    assert cmd[cmd.index("--quantization") + 1] == "int8"
    assert cmd[cmd.index("--with") + 1] == "ctranslate2==4.9.0"


def test_tampered_large_file_is_rejected(tmp_path: Path):
    fetch = HfFetch({"config.json": SMALL, "model.safetensors": b"evil"})
    with pytest.raises(dm.DownloadError, match="sha256 does not match"):
        _run(tmp_path, fetch=fetch)
    assert not (tmp_path / "whisper-x-ct2").exists()


def test_tampered_small_file_is_rejected(tmp_path: Path):
    fetch = HfFetch({"config.json": b"{}", "model.safetensors": BIG})
    with pytest.raises(dm.DownloadError, match="git-sha1 does not match"):
        _run(tmp_path, fetch=fetch)


def test_file_missing_from_the_pinned_commit_fails_before_download(tmp_path: Path):
    fetch = HfFetch()
    with pytest.raises(dm.DownloadError, match=r"vocab\.json"):
        _run(tmp_path, _hf_spec(files=("config.json", "vocab.json")), fetch=fetch)
    assert fetch.calls == []


def test_failed_conversion_leaves_no_model_folder(tmp_path: Path):
    with pytest.raises(dm.DownloadError, match="conversion failed"):
        _run(tmp_path, run=FakeConverter(returncode=1))
    assert not (tmp_path / "whisper-x-ct2").exists()


def test_verified_conversion_is_not_redone(tmp_path: Path):
    entry = _run(tmp_path)
    fetch, conv = HfFetch(), FakeConverter()
    again = _run(tmp_path, manifest={"whisper_x": entry}, fetch=fetch, run=conv)
    assert again == entry and fetch.calls == [] and conv.cmds == []


def test_changed_converted_file_triggers_reconversion(tmp_path: Path):
    entry = _run(tmp_path)
    (tmp_path / "whisper-x-ct2" / "model.bin").write_bytes(b"corrupt")
    conv = FakeConverter()
    _run(tmp_path, manifest={"whisper_x": entry}, run=conv)
    assert len(conv.cmds) == 1


def test_new_revision_triggers_a_fresh_download(tmp_path: Path):
    entry = _run(tmp_path)
    fetch = HfFetch()
    _run(tmp_path, _hf_spec(revision="b" * 40), manifest={"whisper_x": entry}, fetch=fetch)
    assert len(fetch.calls) == 2


@pytest.mark.parametrize(
    "kw",
    [
        {"revision": "main"},
        {"revision": "A" * 40},
        {"dir": "../escape"},
        {"dir": ".."},
        {"files": ()},
    ],
)
def test_bad_hf_specs_are_refused(tmp_path: Path, kw: dict[str, Any]):
    with pytest.raises(dm.DownloadError):
        _run(tmp_path, _hf_spec(**kw))


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://huggingface.co/api/resolve-cache/x", True),
        ("https://us.aws.cdn.hf.co/xet-bridge-us/abc", True),
        ("https://cas-bridge.xethub.hf.co/x", True),
        ("http://us.aws.cdn.hf.co/x", False),
        ("https://evil.example.com/x", False),
        ("https://hf.co.evil.com/x", False),
        ("https://notahf.co/x", False),
    ],
)
def test_redirect_policy(url: str, ok: bool):
    assert dm.redirect_allowed(url) is ok


@pytest.mark.parametrize(
    "url", ["https://cdn.hf.co/x", "http://huggingface.co/x", "https://example.com/x"]
)
def test_requests_go_to_huggingface_co_only(url: str):
    with pytest.raises(dm.DownloadError, match="not allowed"):
        dm._check_hf_url(url)


def test_git_blob_sha1_matches_git(tmp_path: Path):
    path = tmp_path / "f"
    path.write_bytes(b"hello\n")
    assert dm.git_blob_sha1(path) == "ce013625030ba8dba906f756967f9e9ca394464a"  # git hash-object
