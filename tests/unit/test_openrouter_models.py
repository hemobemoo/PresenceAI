"""OpenRouter model ids: configs/models.yaml is the source of truth; .env may override.

ADR 0003.
"""

from pathlib import Path
from typing import Any

import pytest

from presenceai.settings import MODELS_CONFIG, Settings, openrouter_models


def _settings(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, **kwargs)  # pyright: ignore[reportCallIssue]


def _models_yaml(tmp_path: Path, pinned: str | None, fallbacks: list[str | None]) -> Path:
    def q(v: str | None) -> str:
        return "null" if v is None else f'"{v}"'

    p = tmp_path / "models.yaml"
    p.write_text(
        "llm:\n  openrouter:\n"
        f"    pinned: {q(pinned)}\n"
        f"    fallbacks: [{', '.join(q(f) for f in fallbacks)}]\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_FALLBACK_MODELS", raising=False)


def test_config_values_used_without_overrides(tmp_path: Path):
    cfg = _models_yaml(tmp_path, "a/pinned:free", ["b/one:free", "c/two:free"])
    m = openrouter_models(_settings(), cfg)
    assert m.pinned == "a/pinned:free"
    assert m.fallbacks == ("b/one:free", "c/two:free")


def test_null_fallbacks_are_dropped(tmp_path: Path):
    cfg = _models_yaml(tmp_path, None, [None, None])
    m = openrouter_models(_settings(), cfg)
    assert m.pinned is None
    assert m.fallbacks == ()


def test_env_overrides_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = _models_yaml(tmp_path, "a/pinned:free", ["b/one:free", "c/two:free"])
    monkeypatch.setenv("OPENROUTER_MODEL", "x/override:free")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", " y/one:free , z/two:free ,")
    m = openrouter_models(_settings(), cfg)
    assert m.pinned == "x/override:free"
    assert m.fallbacks == ("y/one:free", "z/two:free")


def test_empty_env_values_do_not_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = _models_yaml(tmp_path, "a/pinned:free", ["b/one:free"])
    monkeypatch.setenv("OPENROUTER_MODEL", "")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", " ")
    m = openrouter_models(_settings(), cfg)
    assert m.pinned == "a/pinned:free"
    assert m.fallbacks == ("b/one:free",)


def test_repo_config_has_pinned_and_at_least_one_fallback():
    m = openrouter_models(_settings(), MODELS_CONFIG)
    assert m.pinned is not None and m.pinned.endswith(":free")
    assert len(m.fallbacks) >= 1
    assert all(f.endswith(":free") for f in m.fallbacks)
    assert len({m.pinned, *m.fallbacks}) == 1 + len(m.fallbacks)  # no model listed twice
