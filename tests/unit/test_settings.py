from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from presenceai.settings import DEFAULT_CONFIG, Settings

SECRET_VARS = ["OPENROUTER_API_KEY", "HF_TOKEN", "SUPABASE_URL", "SUPABASE_KEY", "MEDIA_KEY"]


def _settings(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, **kwargs)  # pyright: ignore[reportCallIssue]


def test_defaults_come_from_yaml():
    assert DEFAULT_CONFIG.is_file()
    s = _settings()
    assert s.runtime == "local"
    assert s.features.ai_wording_enabled is False
    assert s.retention.raw_media_days == 30


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("RUNTIME", "offline")
    monkeypatch.setenv("PRESENCEAI_HOME", str(tmp_path))
    s = _settings()
    assert s.runtime == "offline"
    assert s.presenceai_home == tmp_path


def test_secrets_default_to_none(monkeypatch: pytest.MonkeyPatch):
    for name in SECRET_VARS:
        monkeypatch.delenv(name, raising=False)
    s = _settings()
    assert s.openrouter_api_key is None
    assert s.media_key is None


def test_invalid_runtime_rejected():
    with pytest.raises(ValidationError):
        _settings(runtime="cloud")
