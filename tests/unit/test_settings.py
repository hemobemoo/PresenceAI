from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from presenceai.settings import DEFAULT_CONFIG, Settings

SECRET_VARS = ["OPENROUTER_API_KEY", "HF_TOKEN", "SUPABASE_URL", "SUPABASE_KEY", "MEDIA_KEY"]
# Variables read by their exact documented name (.env.example), not with the PRESENCEAI_ prefix.
NAMED_SECRETS = {
    "OPENROUTER_API_KEY": "openrouter_api_key",
    "HF_TOKEN": "hf_token",
    "SUPABASE_URL": "supabase_url",
    "SUPABASE_KEY": "supabase_key",
    "MEDIA_KEY": "media_key",
}
NAMED_PLAIN = {
    "OPENROUTER_MODEL": "openrouter_model",
    "OPENROUTER_FALLBACK_MODELS": "openrouter_fallback_models",
}
# Generic variables that other programs set (Linux locale, CI, shells) and must not leak in.
GENERIC_ENV = {
    "LANGUAGE": "en_US",
    "LANG": "en_US.UTF-8",
    "LC_ALL": "en_US.UTF-8",
    "RUNTIME": "cloud",
    "PROFILE": "/home/user/.profile",
}


def _settings(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, **kwargs)  # pyright: ignore[reportCallIssue]


def test_defaults_come_from_yaml():
    assert DEFAULT_CONFIG.is_file()
    s = _settings()
    assert s.runtime == "local"
    assert s.features.ai_wording_enabled is False
    assert s.retention.raw_media_days == 30


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("PRESENCEAI_RUNTIME", "offline")
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


def test_generic_locale_and_shell_vars_are_ignored(monkeypatch: pytest.MonkeyPatch):
    """Colab and many Linux hosts set LANGUAGE=en_US; it must not reach `language`."""
    for name, value in GENERIC_ENV.items():
        monkeypatch.setenv(name, value)
    s = _settings()
    assert s.language == "en"  # configs/default.yaml
    assert s.runtime == "local"
    assert s.profile == "standard"


def test_prefixed_vars_override_yaml(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANGUAGE", "en_US")
    monkeypatch.setenv("PRESENCEAI_LANGUAGE", "ar")
    monkeypatch.setenv("PRESENCEAI_PROFILE", "lite")
    s = _settings()
    assert s.language == "ar"
    assert s.profile == "lite"


@pytest.mark.parametrize(("var", "field"), NAMED_SECRETS.items())
def test_secret_read_by_exact_name(monkeypatch: pytest.MonkeyPatch, var: str, field: str):
    monkeypatch.setenv(var, f"sentinel-{var.lower()}")
    value = getattr(_settings(), field)
    assert value is not None
    assert value.get_secret_value() == f"sentinel-{var.lower()}"


@pytest.mark.parametrize(("var", "field"), NAMED_PLAIN.items())
def test_openrouter_override_read_by_exact_name(
    monkeypatch: pytest.MonkeyPatch, var: str, field: str
):
    monkeypatch.setenv(var, "org/model:free")
    assert getattr(_settings(), field) == "org/model:free"


@pytest.mark.parametrize("field", [*NAMED_SECRETS.values(), *NAMED_PLAIN.values()])
def test_prefixed_name_does_not_replace_documented_name(
    monkeypatch: pytest.MonkeyPatch, field: str
):
    for var in [*NAMED_SECRETS, *NAMED_PLAIN]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(f"PRESENCEAI_{field.upper()}", "sentinel-prefixed")
    assert getattr(_settings(), field) is None


def test_fields_still_settable_by_name():
    s = _settings(language="ar", openrouter_model="org/m:free")
    assert s.language == "ar"
    assert s.openrouter_model == "org/m:free"


def test_dotenv_file_uses_same_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for var in [*NAMED_SECRETS, *GENERIC_ENV, "PRESENCEAI_LANGUAGE"]:
        monkeypatch.delenv(var, raising=False)
    env_file = tmp_path / ".env"  # a fake file in tmp_path, never the real .env
    env_file.write_text(
        "MEDIA_KEY=sentinel-dotenv\nLANGUAGE=en_US\nPRESENCEAI_LANGUAGE=ar\nKAGGLE_API_TOKEN=x\n",
        encoding="utf-8",
    )
    s = Settings(_env_file=env_file)  # pyright: ignore[reportCallIssue]
    assert s.media_key is not None
    assert s.media_key.get_secret_value() == "sentinel-dotenv"
    assert s.language == "ar"
