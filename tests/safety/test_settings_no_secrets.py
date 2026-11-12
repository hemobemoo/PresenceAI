"""Secrets never appear in the string forms of Settings (AGENTS.md rules 5 and 9)."""

from collections.abc import Callable

import pytest

from presenceai.settings import Settings

SECRET_VARS = ["OPENROUTER_API_KEY", "HF_TOKEN", "SUPABASE_URL", "SUPABASE_KEY", "MEDIA_KEY"]


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for name in SECRET_VARS:
        monkeypatch.setenv(name, f"sentinel-{name.lower()}-value")
    return Settings(_env_file=None)  # pyright: ignore[reportCallIssue]  # never read the real .env


def test_secrets_are_loaded(settings: Settings):
    assert settings.media_key is not None
    assert settings.media_key.get_secret_value() == "sentinel-media_key-value"


RENDERERS: list[Callable[[Settings], str]] = [
    repr,
    str,
    lambda s: s.model_dump_json(),
    lambda s: str(s.model_dump()),
]


@pytest.mark.parametrize("render", RENDERERS)
def test_no_secret_in_rendered_settings(settings: Settings, render: Callable[[Settings], str]):
    assert "sentinel" not in render(settings)
