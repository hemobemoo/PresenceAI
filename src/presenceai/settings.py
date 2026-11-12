"""Application settings: secrets from the environment / `.env`, everything else from `configs/`.

Secrets are `SecretStr` so they never appear in `repr()`, `str()`, logs or JSON dumps.
Refer to them only by variable name: OPENROUTER_API_KEY, HF_TOKEN, SUPABASE_URL,
SUPABASE_KEY, MEDIA_KEY.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"

Runtime = Literal["local", "space", "offline"]
Profile = Literal["lite", "standard", "space"]
Language = Literal["en", "ar"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureSwitches(_Section):
    camera_enabled: bool = True
    live_prompts_enabled: bool = True
    ai_wording_enabled: bool = False  # OpenRouter is called only when the user enables this
    keep_raw_media: bool = False


class RetentionPolicy(_Section):
    raw_media_days: int = Field(default=30, ge=1)
    study_media_max_days: int = Field(default=60, ge=1)


class ExpressionThresholds(_Section):
    """Thresholds for the five expression labels; null until an ADR decides them."""

    baseline_delta: float | None = None
    min_duration_s: float | None = None
    eyes_closed_min_s: float | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file=DEFAULT_CONFIG,
        extra="ignore",  # .env may hold variables for other tools (e.g. KAGGLE_*)
        frozen=True,
    )

    runtime: Runtime = "local"
    profile: Profile = "standard"
    language: Language = "en"
    presenceai_home: Path = Path.home() / ".presenceai"

    features: FeatureSwitches = FeatureSwitches()
    retention: RetentionPolicy = RetentionPolicy()
    expressions: ExpressionThresholds = ExpressionThresholds()

    openrouter_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None
    supabase_url: SecretStr | None = None
    supabase_key: SecretStr | None = None
    media_key: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: explicit args > environment > .env > configs/default.yaml
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
