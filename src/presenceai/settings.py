"""Application settings: secrets from the environment / `.env`, everything else from `configs/`.

Secrets are `SecretStr` so they never appear in `repr()`, `str()`, logs or JSON dumps.
Refer to them only by variable name: OPENROUTER_API_KEY, HF_TOKEN, SUPABASE_URL,
SUPABASE_KEY, MEDIA_KEY.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
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
MODELS_CONFIG = CONFIG_DIR / "models.yaml"

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
    # Optional overrides of configs/models.yaml `llm.openrouter` (ADR 0003). Not secrets.
    openrouter_model: str | None = None
    openrouter_fallback_models: str | None = None  # comma-separated
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


class OpenRouterModels(_Section):
    pinned: str | None
    fallbacks: tuple[str, ...]


def openrouter_models(
    settings: Settings | None = None, models_yaml: Path = MODELS_CONFIG
) -> OpenRouterModels:
    """Pinned and fallback OpenRouter model ids.

    `configs/models.yaml` (`llm.openrouter`) is the source of truth. Non-empty
    OPENROUTER_MODEL / OPENROUTER_FALLBACK_MODELS (comma-separated) override it (ADR 0003).
    """
    s = settings if settings is not None else get_settings()
    data = yaml.safe_load(models_yaml.read_text(encoding="utf-8")) or {}
    cfg = (data.get("llm") or {}).get("openrouter") or {}
    pinned: str | None = cfg.get("pinned") or None
    fallbacks = tuple(str(f) for f in (cfg.get("fallbacks") or []) if f)
    if s.openrouter_model and s.openrouter_model.strip():
        pinned = s.openrouter_model.strip()
    if s.openrouter_fallback_models and s.openrouter_fallback_models.strip():
        fallbacks = tuple(f.strip() for f in s.openrouter_fallback_models.split(",") if f.strip())
    return OpenRouterModels(pinned=pinned, fallbacks=fallbacks)
