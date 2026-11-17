"""Application settings: secrets from the environment / `.env`, everything else from `configs/`.

Secrets are `SecretStr` so they never appear in `repr()`, `str()`, logs or JSON dumps.
Refer to them only by variable name: OPENROUTER_API_KEY, HF_TOKEN, SUPABASE_URL,
SUPABASE_KEY, MEDIA_KEY.

Environment variable names: ordinary fields need the `PRESENCEAI_` prefix
(`PRESENCEAI_RUNTIME`, `PRESENCEAI_PROFILE`, `PRESENCEAI_LANGUAGE`, `PRESENCEAI_HOME`), so
generic variables set by the OS or other tools (e.g. the Linux locale `LANGUAGE=en_US`) are
never read as settings. Secrets and the documented OpenRouter overrides keep their exact
names (see `.env.example`) through `validation_alias`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr
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


class AsrSettings(_Section):
    """Speech recognition options (FINAL_04 §2, FINAL_06 §1); same decoding as S2/S3."""

    model: Literal["whisper_small", "whisper_turbo"] = "whisper_small"  # configs/models.yaml asr
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    compute_type: str = "int8"
    beam_size: int = Field(default=5, ge=1)
    temperature: float = Field(default=0.0, ge=0)
    condition_on_previous_text: bool = False


class VadSettings(_Section):
    threshold: float = Field(gt=0, lt=1)
    min_speech_duration_ms: int = Field(ge=0)
    min_silence_duration_ms: int = Field(ge=0)
    speech_pad_ms: int = Field(ge=0)


class PauseSettings(_Section):
    thresholds_s: tuple[float, ...] = Field(min_length=1)
    sentence_marks: tuple[str, ...] = Field(min_length=1)
    clause_marks: tuple[str, ...]
    max_words_per_sentence_mark: float = Field(gt=0)


class RateSettings(_Section):
    window_s: float = Field(gt=0)
    hop_s: float = Field(gt=0)
    min_speaking_time_s: float = Field(ge=0)


class SyllableSettings(_Section):
    window_ms: int = Field(gt=0)
    hop_ms: int = Field(gt=0)
    reference_quantile: float = Field(gt=0, le=1)
    silence_threshold_db: float = Field(le=0)
    min_dip_db: float = Field(ge=0)


class ConfidenceBand(_Section):
    min_word_probability: float = Field(ge=0, le=1)
    min_speaking_time_s: float = Field(ge=0)


class ConfidenceSettings(_Section):
    high: ConfidenceBand
    medium: ConfidenceBand


class PitchSettings(_Section):
    fmin_hz: float = Field(gt=0)
    fmax_hz: float = Field(gt=0)
    frame_length: int = Field(gt=0)
    hop_length: int = Field(gt=0)
    limit_band: float = Field(ge=0, lt=1)


class LoudnessSettings(_Section):
    frame_ms: int = Field(gt=0)
    window_s: float = Field(gt=0)
    min_speech_share: float = Field(ge=0, le=1)
    clip_level: float = Field(gt=0, le=1)
    clip_min_run: int = Field(ge=1)
    clip_peak_min: float = Field(gt=0, le=1)
    clip_pileup_band: float = Field(gt=0, lt=0.5)
    clip_pileup_ratio: float = Field(gt=1)
    noise_percentile: float = Field(ge=0, le=100)
    snr_not_measurable_db: float
    snr_low_db: float
    min_non_speech_s: float = Field(ge=0)


class PitchVariationSettings(_Section):
    octave_limit_st: float = Field(gt=0)
    fold_tolerance_st: float = Field(gt=0)
    running_median_s: float = Field(gt=0)
    max_corrected_share: float = Field(ge=0, le=1)
    max_at_limit_share: float = Field(ge=0, le=1)
    filler_margin_s: float = Field(ge=0)
    min_voiced_s: float = Field(ge=0)
    low_voiced_s: float = Field(ge=0)


ConfidenceName = Literal["high", "medium", "low"]


class AcousticFillerSettings(_Section):
    min_duration_s: float = Field(gt=0)
    min_voiced_fraction: float = Field(ge=0, le=1)
    max_pitch_sd_semitones: float = Field(gt=0)
    max_unvoiced_gap_s: float = Field(ge=0)
    agreement_tolerance_s: float = Field(ge=0)


class FillerSettings(_Section):
    confidence_cap_without_verbatim: dict[Language, ConfidenceName]
    acoustic: AcousticFillerSettings


class RepetitionSettings(_Section):
    max_ngram: int = Field(ge=1)
    window_s: float = Field(gt=0)


class SpeechSettings(_Section):
    """ADR 0011-0013 values. No defaults in code: thresholds live only in configs/ (rule 7)."""

    vad: VadSettings
    pauses: PauseSettings
    rate: RateSettings
    syllables: SyllableSettings
    confidence: ConfidenceSettings
    pitch: PitchSettings
    fillers: FillerSettings
    repetitions: RepetitionSettings
    loudness: LoudnessSettings
    pitch_variation: PitchVariationSettings


def _named(env_var: str) -> AliasChoices:
    """Read `env_var` by its exact name (no prefix); the field name stays valid for args."""
    return AliasChoices(env_var, env_var.lower())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PRESENCEAI_",  # generic names such as LANGUAGE (Linux locale) are ignored
        yaml_file=DEFAULT_CONFIG,
        extra="ignore",  # .env may hold variables for other tools (e.g. KAGGLE_*)
        frozen=True,
    )

    runtime: Runtime = "local"
    profile: Profile = "standard"
    language: Language = "en"
    presenceai_home: Path = Field(
        default=Path.home() / ".presenceai", validation_alias=_named("PRESENCEAI_HOME")
    )

    features: FeatureSwitches = FeatureSwitches()
    retention: RetentionPolicy = RetentionPolicy()
    expressions: ExpressionThresholds = ExpressionThresholds()
    asr: AsrSettings = AsrSettings()
    speech: SpeechSettings  # required: comes from configs/default.yaml

    # Read by exact variable name (no prefix); an alias replaces env_prefix for these fields.
    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias=_named("OPENROUTER_API_KEY")
    )
    # Optional overrides of configs/models.yaml `llm.openrouter` (ADR 0003). Not secrets.
    openrouter_model: str | None = Field(default=None, validation_alias=_named("OPENROUTER_MODEL"))
    openrouter_fallback_models: str | None = Field(  # comma-separated
        default=None, validation_alias=_named("OPENROUTER_FALLBACK_MODELS")
    )
    hf_token: SecretStr | None = Field(default=None, validation_alias=_named("HF_TOKEN"))
    supabase_url: SecretStr | None = Field(default=None, validation_alias=_named("SUPABASE_URL"))
    supabase_key: SecretStr | None = Field(default=None, validation_alias=_named("SUPABASE_KEY"))
    media_key: SecretStr | None = Field(default=None, validation_alias=_named("MEDIA_KEY"))

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
    return Settings()  # pyright: ignore[reportCallIssue]  (fields come from configs/ and the env)


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
