"""Evaluation config (configs/evaluation.yaml): the E2 protocol and the C1 build.

ADR 0009 (targets), ADR 0011 (split), ADR 0014 (C1, matching rules). Kept apart from
`settings.py` on purpose: it needs only pydantic and PyYAML, so the C1 generator can run on
Colab without the app's dependencies (ADR 0014 §1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


EVALUATION_CONFIG = CONFIG_DIR / "evaluation.yaml"
Range = tuple[float, float]


class E2Split(_Section):
    unit: Literal["script"]
    dev_fraction: float = Field(gt=0, lt=1)
    seed: int
    stratify_by: tuple[str, ...]
    print_counts: bool
    tune_on: Literal["dev"]
    score_on: Literal["test"]


class E2Match(_Section):
    onset_tolerance_s: float = Field(gt=0)
    require_overlap: bool


class E2Noise(_Section):
    part: Literal["dev", "test"]
    snr_db: tuple[float, ...]
    color: Literal["pink", "white"]
    seed: int


class E2Settings(_Section):
    wpm_error_pct_max: float = Field(gt=0)
    pause_f1_min: dict[str, float]
    duration_error_s_max: float = Field(gt=0)
    report_per_clip: bool
    report_pause_position: bool
    split: E2Split
    wpm_judged_on: Literal["all_clips"]
    match: E2Match
    f1_exclusion_s: float = Field(ge=0)
    min_pause_s: float = Field(gt=0)
    pipeline: tuple[str, ...]
    noise: E2Noise

    @property
    def thresholds(self) -> tuple[float, ...]:
        return tuple(sorted(float(k) for k in self.pause_f1_min))


class C1Gaps(_Section):
    short_s: Range
    pause_s: Range
    pause_distribution: Literal["log_uniform", "uniform"]
    pause_probability: dict[Literal["sentence_boundary", "clause_boundary", "mid_sentence"], float]


class C1Fillers(_Section):
    clip_share: float = Field(ge=0, le=1)
    per_clip: tuple[int, int]
    tokens: tuple[str, ...]
    after_s: Range


class C1Trim(_Section):
    frame_ms: float = Field(gt=0)
    threshold_db_below_peak: float = Field(gt=0)
    internal_min_silence_s: float = Field(gt=0)
    internal_joins_truth_s: float = Field(gt=0)


class C1Settings(_Section):
    seed: int
    scripts: str
    generator: dict[str, object]
    voices: dict[str, Literal["a", "b"]]
    scripts_per_voice: int = Field(gt=0)
    renders_per_script: int = Field(gt=0)
    target_wpm: tuple[float, ...]
    accept_within_wpm: float = Field(gt=0)
    max_rerenders: int = Field(ge=0)
    wpm_range: Range
    rate_bands: tuple[Range, ...]
    clip_s: Range
    speed_range: Range
    words_per_script: tuple[int, int]
    max_words_per_sentence_mark: int = Field(gt=0)
    gaps: C1Gaps
    fillers: C1Fillers
    edge_silence_s: float = Field(ge=0)
    noise_floor_dbfs: float
    trim: C1Trim
    timing_run_clips: int = Field(gt=0)


class B1Labels(_Section):
    overlap_margin_s: float = Field(ge=0)


class B1Settings(_Section):
    labels: B1Labels


class EvaluationSettings(_Section):
    model_config = ConfigDict(extra="ignore", frozen=True)  # e3 is read by its own experiment
    e2: E2Settings
    b1: B1Settings
    c1: C1Settings


def load_evaluation(path: Path = EVALUATION_CONFIG) -> EvaluationSettings:
    return EvaluationSettings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
