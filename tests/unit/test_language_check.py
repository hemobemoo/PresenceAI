"""Language check on the first speech (FINAL_04 §5, ADR 0016): a notice, never an error."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from presenceai.schemas.speech import LanguageCheck, SpeechRegion
from presenceai.settings import AsrSettings, LanguageCheckSettings
from presenceai.speech import analyze, language_check

SR = 16_000
CFG = LanguageCheckSettings(window_s=30.0, mismatch_min_probability=0.5)  # mechanism tests
# configs/default.yaml: 0.95, provisional and B1-informed (owner, 2026-11-21, ADR 0016 item 7)
DEFAULT = LanguageCheckSettings(window_s=30.0, mismatch_min_probability=0.95)


class _Model:
    def __init__(self, code: str, p: float, others: list[tuple[str, float]] | None = None):
        self.code, self.p = code, p
        self.others = others if others is not None else []
        self.audio: list[np.ndarray] = []
        self.options: list[dict[str, Any]] = []

    def detect_language(self, audio: np.ndarray, /, **options: Any) -> tuple[str, float, Any]:
        self.audio.append(audio)
        self.options.append(options)
        return self.code, self.p, [(self.code, self.p), *self.others]


def _regions(*spans: tuple[float, float]) -> tuple[SpeechRegion, ...]:
    return tuple(SpeechRegion(start_s=a, end_s=b) for a, b in spans)


def test_window_starts_at_first_speech_and_lasts_30_s():
    samples = np.zeros(SR * 60, dtype=np.float32)
    model = _Model("en", 0.99)
    result = language_check.check(model, samples, _regions((4.0, 50.0)), "en", CFG)
    assert result is not None
    assert model.audio[0].size == 30 * SR
    assert (result.window.start_s, result.window.end_s) == (4.0, 34.0)
    assert model.options[0] == {"vad_filter": False, "language_detection_segments": 1}


def test_short_recording_uses_what_there_is():
    samples = np.zeros(SR * 10, dtype=np.float32)
    model = _Model("en", 0.9)
    result = language_check.check(model, samples, _regions((1.0, 9.0)), "en", CFG)
    assert result is not None and result.window.end_s == pytest.approx(10.0)
    assert model.audio[0].size == 9 * SR


def test_match_has_no_notice():
    result = language_check.check(
        _Model("ar", 0.97, [("en", 0.02)]), np.zeros(SR * 5), _regions((0, 5)), "ar", CFG
    )
    assert isinstance(result, LanguageCheck)
    assert (result.chosen, result.detected, result.probability) == ("ar", "ar", 0.97)
    assert result.chosen_probability == 0.97
    assert not result.mismatch and result.notice is None


@pytest.mark.parametrize(("p", "mismatch"), [(0.5, True), (0.49, False), (0.92, True)])
def test_mismatch_needs_the_threshold_probability(p: float, mismatch: bool):
    result = language_check.check(
        _Model("en", p, [("ar", 0.3)]), np.zeros(SR * 5), _regions((0, 5)), "ar", CFG
    )
    assert result is not None and result.mismatch is mismatch
    assert result.chosen_probability == 0.3
    assert (result.notice is not None) is mismatch


def test_notice_is_in_the_chosen_language_with_numbers():
    result = language_check.check(_Model("en", 0.92), np.zeros(SR * 5), _regions((0, 5)), "ar", CFG)
    assert result is not None and result.notice is not None
    assert "الإنجليزية" in result.notice and "العربية" in result.notice
    assert "92%" in result.notice and "30" in result.notice

    en = language_check.check(_Model("ar", 0.8), np.zeros(SR * 5), _regions((0, 5)), "en", CFG)
    assert en is not None and en.notice is not None
    assert en.notice.startswith("The language detected in the first 30 s of speech is Arabic")
    assert "80%" in en.notice


def test_other_language_is_named_by_its_code():
    result = language_check.check(_Model("fr", 0.7), np.zeros(SR * 5), _regions((0, 5)), "en", CFG)
    assert result is not None and result.notice is not None
    assert "another language (code fr)" in result.notice
    assert result.chosen_probability is None  # "en" not in the list the model returned


def test_no_speech_means_no_check():
    model = _Model("en", 0.99)
    assert language_check.check(model, np.zeros(SR), (), "en", CFG) is None
    assert model.audio == []


def test_threshold_comes_from_config(speech_cfg: Any):
    assert speech_cfg.language_check == DEFAULT  # configs/default.yaml, ADR 0016


@pytest.mark.parametrize(("p", "mismatch"), [(0.94, False), (0.95, True), (0.97, True)])
def test_default_threshold_ignores_the_b1_false_notices(p: float, mismatch: bool):
    # B1: the owner's English was detected as Arabic at up to 0.94; their Arabic was >= 0.97
    result = language_check.check(
        _Model("ar", p, [("en", 1 - p)]), np.zeros(SR * 5), _regions((0, 5)), "en", DEFAULT
    )
    assert result is not None and result.mismatch is mismatch


def test_mismatch_never_stops_or_changes_the_analysis(
    fake_whisper: Any, silent_wav: Path, speech_cfg: Any
):
    """Same measurements with and without a mismatch; only the notice differs."""

    def run(detected: tuple[str, float] | None) -> Any:
        model = fake_whisper()
        model.detected = detected
        return analyze.analyze_file(
            silent_wav,
            language="en",
            model=model,
            model_name="whisper_small",
            options=AsrSettings(),
            speech=speech_cfg,
            vad=lambda _: _regions((0.0, 1.2), (2.0, 3.6)),
        )

    same = run(None)
    other = run(("ar", 0.95))
    assert same.language_check is not None and not same.language_check.mismatch
    assert other.language_check is not None and other.language_check.mismatch
    assert other.transcript == same.transcript
    assert other.measurements == same.measurements
    assert other.not_measurable == same.not_measurable
