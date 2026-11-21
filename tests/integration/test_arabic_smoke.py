"""Arabic smoke test: one fake Arabic recording through the whole `analyze` path (P1 slice 6).

No models: the Whisper output, the VAD regions and the pitch track are fakes; the audio is a
synthetic 40 s file. The transcript has a hesitation, «يعني» markers, a doubled «يعني يعني», an
English term ("slides", code-switching) and almost no punctuation, as base Whisper often gives
for Arabic. The slow real-model run is `test_arabic_real_model.py` (owner's laptop only).
"""

import re
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from presenceai.coach.guardrails import contains_banned_term, contains_verdict_term
from presenceai.lang import report_texts, text
from presenceai.schemas.assessment import Confidence
from presenceai.schemas.speech import PausePosition, PitchTrack, SpeechRegion
from presenceai.settings import AsrSettings
from presenceai.speech import analyze

SR = 16_000
DURATION_S = 40.0
REGIONS = ((0.5, 12.0), (13.5, 25.0), (26.2, 39.5))
ARABIC = re.compile(r"[؀-ۿ]")
# one line of a talk, repeated: hesitation, markers, doubled marker, English term, one full stop
TALK = [
    *("امم", "سنتحدث", "اليوم", "عن", "المشروع", "يعني", "يعني", "الفكرة", "الأساسية"),
    *("بسيطة", "وسأعرض", "النتائج", "في", "slides", "قصيرة", "طيب", "ننتقل", "الآن"),
    *("إلى", "الجزء", "الثاني", "من", "العرض"),
]


def _wav(path: Path) -> Path:
    t = np.arange(int(DURATION_S * SR)) / SR
    rng = np.random.default_rng(20261118)
    x = 1e-3 * rng.standard_normal(t.size)
    for a, b in REGIONS:  # voiced-like tone bursts inside speech, louder and quieter
        on = (t >= a) & (t < b) & ((t * 2) % 1 < 0.6)
        x[on] += (0.2 + 0.1 * np.sin(2 * np.pi * 0.1 * t[on])) * np.sin(2 * np.pi * 180 * t[on])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _segments(whisper_parts: Any) -> list[Any]:
    words, t = [], 0.0
    slots = [s for a, b in REGIONS for s in np.arange(a, b - 0.4, 0.5)]
    for i, start in enumerate(slots):
        token = TALK[i % len(TALK)]
        if i == len(slots) - 1:
            token += "."
        words.append(whisper_parts.Word(" " + token, float(start), float(start) + 0.4, 0.8))
        t = float(start) + 0.4
    return [whisper_parts.Segment(0.5, t, "", words)]


def _pitch(samples: Any, sr: int, cfg: Any) -> PitchTrack:
    n = int(DURATION_S / 0.01)
    times = np.arange(n) * 0.01
    voiced = tuple(bool(any(a <= x < b for a, b in REGIONS)) for x in times)
    f0 = tuple(
        150.0 * 2 ** (np.sin(x) / 12) if v else 0.0 for x, v in zip(times, voiced, strict=True)
    )
    return PitchTrack(hop_s=0.01, fmin_hz=65.0, fmax_hz=600.0, f0_hz=f0, voiced=voiced)


@pytest.fixture
def report(tmp_path: Path, fake_whisper: Any, whisper_parts: Any, speech_cfg: Any) -> Any:
    return analyze.analyze_file(
        _wav(tmp_path / "talk_ar.wav"),
        language="ar",
        model=fake_whisper(_segments(whisper_parts), language="ar", duration=DURATION_S),
        model_name="whisper_small",
        options=AsrSettings(),
        speech=speech_cfg,
        vad=lambda _: tuple(SpeechRegion(start_s=a, end_s=b) for a, b in REGIONS),
        pitch_tracker=_pitch,
    )


def test_core_measures_exist(report: Any):
    for key in ("rate.wpm", "fillers.count", "repetitions.count", "pace.syllables_per_s"):
        assert key in report.measurements, report.not_measurable
    assert all(m.language == "ar" for m in report.measurements.values())


def test_fillers_markers_and_repetitions(report: Any):
    m = report.measurements
    assert m["fillers.count"].value >= 1
    assert m["fillers.count"].confidence == Confidence.LOW  # base model, Arabic cap (ADR 0012)
    assert m["discourse_markers.count"].value >= 1  # «يعني», «طيب»: information only
    assert m["discourse_markers.count"].confidence == Confidence.LOW
    assert m["repetitions.count"].value >= 1  # «يعني يعني»


def test_pause_positions_are_unknown_with_sparse_punctuation(report: Any):
    assert report.pauses, "the three regions leave two pauses"
    assert {p.position for p in report.pauses} == {PausePosition.UNKNOWN}
    assert report.measurements["pauses.count.mid_sentence"].value == 0


def test_syllable_pace_is_information_only(report: Any):
    pace = report.measurements["pace.syllables_per_s"]
    assert pace.confidence == Confidence.LOW
    assert pace.method.endswith(text("ar", "rate.syllables_info_only"))


def test_language_check_matches(report: Any):
    assert report.language_check is not None
    assert report.language_check.chosen == "ar" and not report.language_check.mismatch


def test_every_report_text_is_arabic_and_clean(report: Any):
    english = {v for k, v in report_texts("en").items() if len(v.split()) >= 2}
    texts = [m.method for m in report.measurements.values()] + list(report.not_measurable.values())
    assert texts
    for value in texts:
        assert ARABIC.search(value), value
        assert not any(e in value for e in english), value
        assert not contains_verdict_term(value, "ar") and not contains_verdict_term(value, "en")
        assert not contains_banned_term(value, "ar") and not contains_banned_term(value, "en")


def test_english_term_is_kept_in_the_transcript(report: Any):
    assert "slides" in report.transcript.text  # quotes stay exactly as spoken (ADR 0003)
