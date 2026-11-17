"""No verdict words ("monotone", "رتيب", ...) in speech report text, English or Arabic (ADR 0013).

FINAL_04 §2: pitch variation is information only, with no "monotone" verdict unless the user
chose that goal. The lists live in lang/<lang>/verdict_terms.yaml and are matched as whole words
before Arabic letter normalisation (diacritics removed, attached prefixes allowed).
"""

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
import yaml

from presenceai.coach.guardrails import contains_verdict_term, verdict_terms
from presenceai.lang import LANGUAGES, Language
from presenceai.schemas.speech import PitchTrack, SpeechRegion
from presenceai.settings import AsrSettings
from presenceai.speech import analyze


@pytest.mark.parametrize("lang", LANGUAGES)
def test_lists_exist(lang: Language):
    assert len(verdict_terms(lang)) >= 5


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Your delivery was monotone.", "en"),
        ("A flat, robotic voice", "en"),
        ("Rather dull pitch", "en"),
        ("كان صوتك رتيبا", "ar"),
        ("صوتٌ رَتِيبٌ", "ar"),  # diacritics
        ("بصوت رتيب", "ar"),  # attached preposition
        ("الإلقاء ممل", "ar"),
        ("على وتيرة واحدة", "ar"),
        ("صوت آلي", "ar"),
        ("بصوتٍ آليٍّ", "ar"),  # diacritics
        ("الإلقاء الآلي", "ar"),  # article
        ("تحدّث رتيبًا", "ar"),  # tanween on the alef
    ],
)
def test_positives(text: str, lang: Language):
    assert contains_verdict_term(text, lang)


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Pitch varied by 2.8 semitones (information only).", "en"),
        ("A flat-rate plan", "en"),  # 'flat-rate' is one token
        ("تغيّر الصوت بمقدار 2.8 نصف نغمة", "ar"),
        ("انتقل إلى السؤال التالي", "ar"),  # إلى is not آلي
        ("إلى", "ar"),
        ("الى اللقاء", "ar"),  # إلى written without hamza
        ("آليات التقييم", "ar"),  # "mechanisms": whole words only
        ("آلية العمل", "ar"),
    ],
)
def test_negatives(text: str, lang: Language):
    assert not contains_verdict_term(text, lang)


def _strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in [*_strings(k), *_strings(v)]]
    if isinstance(obj, list | tuple):
        return [s for v in obj for s in _strings(v)]
    return []


SKIP = {"banned_terms.yaml", "verdict_terms.yaml"}  # the lists themselves


@pytest.mark.parametrize("lang", LANGUAGES)
def test_language_pack_text_has_no_verdicts(lang: Language):
    folder = files("presenceai.lang").joinpath(lang)
    for item in folder.iterdir():
        if not item.name.endswith(".yaml") or item.name in SKIP:
            continue
        for text in _strings(yaml.safe_load(item.read_text(encoding="utf-8"))):
            assert not contains_verdict_term(text, lang), f"{lang}/{item.name}: {text!r}"


@pytest.mark.parametrize("lang", LANGUAGES)
def test_speech_report_text_has_no_verdicts(
    lang: Language, fake_whisper: Any, silent_wav: Path, speech_cfg: Any
):
    """Every string in a full report (methods, reasons), in both languages, including a
    flat-pitch recording where a verdict would be most tempting."""
    n = 400

    def flat_pitch(samples: Any, sr: int, cfg: Any) -> PitchTrack:
        return PitchTrack(
            hop_s=0.01, fmin_hz=65.0, fmax_hz=600.0, f0_hz=(150.0,) * n, voiced=(True,) * n
        )

    report = analyze.analyze_file(
        silent_wav,
        language=lang,
        model=fake_whisper(language=lang),
        model_name="whisper_small",
        options=AsrSettings(),
        speech=speech_cfg,
        vad=lambda x: (SpeechRegion(start_s=0.0, end_s=3.6),),
        pitch_tracker=flat_pitch,
    )
    data = json.loads(report.model_dump_json())
    data.pop("transcript")  # the speaker's own words are quoted, not ours
    for text in _strings(data):
        for check in LANGUAGES:  # an Arabic report must not carry English verdicts either
            assert not contains_verdict_term(text, check), text
