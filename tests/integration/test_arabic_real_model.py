"""Slow real-model Arabic run (ADR 0016 open item 5): whisper_small, int8 on CPU, on one of the
owner's Arabic B1 clips, once.

The clip is `b1_ar_04_normal` (138.8 s; owner, 2026-11-21): the shortest Arabic clip with «يعني»
in the owner's corrected transcript. The shortest clip of all, `b1_ar_08_fast` (135.2 s), has
none, so the marker check could not pass on it.

Runs only on the owner's laptop: it skips when the clip is not in `data/b1/audio/` (always the
case in CI, since `data/` is git-ignored) or when whisper_small is not downloaded. It asserts
behaviour, not exact WER or word counts. No transcript text is printed, logged or written:
the asserts compare booleans and counts only, so a failure shows no transcript words.
"""

import re
from typing import Any

import pytest

from presenceai.evalconfig import REPO_ROOT
from presenceai.schemas.speech import FillerKind, SpeechReport
from presenceai.settings import Settings
from presenceai.speech import analyze, asr
from presenceai.text.normalize import normalize_token

pytestmark = pytest.mark.slow

CLIP = REPO_ROOT / "data" / "b1" / "audio" / "b1_ar_04_normal.wav"
ARABIC = re.compile(r"[؀-ۿ]")
MARKER = normalize_token("يعني", "ar")


@pytest.fixture(scope="module")
def report() -> SpeechReport:
    if not CLIP.is_file():
        pytest.skip("B1 clip b1_ar_04_normal not in data/b1/audio/ (owner's laptop only)")
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    options = settings.asr.model_copy(update={"device": "cpu", "compute_type": "int8"})
    try:
        model = asr.load_model("whisper_small", options)
    except asr.ModelNotAvailable:
        pytest.skip("whisper_small is not downloaded (scripts/download_models.py)")
    return analyze.analyze_file(  # one run for every test in this module
        CLIP,
        language="ar",
        model=model,
        model_name="whisper_small",
        options=options,
        speech=settings.speech,
    )


def test_no_crash_and_a_valid_report(report: Any):
    SpeechReport.model_validate_json(report.model_dump_json())
    assert report.transcript.language == "ar"


def test_detected_language_is_arabic(report: Any):
    check = report.language_check
    assert check is not None
    assert check.detected == "ar" and not check.mismatch


def test_transcript_is_arabic_and_not_empty(report: Any):
    n_words = len(report.transcript.words)
    has_arabic = ARABIC.search(report.transcript.text) is not None
    assert n_words > 0, "empty transcript"
    assert has_arabic, "no Arabic letters in the transcript"


def test_yani_marker_is_found(report: Any):
    words = report.transcript.words
    found = sum(
        1
        for e in report.fillers
        if e.kind == FillerKind.DISCOURSE_MARKER
        and any(
            normalize_token(w.text, "ar") == MARKER for w in words[e.first_word : e.last_word + 1]
        )
    )
    assert found >= 1, "no «يعني» discourse marker found"
    assert report.measurements["discourse_markers.count"].value >= found
