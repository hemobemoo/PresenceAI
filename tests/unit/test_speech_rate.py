"""speech/rate.py: words per minute, articulation rate, 10 s windows, syllable-nucleus pace."""

from typing import Any

import numpy as np
import pytest

from presenceai.schemas.assessment import Confidence
from presenceai.schemas.speech import Segment, SpeechRegion, Transcript, Word
from presenceai.speech import rate

SR = 16_000


def _transcript(times: list[float], p: float = 0.95, language: str = "en") -> Transcript:
    """One word every entry in `times` (word lasts 0.3 s)."""
    text, words = "", []
    for i, t in enumerate(times):
        token = f"w{i}"
        if text:
            text += " "
        s = len(text)
        text += token
        words.append(
            Word(
                text=token,
                start_s=t,
                end_s=t + 0.3,
                probability=p,
                start_char=s,
                end_char=len(text),
            )
        )
    segs = (
        (
            Segment(
                start_s=times[0],
                end_s=times[-1] + 0.3,
                words=tuple(words),
                avg_logprob=-0.1,
                no_speech_prob=0.0,
            ),
        )
        if words
        else ()
    )
    return Transcript(language=language, duration_s=120.0, text=text, segments=segs)  # pyright: ignore[reportArgumentType]


def _r(a: float, b: float) -> SpeechRegion:
    return SpeechRegion(start_s=a, end_s=b)


def test_wpm_and_articulation_rate(speech_cfg: Any):
    # 60 words, one every 0.5 s from 0 s to 29.5 s; speech regions cover 24 s of a 30 s span
    t = _transcript([i * 0.5 for i in range(60)])
    regions = (_r(0.0, 12.0), _r(15.0, 27.0), _r(29.5, 30.0))
    m, nm, _windows = rate.word_rate(t, regions, speech_cfg)
    assert m["rate.wpm"].value == pytest.approx(60 / 30.0 * 60)  # words / speaking time
    assert m["rate.articulation_wpm"].value == pytest.approx(60 / 24.5 * 60)  # words / speech time
    assert m["rate.wpm"].unit == "words/min"
    assert nm == {}


def test_windows_are_10s_with_5s_hop_and_count_words_by_midpoint(speech_cfg: Any):
    t = _transcript([i * 1.0 for i in range(31)])  # one word per second, 0..30 s
    regions = (_r(0.0, 30.3),)
    _, _, windows = rate.word_rate(t, regions, speech_cfg)
    assert [(w.start_s, w.end_s) for w in windows] == [
        (0.0, 10.0),
        (5.0, 15.0),
        (10.0, 20.0),
        (15.0, 25.0),
        (20.0, 30.0),
    ]
    assert {w.words for w in windows} == {10}
    assert {w.wpm for w in windows} == {60.0}


def test_below_10s_of_speech_is_not_measurable(speech_cfg: Any):
    t = _transcript([i * 0.5 for i in range(18)])
    m, nm, windows = rate.word_rate(t, (_r(0.0, 9.0),), speech_cfg)
    assert "rate.wpm" not in m and "rate.wpm" in nm and "10" in nm["rate.wpm"]
    assert windows == ()


def test_no_speech_is_not_measurable(speech_cfg: Any):
    m, nm, _ = rate.word_rate(_transcript([]), (), speech_cfg)
    assert m == {} and set(nm) >= {"rate.wpm", "rate.articulation_wpm"}


@pytest.mark.parametrize(
    ("p", "span", "expected"),
    [
        (0.95, 40.0, Confidence.HIGH),
        (0.95, 20.0, Confidence.MEDIUM),  # speaking time below 30 s
        (0.80, 40.0, Confidence.MEDIUM),  # word probability below 0.85
        (0.60, 40.0, Confidence.LOW),
    ],
)
def test_confidence_bands(speech_cfg: Any, p: float, span: float, expected: Confidence):
    t = _transcript([i * 0.5 for i in range(int(span * 2))], p=p)
    m, _, _ = rate.word_rate(t, (_r(0.0, span),), speech_cfg)
    assert m["rate.wpm"].confidence == expected


def test_rate_evidence_quotes_the_whole_transcript(speech_cfg: Any):
    t = _transcript([i * 0.5 for i in range(40)])
    m, _, _ = rate.word_rate(t, (_r(0.0, 20.0),), speech_cfg)
    (ev,) = m["rate.wpm"].evidence
    assert ev.time is not None and (ev.time.start_s, ev.time.end_s) == (0.0, 20.0)
    assert ev.transcript is not None and ev.transcript.text == t.text


# --- syllable nuclei (own implementation of the published method, ADR 0011) ------------------


def _bursts(
    n: int, burst_s: float = 0.15, gap_s: float = 0.10, gap_amp: float = 0.01
) -> np.ndarray:
    """n tone bursts (Hann envelope) separated by quiet gaps, with 0.3 s of quiet on each side."""
    rng = np.random.default_rng(0)
    parts = [gap_amp * rng.standard_normal(int(0.3 * SR))]
    tb = np.arange(int(burst_s * SR)) / SR
    tone = 0.5 * np.sin(2 * np.pi * 180 * tb) * np.hanning(tb.size)
    for _ in range(n):
        parts.append(tone)
        parts.append(gap_amp * rng.standard_normal(int(gap_s * SR)))
    parts.append(gap_amp * rng.standard_normal(int(0.3 * SR)))
    return np.concatenate(parts).astype(np.float32)


def test_one_nucleus_per_burst(speech_cfg: Any):
    x = _bursts(8)
    region = (_r(0.0, x.size / SR),)
    nuclei = rate.syllable_nuclei(x, SR, region, speech_cfg.syllables)
    assert len(nuclei) == 8
    assert np.all(np.diff(nuclei) > 0.2)  # one per burst, 0.25 s apart


def test_shallow_dip_does_not_split_a_nucleus(speech_cfg: Any):
    """A steady tone whose loudness wobbles by < 1 dB has one nucleus, not three."""
    t = np.arange(int(0.6 * SR)) / SR
    env = 1 - 0.08 * (1 - np.cos(2 * np.pi * t / 0.3)) / 2  # dips of 20*log10(0.92) = -0.7 dB
    tone = 0.5 * np.sin(2 * np.pi * 180 * t) * env
    pad = np.zeros(int(0.3 * SR))
    x = np.concatenate([pad, tone, pad]).astype(np.float32)
    nuclei = rate.syllable_nuclei(x, SR, (_r(0.0, x.size / SR),), speech_cfg.syllables)
    assert len(nuclei) == 1


def test_peaks_outside_speech_regions_are_ignored(speech_cfg: Any):
    x = _bursts(8)
    half = (_r(0.0, 0.3 + 4 * 0.25),)  # covers the first 4 bursts only
    assert len(rate.syllable_nuclei(x, SR, half, speech_cfg.syllables)) == 4


def test_quiet_bursts_below_the_silence_threshold_are_ignored(speech_cfg: Any):
    loud, quiet = _bursts(4), _bursts(4) * 0.02  # quiet copy is ~34 dB lower
    x = np.concatenate([loud, quiet])
    nuclei = rate.syllable_nuclei(x, SR, (_r(0.0, x.size / SR),), speech_cfg.syllables)
    assert len(nuclei) == 4


def test_syllable_rate_measure(speech_cfg: Any):
    x = _bursts(40)  # 10.6 s: just above the 10 s minimum
    span = x.size / SR
    m, nm = rate.syllable_rate(x, SR, (_r(0.0, span),), "en", speech_cfg)
    assert nm == {}
    assert m["pace.syllables_per_s"].value == pytest.approx(40 / span)
    assert m["pace.syllables_per_s"].unit == "syllables/s"
    assert m["pace.syllables_per_s"].confidence == Confidence.MEDIUM  # 10–30 s of speech


def test_syllable_rate_is_information_only_for_arabic(speech_cfg: Any):
    x = np.concatenate([_bursts(40)] * 2)
    span = x.size / SR
    m, _ = rate.syllable_rate(x, SR, (_r(0.0, span),), "ar", speech_cfg)
    meas = m["pace.syllables_per_s"]
    assert meas.confidence == Confidence.LOW and "information only" in meas.method
    assert meas.language == "ar"


def test_syllable_rate_not_measurable_when_short(speech_cfg: Any):
    x = _bursts(4)
    m, nm = rate.syllable_rate(x, SR, (_r(0.0, x.size / SR),), "en", speech_cfg)
    assert m == {} and "pace.syllables_per_s" in nm
