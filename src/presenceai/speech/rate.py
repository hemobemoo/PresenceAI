"""Speech rate and pace (FINAL_04 §2, ADR 0011).

- `rate.wpm`: words / speaking time, where speaking time runs from the first VAD speech onset to
  the last offset (pauses included), plus 10 s windows with a 5 s hop.
- `rate.articulation_wpm`: words / time inside VAD speech regions (pauses excluded).
- `pace.syllables_per_s`: syllable nuclei / time inside speech regions, from our own
  implementation of the method published by de Jong & Wempe (2009, Behavior Research Methods
  41(2), 385-390): intensity peaks that rise at least `min_dip_db` above the preceding dip and lie
  above (0.99-quantile intensity + `silence_threshold_db`), using the values of the authors' 2010
  v2 script. Their Praat scripts are GPL-3; no code is taken from them. VAD speech regions stand
  in for the original's pitch-based voicing check. The values were validated on Dutch speech, so
  for Arabic the pace is information only (confidence low).

Below `min_speaking_time_s` a measure is "not measurable", never a low score.
"""

from __future__ import annotations

import numpy as np

from presenceai.schemas.assessment import Confidence, Evidence, Measurement, TimeSpan
from presenceai.schemas.speech import RateWindow, SpeechLanguage, SpeechRegion, Transcript
from presenceai.settings import SpeechSettings, SyllableSettings
from presenceai.speech.confidence import band


def _speech_time(regions: tuple[SpeechRegion, ...]) -> float:
    return sum(r.duration_s for r in regions)


def word_rate(
    transcript: Transcript, regions: tuple[SpeechRegion, ...], cfg: SpeechSettings
) -> tuple[dict[str, Measurement], dict[str, str], tuple[RateWindow, ...]]:
    keys = ("rate.wpm", "rate.articulation_wpm")
    if not regions:
        return {}, dict.fromkeys(keys, "no speech detected"), ()
    first, last = regions[0].start_s, regions[-1].end_s
    speaking = last - first
    minimum = cfg.rate.min_speaking_time_s
    if speaking < minimum:
        reason = f"only {speaking:.1f} s of speech; at least {minimum:g} s needed"
        return {}, dict.fromkeys(keys, reason), ()

    words = transcript.words
    n = len(words)
    mean_p = sum(w.probability for w in words) / n if n else None
    conf = band(cfg.confidence, probability=mean_p, speaking_time_s=speaking)
    quote = transcript.span(0, n) if n else None
    evidence = (Evidence(transcript=quote, time=TimeSpan(start_s=first, end_s=last)),)

    def measure(value: float, method: str) -> Measurement:
        return Measurement(
            value=value,
            unit="words/min",
            confidence=conf,
            method=method,
            evidence=evidence,
            language=transcript.language,
            live=False,
        )

    out = {
        "rate.wpm": measure(n / speaking * 60, "words / speaking time (first to last speech)"),
        "rate.articulation_wpm": measure(
            n / _speech_time(regions) * 60, "words / time inside VAD speech regions"
        ),
    }
    mids = np.array([(w.start_s + w.end_s) / 2 for w in words])
    windows: list[RateWindow] = []
    w, hop = cfg.rate.window_s, cfg.rate.hop_s
    start = first
    while start + w <= last + 1e-9:
        count = int(np.count_nonzero((mids >= start) & (mids < start + w))) if n else 0
        windows.append(RateWindow(start_s=start, end_s=start + w, words=count, wpm=count / w * 60))
        start += hop
    return out, {}, tuple(windows)


def _intensity_db(
    samples: np.ndarray, sr: int, cfg: SyllableSettings
) -> tuple[np.ndarray, np.ndarray]:
    win = max(1, int(sr * cfg.window_ms / 1000))
    hop = max(1, int(sr * cfg.hop_ms / 1000))
    if samples.size < win:
        return np.zeros(0), np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(samples.astype(np.float64), win)[::hop]
    weights = np.hanning(win)
    power = (frames**2 * weights).sum(axis=1) / weights.sum()
    db = 10 * np.log10(power + 1e-12)
    times = (np.arange(db.size) * hop + win / 2) / sr
    return times, db


def syllable_nuclei(
    samples: np.ndarray, sr: int, regions: tuple[SpeechRegion, ...], cfg: SyllableSettings
) -> np.ndarray:
    """Times (s) of syllable nuclei inside the speech regions."""
    times, db = _intensity_db(samples, sr, cfg)
    if db.size < 3 or not regions:
        return np.zeros(0)
    threshold = float(np.quantile(db, cfg.reference_quantile)) + cfg.silence_threshold_db
    is_peak = np.zeros(db.size, dtype=bool)
    is_peak[1:-1] = (db[1:-1] > db[:-2]) & (db[1:-1] >= db[2:]) & (db[1:-1] > threshold)
    nuclei: list[float] = []
    for r in regions:
        idx = np.flatnonzero((times >= r.start_s) & (times <= r.end_s))
        if idx.size == 0:
            continue
        ref = int(idx[0])  # the preceding dip is searched from here
        for i in idx[is_peak[idx]]:
            dip = float(db[ref : i + 1].min())
            if db[i] - dip >= cfg.min_dip_db:
                nuclei.append(float(times[i]))
                ref = int(i)
    return np.array(nuclei)


def syllable_rate(
    samples: np.ndarray,
    sr: int,
    regions: tuple[SpeechRegion, ...],
    language: SpeechLanguage,
    cfg: SpeechSettings,
) -> tuple[dict[str, Measurement], dict[str, str]]:
    key = "pace.syllables_per_s"
    speech = _speech_time(regions)
    minimum = cfg.rate.min_speaking_time_s
    if speech < minimum:
        return {}, {key: f"only {speech:.1f} s of speech; at least {minimum:g} s needed"}
    nuclei = syllable_nuclei(samples, sr, regions, cfg.syllables)
    method = "syllable nuclei (intensity peaks, de Jong & Wempe 2009) / speech time"
    if language == "ar":
        conf = Confidence.LOW
        method += "; information only (values validated on Dutch speech)"
    else:
        conf = band(cfg.confidence, probability=None, speaking_time_s=speech)
    evidence = (Evidence(time=TimeSpan(start_s=regions[0].start_s, end_s=regions[-1].end_s)),)
    m = Measurement(
        value=len(nuclei) / speech,
        unit="syllables/s",
        confidence=conf,
        method=method,
        evidence=evidence,
        language=language,
        live=False,
    )
    return {key: m}, {}
