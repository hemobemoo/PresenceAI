"""Loudness, relative only (FINAL_04 §2, ADR 0013).

- **One level function**, `frame_dbfs`, gives the level of a 20 ms frame. The live meter (P6)
  calls it on each 20 ms chunk. The report's 1 s windows are power means of the same frames
  (`power_mean_db`), so live and post numbers agree exactly.
- Measures: variation across 1 s speech windows (coefficient of variation of the RMS amplitude,
  FINAL_04), the spread between loud and quiet speech (p90 - p10 dB, information only), clipping
  share, and the change from the user's own baseline report. **An absolute level never becomes
  a measurement**; it is kept only in `LoudnessProfile` for baseline comparisons.
- Only windows that are mostly VAD speech count, so pauses are not "quiet".
- Clipping is detected on the original samples before resampling (`audio.load_native`).
  Samples at full scale are clipping; a plateau just under the peak (lossy files) is reported
  apart as "possible clipping", always low confidence.
- Noise: the floor is a low percentile of non-speech frames. Below `snr_not_measurable_db` the
  measures are "not measurable in this recording"; below `snr_low_db` confidence is low.
- Browser processing (auto-gain, noise suppression, echo cancellation) flattens loudness. When
  any of it was on, or is unknown (every uploaded file), variation, spread and baseline change
  are capped at low confidence.
"""

from __future__ import annotations

import numpy as np

from presenceai.lang import text
from presenceai.schemas.assessment import Confidence, Evidence, Measurement, TimeSpan
from presenceai.schemas.speech import (
    CaptureProcessing,
    LoudnessProfile,
    SpeechLanguage,
    SpeechRegion,
)
from presenceai.settings import LoudnessSettings, SpeechSettings
from presenceai.speech.confidence import band, cap

_EPS = 1e-12


def frame_dbfs(frame: np.ndarray) -> float:
    """Level of one frame in dB relative to full scale (a full-scale sine is -3.01 dBFS)."""
    x = np.asarray(frame, dtype=np.float64)
    return float(10 * np.log10(np.mean(x * x) + _EPS)) if x.size else float(10 * np.log10(_EPS))


def power_mean_db(levels_db: list[float] | np.ndarray) -> float:
    """Combine frame levels into one level (mean power), exactly as the report does."""
    db = np.asarray(levels_db, dtype=np.float64)
    return float(10 * np.log10(np.mean(10 ** (db / 10)))) if db.size else float(10 * np.log10(_EPS))


def frame_levels(samples: np.ndarray, sr: int, cfg: LoudnessSettings) -> np.ndarray:
    """`frame_dbfs` of consecutive, non-overlapping frames (vectorised, same numbers)."""
    n = int(sr * cfg.frame_ms / 1000)
    count = samples.size // n
    if count == 0:
        return np.zeros(0)
    frames = samples[: count * n].astype(np.float64).reshape(count, n)
    return 10 * np.log10(np.mean(frames * frames, axis=1) + _EPS)


def window_levels(samples: np.ndarray, sr: int, cfg: LoudnessSettings) -> np.ndarray:
    frames = frame_levels(samples, sr, cfg)
    per = round(cfg.window_s * 1000 / cfg.frame_ms)
    count = frames.size // per
    return np.array([power_mean_db(frames[i * per : (i + 1) * per]) for i in range(count)])


def _hard_mask(native: np.ndarray, cfg: LoudnessSettings) -> np.ndarray:
    """Rule 1, clipping: |x| >= `clip_level` in any channel (lossless files, and lossy overshoot
    above full scale)."""
    return np.any(np.abs(native) >= cfg.clip_level, axis=0)


def pileup_ratio(channel: np.ndarray, cfg: LoudnessSettings) -> float | None:
    """Samples within `clip_pileup_band` of the channel's peak divided by samples in the band
    just below; None when the peak is under `clip_peak_min` (rule 2 does not apply)."""
    a = np.abs(channel)
    peak = float(a.max()) if a.size else 0.0
    if peak < cfg.clip_peak_min:
        return None
    top = a >= peak * (1 - cfg.clip_pileup_band)
    below = (a >= peak * (1 - 2 * cfg.clip_pileup_band)) & ~top
    return float(top.sum() / max(1, int(below.sum())))


def _plateau_mask(native: np.ndarray, cfg: LoudnessSettings) -> np.ndarray:
    """Rule 2, possible clipping: a plateau just under the peak, as a lossy codec leaves it
    (~0.97). Samples that rule 1 already counts are left out."""
    mask = np.zeros(native.shape[-1], dtype=bool)
    for ch in np.atleast_2d(native):
        r = pileup_ratio(ch, cfg)
        if r is not None and r >= cfg.clip_pileup_ratio:
            mask |= np.abs(ch) >= float(np.abs(ch).max()) * (1 - cfg.clip_pileup_band)
    return mask & ~_hard_mask(native, cfg)


def _runs(mask: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [
        (int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a >= min_run
    ]


def _count(mask: np.ndarray, sr: int, cfg: LoudnessSettings) -> tuple[int, float]:
    runs = _runs(mask, cfg.clip_min_run)
    return len(runs), float(sum(b - a for a, b in runs) / sr)


def clipping(native: np.ndarray, sr: int, cfg: LoudnessSettings) -> tuple[int, float]:
    """(runs, seconds) of clipping (rule 1) in the original samples; runs shorter than
    `clip_min_run` samples are ignored."""
    if native.size == 0:
        return 0, 0.0
    return _count(_hard_mask(np.atleast_2d(native), cfg), sr, cfg)


def possible_clipping(native: np.ndarray, sr: int, cfg: LoudnessSettings) -> tuple[int, float]:
    """(runs, seconds) of possible clipping (rule 2, plateau), not counting rule-1 samples."""
    if native.size == 0:
        return 0, 0.0
    return _count(_plateau_mask(np.atleast_2d(native), cfg), sr, cfg)


def _spans(mask: np.ndarray, sr: int, cfg: LoudnessSettings) -> list[TimeSpan]:
    return [TimeSpan(start_s=a / sr, end_s=b / sr) for a, b in _runs(mask, cfg.clip_min_run)]


def _speech_mask(times: np.ndarray, regions: tuple[SpeechRegion, ...]) -> np.ndarray:
    mask = np.zeros(times.size, dtype=bool)
    for r in regions:
        mask |= (times >= r.start_s) & (times < r.end_s)
    return mask


def summarise(
    samples: np.ndarray,
    sr: int,
    native: tuple[np.ndarray, int],
    regions: tuple[SpeechRegion, ...],
    *,
    capture: CaptureProcessing,
    baseline: tuple[LoudnessProfile, CaptureProcessing] | None,
    language: SpeechLanguage,
    cfg: SpeechSettings,
) -> tuple[dict[str, Measurement], dict[str, str], LoudnessProfile]:
    lc = cfg.loudness
    keys = [
        "loudness.variation_cv",
        "loudness.spread_db",
        "loudness.clipping_share",
        "loudness.possible_clipping_share",
    ]
    if baseline is not None:
        keys.append("loudness.change_from_baseline_db")
    native_samples, native_sr = native
    native_2d = np.atleast_2d(native_samples)
    runs, clipped_s = clipping(native_2d, native_sr, lc)
    possible_runs, possible_s = possible_clipping(native_2d, native_sr, lc)

    frames = frame_levels(samples, sr, lc)
    frame_s = lc.frame_ms / 1000
    times = (np.arange(frames.size) + 0.5) * frame_s
    speech = _speech_mask(times, regions)
    non_speech_s = float((~speech).sum() * frame_s)
    median_speech = float(np.median(frames[speech])) if speech.any() else None
    noise = (
        float(np.percentile(frames[~speech], lc.noise_percentile))
        if non_speech_s >= lc.min_non_speech_s
        else None
    )
    snr = median_speech - noise if median_speech is not None and noise is not None else None
    profile = LoudnessProfile(
        median_speech_dbfs=median_speech,
        noise_floor_dbfs=noise,
        snr_db=snr,
        native_sample_rate=native_sr,
        clipped_runs=runs,
        possible_clipped_runs=possible_runs,
    )
    if not regions or median_speech is None:
        return {}, dict.fromkeys(keys, text(language, "common.no_speech")), profile
    if snr is not None and snr < lc.snr_not_measurable_db:
        reason = text(language, "common.too_noisy", snr=snr)
        return {}, dict.fromkeys(keys, reason), profile

    # 1 s windows that are mostly speech
    per = round(lc.window_s / frame_s)
    n_win = frames.size // per
    win_db, win_spans = [], []
    for i in range(n_win):
        sl = slice(i * per, (i + 1) * per)
        if speech[sl].mean() >= lc.min_speech_share:
            win_db.append(power_mean_db(frames[sl]))
            win_spans.append(TimeSpan(start_s=i * lc.window_s, end_s=(i + 1) * lc.window_s))
    speech_s = len(win_db) * lc.window_s
    span = TimeSpan(start_s=regions[0].start_s, end_s=regions[-1].end_s)

    conf = band(cfg.confidence, probability=None, speaking_time_s=speech_s)
    notes = ""
    if snr is None:
        conf = cap(conf, Confidence.MEDIUM)
        notes += text(language, "loudness.noise_unknown")
    elif snr < lc.snr_low_db:
        conf = Confidence.LOW
        notes += text(language, "common.noisy_note", snr=snr)
    proc_conf, proc_note = conf, ""
    if not capture.all_off:
        proc_conf = Confidence.LOW
        proc_note = text(language, "loudness.processing")

    def measure(
        value: float, unit: str, c: Confidence, method: str, ev: tuple[Evidence, ...]
    ) -> Measurement:
        return Measurement(
            value=value, unit=unit, confidence=c, method=method, evidence=ev,
            language=language, live=False,
        )  # fmt: skip

    out: dict[str, Measurement] = {}
    missing: dict[str, str] = {}
    if speech_s < cfg.rate.min_speaking_time_s:
        reason = text(
            language,
            "common.too_little_speech_whole",
            speech_s=speech_s,
            minimum=cfg.rate.min_speaking_time_s,
        )
        missing |= {"loudness.variation_cv": reason, "loudness.spread_db": reason}
    else:
        amp = 10 ** (np.array(win_db) / 20)
        cv = float(amp.std() / amp.mean())
        spread = float(np.percentile(win_db, 90) - np.percentile(win_db, 10))
        ev = (Evidence(time=span),)
        out["loudness.variation_cv"] = measure(
            cv, "ratio", proc_conf,
            text(language, "loudness.variation_cv") + notes + proc_note,
            ev,
        )  # fmt: skip
        out["loudness.spread_db"] = measure(
            spread, "dB", proc_conf,
            text(language, "loudness.spread_db") + notes + proc_note,
            ev,
        )  # fmt: skip

    total_speech = sum(r.duration_s for r in regions) or speech_s

    def share(seconds: float) -> float:
        return min(1.0, seconds / total_speech) if total_speech else 0.0

    def clip_evidence(mask: np.ndarray) -> tuple[Evidence, ...]:
        if native_2d.size == 0:
            return (Evidence(time=span),)
        ev = tuple(Evidence(time=s) for s in _spans(mask, native_sr, lc)[:20])
        return ev or (Evidence(time=span),)

    out["loudness.clipping_share"] = measure(
        share(clipped_s),
        "share of speech time",
        conf,
        text(language, "loudness.clipping", level=lc.clip_level, run=lc.clip_min_run) + notes,
        clip_evidence(_hard_mask(native_2d, lc)),
    )
    # Rule 2 is reported apart and is always low confidence (owner, 2026-11-17): a plateau just
    # under the peak is typical of a clipped file after lossy compression, but its values are
    # provisional and come from synthetic signals.
    out["loudness.possible_clipping_share"] = measure(
        share(possible_s),
        "share of speech time",
        Confidence.LOW,
        text(
            language,
            "loudness.possible_clipping",
            band=lc.clip_pileup_band,
            ratio=lc.clip_pileup_ratio,
        )
        + notes,
        clip_evidence(_plateau_mask(native_2d, lc)),
    )

    if baseline is not None:
        base, base_capture = baseline
        if base.median_speech_dbfs is None:
            missing["loudness.change_from_baseline_db"] = text(
                language, "loudness.baseline_no_level"
            )
        else:
            c = cap(conf, Confidence.MEDIUM)  # a different microphone or distance is invisible
            note = text(language, "loudness.baseline_device_note")
            if not (capture.all_off and base_capture.all_off):
                c = Confidence.LOW
                note += proc_note or text(language, "loudness.baseline_processing")
            out["loudness.change_from_baseline_db"] = measure(
                median_speech - base.median_speech_dbfs, "dB", c,
                text(language, "loudness.baseline") + notes + note,
                (Evidence(time=span),),
            )  # fmt: skip
    return out, missing, profile
