"""Pitch variation in semitones from the speaker's own median pitch (FINAL_04 §2, ADR 0013).

- Reuses the slice 3 pitch track; pYIN is never run twice.
- Frames used: voiced, inside VAD speech, outside filler stretches (lexicon hesitations and
  acoustic candidates from slice 3, +/- `filler_margin_s`).
- Each frame becomes `12 * log2(f0 / median_f0)`, so the same movement gives the same number for
  low and high voices. No absolute pitch (Hz) is ever reported.
- Octave errors: a frame more than `octave_limit_st` from the running median (+/-
  `running_median_s`) is folded by 12 semitones if that lands within `fold_tolerance_st`, and
  dropped otherwise. This can also remove a real very large jump; the folded and dropped counts
  in `PitchDiagnostics` show when that may have happened.
- Information only: no "monotone" verdict (FINAL_04 §2); `vocal_variety` is opt-in (FINAL_05).
"""

from __future__ import annotations

import numpy as np

from presenceai.schemas.assessment import Confidence, Evidence, Measurement, TimeSpan
from presenceai.schemas.speech import (
    AcousticCandidate,
    FillerEvent,
    FillerKind,
    PitchDiagnostics,
    PitchTrack,
    SpeechLanguage,
    SpeechRegion,
)
from presenceai.settings import PitchVariationSettings, SpeechSettings
from presenceai.speech.confidence import band, cap

INFO = "information only"


def _usable(
    track: PitchTrack,
    regions: tuple[SpeechRegion, ...],
    fillers: tuple[FillerEvent, ...],
    acoustic: tuple[AcousticCandidate, ...],
    margin: float,
) -> np.ndarray:
    n = len(track.voiced)
    times = np.arange(n) * track.hop_s
    mask = np.array(track.voiced, dtype=bool) & (np.array(track.f0_hz) > 0)
    in_speech = np.zeros(n, dtype=bool)
    for r in regions:
        in_speech |= (times >= r.start_s) & (times < r.end_s)
    mask &= in_speech
    spans = [(f.start_s, f.end_s) for f in fillers if f.kind == FillerKind.HESITATION]
    spans += [(c.start_s, c.end_s) for c in acoustic]
    for a, b in spans:
        mask &= ~((times >= a - margin) & (times <= b + margin))
    return mask


def clean_octaves(
    semis: np.ndarray, hop_s: float, cfg: PitchVariationSettings
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(values, kept mask, folded, dropped) for a 1-D array of semitone values (NaN = unused)."""
    half = max(1, round(cfg.running_median_s / hop_s))
    valid = ~np.isnan(semis)
    out = semis.copy()
    keep = valid.copy()
    folded = dropped = 0
    idx = np.flatnonzero(valid)
    for i in idx:
        lo, hi = max(0, i - half), min(semis.size, i + half + 1)
        window = semis[lo:hi]
        med = float(np.nanmedian(window))
        diff = semis[i] - med
        if abs(diff) <= cfg.octave_limit_st:
            continue
        shifted = semis[i] - 12.0 * np.sign(diff)
        if abs(shifted - med) <= cfg.fold_tolerance_st:
            out[i] = shifted
            folded += 1
        else:
            keep[i] = False
            dropped += 1
    return out, keep, folded, dropped


def analyse(
    track: PitchTrack,
    regions: tuple[SpeechRegion, ...],
    fillers: tuple[FillerEvent, ...],
    acoustic: tuple[AcousticCandidate, ...],
    *,
    language: SpeechLanguage,
    snr_db: float | None,
    cfg: SpeechSettings,
) -> tuple[dict[str, Measurement], dict[str, str], PitchDiagnostics | None]:
    pv = cfg.pitch_variation
    keys = ("pitch.sd_semitones", "pitch.range_semitones")
    if snr_db is not None and snr_db < cfg.loudness.snr_not_measurable_db:
        reason = f"not measurable in this recording: too noisy ({snr_db:.0f} dB signal-to-noise)"
        return {}, dict.fromkeys(keys, reason), None
    if not track.voiced or not regions:
        return {}, dict.fromkeys(keys, "no voiced speech"), None

    mask = _usable(track, regions, fillers, acoustic, pv.filler_margin_s)
    f0 = np.array(track.f0_hz, dtype=float)
    usable = int(mask.sum())
    voiced_s = usable * track.hop_s
    lim = cfg.pitch.limit_band
    at_limit = mask & ((f0 >= track.fmax_hz * (1 - lim)) | (f0 <= track.fmin_hz * (1 + lim)))
    at_limit_share = float(at_limit.sum() / usable) if usable else 0.0
    if voiced_s < pv.min_voiced_s:
        diag = PitchDiagnostics(
            usable_frames=usable, usable_voiced_s=voiced_s, octave_folded=0, dropped=0,
            at_limit_share=at_limit_share,
        )  # fmt: skip
        reason = (
            f"only {voiced_s:.1f} s of usable voiced speech; at least {pv.min_voiced_s:g} s needed"
        )
        return {}, dict.fromkeys(keys, reason), diag

    semis = np.full(f0.size, np.nan)
    semis[mask] = 12 * np.log2(f0[mask] / np.median(f0[mask]))
    values, keep, folded, dropped = clean_octaves(semis, track.hop_s, pv)
    final = values[keep & mask]
    diag = PitchDiagnostics(
        usable_frames=usable, usable_voiced_s=voiced_s, octave_folded=folded, dropped=dropped,
        at_limit_share=at_limit_share,
    )  # fmt: skip

    conf = band(cfg.confidence, probability=None, speaking_time_s=voiced_s)
    notes = ""
    if voiced_s < pv.low_voiced_s:
        conf = Confidence.LOW
        notes += f"; only {voiced_s:.0f} s of usable voiced speech"
    if snr_db is None:
        conf = cap(conf, Confidence.MEDIUM)
        notes += "; noise level unknown"
    elif snr_db < cfg.loudness.snr_low_db:
        conf = Confidence.LOW
        notes += f"; noisy recording ({snr_db:.0f} dB signal-to-noise)"
    if (folded + dropped) / usable > pv.max_corrected_share:
        conf = Confidence.LOW
        notes += f"; {folded + dropped} of {usable} frames corrected for octave errors"
    if at_limit_share > pv.max_at_limit_share:
        conf = Confidence.LOW
        notes += f"; {at_limit_share:.0%} of frames at the pitch range limit"

    span = TimeSpan(start_s=regions[0].start_s, end_s=regions[-1].end_s)
    base = (
        "semitones from your own typical pitch, voiced speech without fillers, "
        f"octave errors corrected; {INFO}"
    )

    def measure(value: float, method: str) -> Measurement:
        return Measurement(
            value=value, unit="semitones", confidence=conf, method=method + notes,
            evidence=(Evidence(time=span),), language=language, live=False,
        )  # fmt: skip

    return (
        {
            "pitch.sd_semitones": measure(float(np.std(final)), "spread (SD) of " + base),
            "pitch.range_semitones": measure(
                float(np.percentile(final, 90) - np.percentile(final, 10)),
                "range (p90 - p10) of " + base,
            ),
        },
        {},
        diag,
    )
