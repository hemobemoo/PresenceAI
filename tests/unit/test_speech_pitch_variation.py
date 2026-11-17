"""speech/pitch_variation.py: semitones from the speaker's own pitch, information only (ADR 0013).

Fast tests build pitch tracks directly; the two tests that run real pYIN are marked `slow`."""

from typing import Any

import numpy as np
import pytest

from presenceai.schemas.assessment import Confidence
from presenceai.schemas.speech import AcousticCandidate, FillerEvent, FillerKind, PitchTrack
from presenceai.speech import pitch, pitch_variation

HOP = 0.01


def vibrato(base_hz: float, depth_st: float, seconds: float, rate_hz: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds / HOP)) * HOP
    return base_hz * 2 ** (depth_st * np.sin(2 * np.pi * rate_hz * t) / 12)


def track(f0: np.ndarray, voiced: np.ndarray | None = None, fmax: float = 600.0) -> PitchTrack:
    v = np.ones(f0.size, bool) if voiced is None else voiced
    return PitchTrack(
        hop_s=HOP,
        fmin_hz=65.0,
        fmax_hz=fmax,
        f0_hz=tuple(float(x) if ok else 0.0 for x, ok in zip(f0, v, strict=True)),
        voiced=tuple(bool(x) for x in v),
    )


def run(
    sb: Any,
    cfg: Any,
    tr: PitchTrack,
    *,
    fillers: Any = (),
    acoustic: Any = (),
    snr: Any = 40.0,
    regions: Any = None,
) -> Any:
    regions = regions or (sb.region(0.0, len(tr.voiced) * HOP),)
    return pitch_variation.analyse(
        tr, regions, fillers, acoustic, language="en", snr_db=snr, cfg=cfg
    )


def test_vibrato_depth_gives_the_expected_spread(sb: Any, speech_cfg: Any):
    m, nm, diag = run(sb, speech_cfg, track(vibrato(150.0, 4.0, 40.0)))
    assert nm == {}
    # sinusoid of amplitude A semitones has SD A / sqrt(2)
    assert m["pitch.sd_semitones"].value == pytest.approx(4.0 / np.sqrt(2), rel=0.03)
    assert m["pitch.range_semitones"].value > 0
    assert diag.octave_folded == 0 and diag.dropped == 0


def test_same_movement_same_result_for_low_and_high_voices(sb: Any, speech_cfg: Any):
    """Fairness: semitones from the speaker's own median, not Hz."""
    low, _, _ = run(sb, speech_cfg, track(vibrato(100.0, 5.0, 40.0)))
    high, _, _ = run(sb, speech_cfg, track(vibrato(320.0, 5.0, 40.0)))  # peaks ~427 Hz > 400
    assert low["pitch.sd_semitones"].value == pytest.approx(
        high["pitch.sd_semitones"].value, rel=0.01
    )


def test_no_absolute_pitch_is_reported(sb: Any, speech_cfg: Any):
    m, _, _ = run(sb, speech_cfg, track(vibrato(150.0, 3.0, 40.0)))
    assert all("hz" not in k.lower() for k in m)
    assert all("information only" in x.method for x in m.values())


def test_octave_jumps_are_folded_back(sb: Any, speech_cfg: Any):
    f0 = vibrato(150.0, 2.0, 40.0)
    clean, _, _ = run(sb, speech_cfg, track(f0))
    bad = f0.copy()
    bad[500:520] *= 2.0  # pYIN doubled the pitch for 0.2 s
    bad[2000:2010] /= 2.0  # and halved it for 0.1 s
    m, _, diag = run(sb, speech_cfg, track(bad))
    assert diag.octave_folded == 30 and diag.dropped == 0
    assert m["pitch.sd_semitones"].value == pytest.approx(
        clean["pitch.sd_semitones"].value, rel=0.01
    )


def test_spikes_that_are_not_octaves_are_dropped(sb: Any, speech_cfg: Any):
    f0 = vibrato(150.0, 2.0, 40.0)
    f0[800:805] *= 2 ** (18 / 12)  # +18 semitones: folding by 12 still leaves 6 > 3
    _, _, diag = run(sb, speech_cfg, track(f0))
    assert diag.dropped == 5 and diag.octave_folded == 0


def test_many_corrections_lower_confidence(sb: Any, speech_cfg: Any):
    f0 = vibrato(150.0, 2.0, 40.0)
    for start in range(0, f0.size, 80):
        f0[start + 30 : start + 50] *= (
            2.0  # 25 % of frames doubled, in runs shorter than the median window
        )
    m, _, diag = run(sb, speech_cfg, track(f0))
    assert diag.octave_folded / diag.usable_frames > 0.2
    assert m["pitch.sd_semitones"].confidence == Confidence.LOW


def test_filler_stretches_are_left_out(sb: Any, speech_cfg: Any):
    f0 = np.full(4000, 150.0)  # 40 s perfectly steady
    f0[1000:1030] = 150.0 * 2 ** (6 / 12)  # a 0.3 s raised "uh" (+6 st)
    with_filler, _, _ = run(sb, speech_cfg, track(f0))
    assert with_filler["pitch.sd_semitones"].value > 0
    hes = FillerEvent(
        kind=FillerKind.HESITATION, first_word=0, last_word=0, start_s=10.0, end_s=10.3
    )
    m, _, diag = run(sb, speech_cfg, track(f0), fillers=(hes,))
    assert m["pitch.sd_semitones"].value == pytest.approx(0.0, abs=1e-6)
    cand = AcousticCandidate(
        start_s=10.0,
        end_s=10.3,
        voiced_fraction=1.0,
        pitch_sd_semitones=0.0,
        agrees_with_lexicon=False,
    )
    m2, _, _ = run(sb, speech_cfg, track(f0), acoustic=(cand,))
    assert m2["pitch.sd_semitones"].value == pytest.approx(0.0, abs=1e-6)
    assert diag.usable_frames < 4000


def test_frames_outside_speech_regions_are_ignored(sb: Any, speech_cfg: Any):
    f0 = np.concatenate([np.full(2000, 150.0), np.full(2000, 300.0)])
    m, _, _ = run(sb, speech_cfg, track(f0), regions=(sb.region(0.0, 20.0),))
    assert m["pitch.sd_semitones"].value == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("voiced_s", "expect"), [(3.0, "not_measurable"), (10.0, "low"), (40.0, "high")]
)
def test_amount_of_voiced_speech(sb: Any, speech_cfg: Any, voiced_s: float, expect: str):
    f0 = vibrato(150.0, 2.0, 40.0)
    voiced = np.zeros(f0.size, bool)
    voiced[: int(voiced_s / HOP)] = True
    m, nm, _ = run(sb, speech_cfg, track(f0, voiced))
    if expect == "not_measurable":
        assert "pitch.sd_semitones" in nm and "pitch.sd_semitones" not in m
    else:
        assert m["pitch.sd_semitones"].confidence == Confidence(expect)


@pytest.mark.parametrize(
    ("snr", "expect"), [(5.0, "not_measurable"), (15.0, "low"), (None, "medium")]
)
def test_noise_rules(sb: Any, speech_cfg: Any, snr: Any, expect: str):
    m, nm, _ = run(sb, speech_cfg, track(vibrato(150.0, 2.0, 40.0)), snr=snr)
    if expect == "not_measurable":
        assert "pitch.sd_semitones" in nm and "noisy" in nm["pitch.sd_semitones"]
    else:
        assert m["pitch.sd_semitones"].confidence == Confidence(expect)


def test_frames_at_the_range_limit_are_reported_and_lower_confidence(sb: Any, speech_cfg: Any):
    f0 = vibrato(450.0, 6.0, 40.0)  # a high voice whose peaks reach ~636 Hz
    f0 = np.minimum(f0, 600.0)  # what a 600 Hz ceiling would return
    m, _, diag = run(sb, speech_cfg, track(f0))
    assert diag.at_limit_share > speech_cfg.pitch_variation.max_at_limit_share
    assert m["pitch.sd_semitones"].confidence == Confidence.LOW
    assert "range limit" in m["pitch.sd_semitones"].method


# --- real pYIN (slow) --------------------------------------------------------------------------


def _tone_with_vibrato(
    base_hz: float, depth_st: float, seconds: float, sr: int = 16_000
) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    f = base_hz * 2 ** (depth_st * np.sin(2 * np.pi * 0.5 * t) / 12)
    phase = 2 * np.pi * np.cumsum(f) / sr
    return (0.3 * np.sin(phase)).astype(np.float32)


@pytest.mark.slow
def test_real_pyin_is_fair_to_a_voice_whose_peaks_pass_400_hz(sb: Any, speech_cfg: Any):
    """Owner's point 2 (ADR 0013): a 340 Hz voice moving +/-6 semitones peaks near 480 Hz. With
    fmax 600 Hz it is measured like a 120 Hz voice with the same movement; with the old 400 Hz
    ceiling it came out ~29 % low (3.02 vs 4.22 semitones on 2026-11-17)."""

    def sd(base: float, fmax: float) -> float:
        cfg = speech_cfg.pitch.model_copy(update={"fmax_hz": fmax})
        tr = pitch.track(_tone_with_vibrato(base, 6.0, 12.0), 16_000, cfg)
        m, _, _ = pitch_variation.analyse(
            tr, (sb.region(0.0, 12.0),), (), (), language="en", snr_db=40.0, cfg=speech_cfg
        )
        return m["pitch.sd_semitones"].value

    assert speech_cfg.pitch.fmax_hz == 600.0
    low, high = sd(120.0, 600.0), sd(340.0, 600.0)
    assert high == pytest.approx(low, rel=0.03)
    assert low == pytest.approx(6.0 / np.sqrt(2), rel=0.05)
    assert sd(340.0, 400.0) < 0.85 * low  # why 400 Hz was unfair
