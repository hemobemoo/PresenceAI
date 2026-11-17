"""speech/loudness.py: relative loudness only (ADR 0013). Synthetic tones with known levels."""

from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from presenceai.schemas.assessment import Confidence
from presenceai.schemas.speech import CaptureProcessing, LoudnessProfile
from presenceai.speech import audio, loudness

SR = 16_000
OFF = CaptureProcessing(auto_gain_control="off", noise_suppression="off", echo_cancellation="off")


def tone(amp: float, seconds: float, hz: float = 200.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def noise(level: float, seconds: float, seed: int = 0) -> np.ndarray:
    return (level * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(
        np.float32
    )


def _summary(
    sb: Any,
    cfg: Any,
    x: np.ndarray,
    regions: Any,
    *,
    capture: Any = OFF,
    baseline: Any = None,
    native: Any = None,
):
    native = native if native is not None else (x[None, :], SR)
    return loudness.summarise(
        x, SR, native, regions, capture=capture, baseline=baseline, language="en", cfg=cfg
    )


# --- one shared level function -------------------------------------------------------------


def test_frame_level_of_a_known_tone(speech_cfg: Any):
    # sine of amplitude A has RMS A/sqrt(2): 20*log10(0.5/sqrt(2)) = -9.03 dBFS
    assert loudness.frame_dbfs(tone(0.5, 0.02)) == pytest.approx(-9.03, abs=0.05)


def test_doubling_the_amplitude_adds_6_db(speech_cfg: Any):
    x = np.concatenate([tone(0.1, 2.0), tone(0.2, 2.0)])
    levels = loudness.window_levels(x, SR, speech_cfg.loudness)
    assert levels[2] - levels[0] == pytest.approx(6.02, abs=0.05)


def test_live_meter_and_report_use_the_same_function(speech_cfg: Any):
    """Owner's point 2: the live path (20 ms chunks -> frame_dbfs) gives exactly the report's
    window levels."""
    x = np.concatenate([tone(0.1, 1.0), tone(0.3, 1.0), tone(0.05, 1.0)])
    frame = int(SR * speech_cfg.loudness.frame_ms / 1000)
    live = [loudness.frame_dbfs(x[i : i + frame]) for i in range(0, x.size - frame + 1, frame)]
    per_window = int(speech_cfg.loudness.window_s * 1000 / speech_cfg.loudness.frame_ms)
    live_windows = [
        loudness.power_mean_db(live[i : i + per_window]) for i in range(0, len(live), per_window)
    ]
    assert np.allclose(live_windows, loudness.window_levels(x, SR, speech_cfg.loudness))


# --- variation, spread, silence ignored ------------------------------------------------------


def _steps(amps: list[float], each_s: float = 2.0) -> np.ndarray:
    return np.concatenate([tone(a, each_s) for a in amps])


def test_variation_and_spread(sb: Any, speech_cfg: Any):
    x = _steps([0.1, 0.2] * 4)  # 16 s, alternating -6 dB / 0 dB steps
    m, nm, _ = _summary(sb, speech_cfg, x, (sb.region(0.0, 16.0),))
    assert nm == {}
    assert m["loudness.spread_db"].value == pytest.approx(6.02, abs=0.1)
    # linear RMS values 0.1/√2 and 0.2/√2 in equal parts: CV = sd/mean = 0.05/0.15
    assert m["loudness.variation_cv"].value == pytest.approx(1 / 3, abs=0.01)
    assert "information only" in m["loudness.spread_db"].method


def test_pauses_do_not_count_as_quiet(sb: Any, speech_cfg: Any):
    x = np.concatenate([tone(0.2, 6.0), np.zeros(4 * SR, np.float32), tone(0.2, 6.0)])
    regions = (sb.region(0.0, 6.0), sb.region(10.0, 16.0))
    m, _, _ = _summary(sb, speech_cfg, x, regions)
    assert m["loudness.spread_db"].value == pytest.approx(0.0, abs=0.05)
    assert m["loudness.variation_cv"].value == pytest.approx(0.0, abs=0.01)


def test_too_little_speech_is_not_measurable(sb: Any, speech_cfg: Any):
    x = tone(0.2, 5.0)
    m, nm, _ = _summary(sb, speech_cfg, x, (sb.region(0.0, 5.0),))
    assert "loudness.variation_cv" in nm and "loudness.variation_cv" not in m


def test_absolute_level_is_never_a_measurement(sb: Any, speech_cfg: Any):
    x = _steps([0.1, 0.2] * 4)
    m, _, profile = _summary(sb, speech_cfg, x, (sb.region(0.0, 16.0),))
    assert isinstance(profile, LoudnessProfile) and profile.median_speech_dbfs is not None
    assert not any("dbfs" in k or "level" in k for k in m)


# --- noise / SNR -----------------------------------------------------------------------------


def _with_noise(level: float) -> np.ndarray:
    speech = _steps([0.2] * 8)  # 16 s at -17 dBFS
    x = np.concatenate([np.zeros(2 * SR, np.float32), speech, np.zeros(2 * SR, np.float32)])
    return x + noise(level, x.size / SR)


@pytest.mark.parametrize(
    ("noise_level", "expect"),
    [
        (0.001, "not_low"),  # speech -17 dBFS, noise -60 dBFS: ~43 dB SNR
        (0.01, "not_low"),  # noise -40 dBFS: ~23 dB
        (0.03, "low"),  # noise -30.5 dBFS: ~13 dB (10-20 dB -> low)
        (0.1, "not_measurable"),  # noise -20 dBFS: ~5 dB (< 10 dB)
    ],
)
def test_snr_sets_confidence(sb: Any, speech_cfg: Any, noise_level: float, expect: str):
    x = _with_noise(noise_level)
    m, nm, profile = _summary(sb, speech_cfg, x, (sb.region(2.0, 18.0),))
    assert profile.snr_db is not None
    if expect == "not_measurable":
        assert "loudness.variation_cv" in nm and "noisy" in nm["loudness.variation_cv"]
    elif expect == "low":
        assert m["loudness.variation_cv"].confidence == Confidence.LOW
    else:
        assert m["loudness.variation_cv"].confidence != Confidence.LOW


def test_no_silence_to_measure_noise_caps_at_medium(sb: Any, speech_cfg: Any):
    x = _steps([0.1, 0.2] * 20)  # 80 s of speech, no non-speech at all
    m, _, profile = _summary(sb, speech_cfg, x, (sb.region(0.0, 80.0),))
    assert profile.snr_db is None
    assert m["loudness.variation_cv"].confidence == Confidence.MEDIUM


# --- capture processing (browser auto-gain) -----------------------------------------------------


@pytest.mark.parametrize(
    "capture",
    [
        CaptureProcessing(),  # unknown: every uploaded file
        CaptureProcessing(auto_gain_control="on", noise_suppression="off", echo_cancellation="off"),
        CaptureProcessing(auto_gain_control="off", noise_suppression="on", echo_cancellation="off"),
    ],
)
def test_capture_processing_on_or_unknown_caps_variation_at_low(
    sb: Any, speech_cfg: Any, capture: Any
):
    x = _with_noise(0.0005)
    x = np.concatenate([x] * 3)  # long, clean: would be high
    regions = (sb.region(2.0, 18.0), sb.region(22.0, 38.0), sb.region(42.0, 58.0))
    m, _, _ = _summary(sb, speech_cfg, x, regions, capture=capture)
    assert m["loudness.variation_cv"].confidence == Confidence.LOW
    assert m["loudness.spread_db"].confidence == Confidence.LOW
    assert "auto-gain" in m["loudness.variation_cv"].method
    m_off, _, _ = _summary(sb, speech_cfg, x, regions, capture=OFF)
    assert m_off["loudness.variation_cv"].confidence == Confidence.HIGH


# --- baseline --------------------------------------------------------------------------------


def test_change_from_own_baseline(sb: Any, speech_cfg: Any):
    regions = (sb.region(2.0, 18.0),)
    _, _, base = _summary(sb, speech_cfg, _with_noise(0.0005), regions)
    louder = _with_noise(0.0005) * 2.0  # +6 dB
    m, _, _ = _summary(sb, speech_cfg, louder, regions, baseline=(base, OFF))
    ch = m["loudness.change_from_baseline_db"]
    assert ch.value == pytest.approx(6.02, abs=0.2)
    assert ch.confidence == Confidence.MEDIUM  # a mic or distance change cannot be detected
    m_u, _, _ = _summary(sb, speech_cfg, louder, regions, baseline=(base, CaptureProcessing()))
    assert m_u["loudness.change_from_baseline_db"].confidence == Confidence.LOW


def test_no_baseline_no_change_measure(sb: Any, speech_cfg: Any):
    m, nm, _ = _summary(sb, speech_cfg, _with_noise(0.0005), (sb.region(2.0, 18.0),))
    assert (
        "loudness.change_from_baseline_db" not in m and "loudness.change_from_baseline_db" not in nm
    )


# --- clipping on the original samples ---------------------------------------------------------


def test_clipping_runs_are_counted_single_peaks_are_not(speech_cfg: Any):
    x = tone(0.5, 2.0, sr=44_100)
    x[1000:1010] = 1.0  # a 10-sample clipped run
    x[5000] = 1.0  # a single peak sample: ignored (min run 3)
    runs, seconds = loudness.clipping(x[None, :], 44_100, speech_cfg.loudness)
    assert runs == 1 and seconds == pytest.approx(10 / 44_100)


def _write_mp3(path: Path, x: np.ndarray, sr: int) -> Path:
    """Stereo MP3 (libmp3lame, bundled with PyAV) from float samples in [-1, 1]."""
    with av.open(str(path), "w") as out:
        st: Any = out.add_stream("libmp3lame", rate=sr, layout="stereo")
        st.bit_rate = 192_000
        frame = av.AudioFrame.from_ndarray(
            np.tile(x, (2, 1)).astype(np.float32), format="fltp", layout="stereo"
        )
        frame.sample_rate = sr
        for packet in st.encode(frame):
            out.mux(packet)
        for packet in st.encode(None):
            out.mux(packet)
    return path


def test_clipping_in_a_compressed_mp3_is_possible_clipping(tmp_path: Path, speech_cfg: Any):
    """Owner's point 3: a clipped recording that was compressed to MP3 still shows clipping when
    the original-rate samples are checked (load_native), before resampling to 16 kHz. LAME lowers
    the clipped peaks to ~0.97, so it is the plateau rule that finds it: possible clipping."""
    sr = 44_100
    clipped = np.clip(tone(1.6, 3.0, sr=sr), -1.0, 1.0)  # hard-clipped sine
    path = _write_mp3(tmp_path / "clipped.mp3", clipped, sr)
    native, native_sr = audio.load_native(path)
    assert native_sr == sr and native.shape[0] == 2
    hard = loudness.clipping(native, native_sr, speech_cfg.loudness)
    runs, seconds = loudness.possible_clipping(native, native_sr, speech_cfg.loudness)
    assert runs > 50 and seconds > 0.1
    assert hard[0] < runs  # plateau samples are not also counted as clipping

    clean = _write_mp3(tmp_path / "clean.mp3", tone(0.5, 3.0, sr=sr), sr)
    native, native_sr = audio.load_native(clean)
    assert loudness.clipping(native, native_sr, speech_cfg.loudness) == (0, 0.0)
    assert loudness.possible_clipping(native, native_sr, speech_cfg.loudness) == (0, 0.0)


def test_a_loud_unclipped_tone_is_not_clipping(speech_cfg: Any):
    """Control for the plateau rule: a sine is the hardest unclipped case (ratio ~2.3 < 4)."""
    x = tone(0.95, 2.0, sr=44_100)[None, :]
    assert loudness.clipping(x, 44_100, speech_cfg.loudness) == (0, 0.0)
    assert loudness.possible_clipping(x, 44_100, speech_cfg.loudness) == (0, 0.0)


def _speechlike(seconds: float, sr: int, seed: int = 1) -> np.ndarray:
    """Harmonic voice with moving pitch, a syllable envelope, phrase loudness and some noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    phase = 2 * np.pi * np.cumsum(140 * 2 ** (2 * np.sin(2 * np.pi * 0.3 * t) / 12)) / sr
    x = sum((0.7**k) * np.sin(k * phase + rng.uniform(0, 6)) for k in range(1, 12))
    env = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 0.7
    env *= 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 0.2 * t))
    x = x * env + 0.02 * rng.standard_normal(t.size)
    return x / np.max(np.abs(x))


def _limit(x: np.ndarray, sr: int, drive_db: float, ceiling: float, release_ms: float):
    """Look-ahead peak limiter (1 ms): |y| never exceeds the ceiling, gain recovers smoothly.
    This is how mastered podcast and video audio gets loud without clipping."""
    from scipy.ndimage import minimum_filter1d

    y = x * 10 ** (drive_db / 20)
    look = max(1, sr // 1000)
    target = minimum_filter1d(np.minimum(1.0, ceiling / np.maximum(np.abs(y), 1e-9)), 2 * look + 1)
    att, rel = np.exp(-2 / look), np.exp(-1 / (release_ms * sr / 1000))
    g, cur = np.empty_like(target), 1.0
    for i, tg in enumerate(target):
        cur = att * cur + (1 - att) * tg if tg < cur else rel * cur + (1 - rel) * tg
        g[i] = min(cur, tg)
    return (y * g).astype(np.float32)


@pytest.mark.parametrize("release_ms", [1.0, 80.0])
def test_a_limited_but_unclipped_signal_is_not_possible_clipping(
    tmp_path: Path, speech_cfg: Any, release_ms: float
):
    """Owner's safeguard 2 (ADR 0013): loud audio through a limiter, as mastered podcast or
    video audio is, must stay well under the plateau ratio, both as WAV and after MP3. Measured
    on 2026-11-17 over 3 seeds, drives +6 to +18 dB, ceilings 0.891-0.98, releases 1-80 ms:
    at most 1.20 (WAV) and 0.20 (MP3), against the rule's 4."""
    sr, lc = 44_100, speech_cfg.loudness
    y = _limit(_speechlike(5.0, sr), sr, drive_db=18.0, ceiling=0.98, release_ms=release_ms)
    assert float(np.abs(y).max()) <= 0.98 + 1e-6
    ratio = loudness.pileup_ratio(y, lc)
    assert ratio is not None and ratio < 0.5 * lc.clip_pileup_ratio
    assert loudness.possible_clipping(y[None, :], sr, lc) == (0, 0.0)
    native, _ = audio.load_native(_write_mp3(tmp_path / "limited.mp3", y, sr))
    assert loudness.clipping(native, sr, lc) == (0, 0.0)
    assert loudness.possible_clipping(native, sr, lc) == (0, 0.0)


def test_clipping_share_measure(sb: Any, speech_cfg: Any):
    x = _steps([0.2] * 8)
    native = x.copy()
    native[SR : SR + 160] = 1.0  # 10 ms clipped
    m, _, profile = _summary(
        sb, speech_cfg, x, (sb.region(0.0, 16.0),), native=(native[None, :], SR)
    )
    assert profile.clipped_runs == 1
    assert m["loudness.clipping_share"].value == pytest.approx(0.01 / 16.0, rel=0.01)
    assert m["loudness.clipping_share"].evidence[0].time is not None
    assert m["loudness.possible_clipping_share"].value == 0.0


def test_possible_clipping_is_reported_apart_and_always_low(sb: Any, speech_cfg: Any):
    """Owner's safeguard 1: plateau detections are "possible clipping", never mixed with hard
    clipping, always low confidence even on a clean, quiet-room recording."""
    x = _steps([0.2] * 8)
    native = 0.5 * np.sin(2 * np.pi * 200 * np.arange(x.size) / SR)
    native = np.clip(native * 2.0, -0.97, 0.97)  # a plateau at 0.97, as LAME leaves clipping
    m, _, profile = _summary(
        sb, speech_cfg, x, (sb.region(0.0, 16.0),), native=(native[None, :].astype(np.float32), SR)
    )
    assert profile.clipped_runs == 0 and profile.possible_clipped_runs > 0
    assert m["loudness.clipping_share"].value == 0.0
    pc = m["loudness.possible_clipping_share"]
    assert pc.value > 0 and pc.confidence == Confidence.LOW
    assert "possible clipping" in pc.method
