"""speech/pitch.py: one pYIN run per recording, limited pitch range, kept for slice 4.

Tests that run real pYIN are marked `slow` (numba compiles on first use, ~10-20 s):
    uv run pytest -m slow tests/unit/test_speech_pitch.py
"""

from typing import Any

import numpy as np
import pytest

from presenceai.speech import pitch

SR = 16_000


def _tone(hz: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@pytest.mark.slow
def test_track_finds_a_steady_pitch(speech_cfg: Any):
    track = pitch.track(_tone(150.0, 1.0), SR, speech_cfg.pitch)
    assert track.hop_s == pytest.approx(speech_cfg.pitch.hop_length / SR)
    assert (track.fmin_hz, track.fmax_hz) == (speech_cfg.pitch.fmin_hz, speech_cfg.pitch.fmax_hz)
    f0 = np.array(track.f0_hz)[np.array(track.voiced)]
    assert np.mean(track.voiced) > 0.8
    assert np.median(f0) == pytest.approx(150.0, rel=0.02)


@pytest.mark.slow
def test_silence_is_unvoiced(speech_cfg: Any):
    track = pitch.track(np.zeros(SR, dtype=np.float32), SR, speech_cfg.pitch)
    assert not any(track.voiced) and set(track.f0_hz) == {0.0}


@pytest.mark.slow
def test_pitch_outside_the_range_is_not_reported(speech_cfg: Any):
    track = pitch.track(_tone(40.0, 1.0), SR, speech_cfg.pitch)  # below fmin 65 Hz
    f0 = np.array(track.f0_hz)[np.array(track.voiced)]
    assert f0.size == 0 or np.all(f0 >= speech_cfg.pitch.fmin_hz)


def test_empty_audio(speech_cfg: Any):
    track = pitch.track(np.zeros(0, dtype=np.float32), SR, speech_cfg.pitch)
    assert track.f0_hz == () and track.voiced == ()


def test_track_is_kept_in_memory_but_not_written_to_json(speech_cfg: Any, sb: Any):
    from presenceai.schemas.speech import AsrQuality, SpeechReport

    track = sb.pitch_track(0.5, segments=[(0.1, 0.4, 150.0, 150.0)])  # no pYIN needed
    r = SpeechReport(
        source_name="a.wav",
        model="whisper_small",
        transcript=sb.transcript([]),
        asr_quality=AsrQuality(word_count=0),
        pitch=track,
    )
    assert r.pitch is track
    assert "pitch" not in r.model_dump() and '"f0_hz"' not in r.model_dump_json()
