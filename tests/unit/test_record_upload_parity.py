"""The same audio through the recording path and the upload path (owner, ADR 0024 acceptance):
the same sample array (scale and dtype) and the same speech measures.

Recording path: int16 frames as FastRTC delivers them -> the FastRTC handler -> RecordingBuffer
-> Recording -> WAV bytes in memory (`media.wav_bytes`) -> `MemoryAudio`.
Upload path: the same int16 samples saved as a 16-bit WAV file -> `check_upload` -> `MemoryAudio`.
Both are then decoded by the one audio loader and analysed with the same fakes (no models), so
any difference comes from the two paths themselves. FastRTC's own resampling is outside this
test: frames are handed over at 16 kHz, the rate the handler asks for.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from presenceai.live.recorder import RecordingBuffer
from presenceai.pipeline import app_run
from presenceai.settings import AppSettings, AsrSettings
from presenceai.speech import analyze
from presenceai.speech.audio import MemoryAudio, load_audio, load_native
from presenceai.storage import media
from presenceai.ui.pages import practice

from .test_analyze_cli import fake_pitch, fake_vad

RATE = 16_000


def _talk_int16(seconds: float = 4.0) -> np.ndarray:
    """Tone bursts of different loudness over low noise, with full-scale peaks of both signs."""
    rng = np.random.default_rng(20261124)
    t = np.arange(int(seconds * RATE)) / RATE
    envelope = np.where((t % 1.0) < 0.6, 0.2 + 0.6 * (t / seconds), 0.0)
    x = envelope * np.sin(2 * np.pi * 180 * t) + rng.normal(0, 0.003, t.size)
    pcm = np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)
    pcm[1000], pcm[2000] = 32767, -32768  # the extremes must survive both paths
    return pcm


def _recorded(pcm: np.ndarray) -> MemoryAudio:
    buffer = RecordingBuffer(limit_s=180, extra_s=60, min_ratio=0.95)
    handler = practice.make_handler(buffer)
    handler.start_up()
    for i in range(0, pcm.size, 960):  # 60 ms frames, shaped (1, n) like FastRTC's
        handler.receive((RATE, pcm[None, i : i + 960]))
    rec = buffer.finish()
    assert rec is not None and rec.sample_rate == RATE
    return MemoryAudio("talk.wav", media.wav_bytes(rec.samples, rec.sample_rate))


def _uploaded(pcm: np.ndarray, folder: Path) -> MemoryAudio:
    path = folder / "talk.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.astype("<i2").tobytes())
    check = app_run.check_upload(path, AppSettings())
    assert check.audio is not None
    return check.audio


@pytest.fixture
def both(tmp_path: Path) -> tuple[MemoryAudio, MemoryAudio, np.ndarray]:
    pcm = _talk_int16()
    return _recorded(pcm), _uploaded(pcm, tmp_path), pcm


def test_same_samples_same_scale_and_dtype(both: Any):
    recorded, uploaded, pcm = both
    a, b = load_audio(recorded), load_audio(uploaded)
    assert a.dtype == b.dtype == np.float32
    assert a.shape == b.shape == (pcm.size,)
    np.testing.assert_array_equal(a, b)  # bit for bit, not just close
    assert float(a.max()) == pytest.approx(32767 / 32768) and float(a.min()) == -1.0
    (na, ra), (nb, rb) = load_native(recorded), load_native(uploaded)
    assert ra == rb == RATE and na.dtype == nb.dtype == np.float32
    np.testing.assert_array_equal(na, nb)  # the clipping check sees the same samples
    assert recorded.data == uploaded.data  # the very same WAV bytes


def test_same_speech_measures(both: Any, fake_whisper: Any, speech_cfg: Any):
    recorded, uploaded, _ = both

    def report(audio: MemoryAudio) -> Any:
        return analyze.analyze_file(
            audio, language="en", model=fake_whisper(), model_name="whisper_small",
            options=AsrSettings(), speech=speech_cfg, vad=fake_vad, pitch_tracker=fake_pitch,
        )  # fmt: skip

    a, b = report(recorded), report(uploaded)
    assert a.measurements and set(a.measurements) == set(b.measurements)
    for key in a.measurements:
        assert a.measurements[key].value == b.measurements[key].value, key
    assert a.not_measurable == b.not_measurable
    assert a.loudness_profile == b.loudness_profile
