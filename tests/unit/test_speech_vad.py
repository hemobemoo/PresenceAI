"""speech/vad.py: Silero VAD (bundled with faster-whisper) -> speech regions in seconds."""

from typing import Any

import numpy as np
import pytest

from presenceai.schemas.speech import SpeechRegion
from presenceai.speech import vad


def test_stamps_become_regions_in_seconds():
    stamps = [{"start": 1600, "end": 8000}, {"start": 16000, "end": 24000}]
    assert vad.regions_from_stamps(stamps, 16_000) == (
        SpeechRegion(start_s=0.1, end_s=0.5),
        SpeechRegion(start_s=1.0, end_s=1.5),
    )


def test_options_from_config_reach_silero(monkeypatch: pytest.MonkeyPatch, speech_cfg: Any):
    seen: dict[str, Any] = {}

    def fake(audio: np.ndarray, vad_options: Any = None, sampling_rate: int = 16000) -> Any:
        seen["opts"], seen["sr"] = vad_options, sampling_rate
        return [{"start": 0, "end": 1600}]

    from faster_whisper import vad as fw_vad

    monkeypatch.setattr(fw_vad, "get_speech_timestamps", fake)
    regions = vad.detect_speech(np.zeros(16_000, dtype=np.float32), speech_cfg.vad)
    assert regions == (SpeechRegion(start_s=0.0, end_s=0.1),)
    opts = seen["opts"]
    assert opts.threshold == speech_cfg.vad.threshold == 0.5
    assert opts.min_speech_duration_ms == speech_cfg.vad.min_speech_duration_ms == 250
    assert opts.min_silence_duration_ms == speech_cfg.vad.min_silence_duration_ms == 100
    assert opts.speech_pad_ms == speech_cfg.vad.speech_pad_ms == 0  # padding would shorten pauses
    assert seen["sr"] == 16_000


def test_silence_has_no_speech(speech_cfg: Any):
    """Runs the real bundled Silero model (ships inside faster-whisper; no download)."""
    assert vad.detect_speech(np.zeros(32_000, dtype=np.float32), speech_cfg.vad) == ()


def test_empty_audio(speech_cfg: Any):
    assert vad.detect_speech(np.zeros(0, dtype=np.float32), speech_cfg.vad) == ()
