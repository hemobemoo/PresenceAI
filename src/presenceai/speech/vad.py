"""Voice activity detection with the Silero VAD model that ships inside faster-whisper.

`silero_vad_v6.onnx` (MIT) is part of the faster-whisper wheel, so nothing is downloaded
(AGENTS.md rule 2). Options come from `configs/default.yaml` `speech.vad` (ADR 0011). Padding is
0 ms: Silero's usual padding would widen every speech region and shorten every pause.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np

from presenceai.schemas.speech import SpeechRegion
from presenceai.settings import VadSettings
from presenceai.speech.audio import SAMPLE_RATE

Vad = Callable[[np.ndarray], tuple[SpeechRegion, ...]]


def regions_from_stamps(
    stamps: Iterable[Mapping[str, int]], sampling_rate: int
) -> tuple[SpeechRegion, ...]:
    return tuple(
        SpeechRegion(start_s=s["start"] / sampling_rate, end_s=s["end"] / sampling_rate)
        for s in stamps
    )


def detect_speech(
    samples: np.ndarray, opts: VadSettings, sampling_rate: int = SAMPLE_RATE
) -> tuple[SpeechRegion, ...]:
    if samples.size == 0:
        return ()
    from faster_whisper import vad as fw_vad  # loads onnxruntime only when needed

    options = fw_vad.VadOptions(
        threshold=opts.threshold,
        min_speech_duration_ms=opts.min_speech_duration_ms,
        min_silence_duration_ms=opts.min_silence_duration_ms,
        speech_pad_ms=opts.speech_pad_ms,
    )
    stamps = fw_vad.get_speech_timestamps(samples, options, sampling_rate=sampling_rate)
    return regions_from_stamps(stamps, sampling_rate)
