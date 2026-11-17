"""Post-recording speech analysis: one audio file -> `SpeechReport`.

Slice 1: transcription and ASR quality. Slice 2: VAD speech regions, pauses (with position),
speech rate, articulation rate and syllable pace (ADR 0011). Fillers and repetitions (slice 3)
and loudness and pitch (slice 4) add their measurements here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from presenceai.schemas.speech import SpeechLanguage, SpeechReport
from presenceai.settings import AsrSettings, SpeechSettings
from presenceai.speech import asr, audio, pauses, rate
from presenceai.speech import vad as vad_mod


def analyze_file(
    path: str | Path,
    *,
    language: SpeechLanguage,
    model: Any,
    model_name: str,
    options: AsrSettings,
    speech: SpeechSettings,
    vad: vad_mod.Vad | None = None,
) -> SpeechReport:
    transcript = asr.transcribe_file(model, path, language=language, options=options)
    samples = audio.load_audio(path)
    regions = vad(samples) if vad is not None else vad_mod.detect_speech(samples, speech.vad)

    events = pauses.find_pauses(regions, transcript, speech)
    measurements, not_measurable = pauses.summarise(events, regions, transcript, speech)
    m_rate, nm_rate, windows = rate.word_rate(transcript, regions, speech)
    m_pace, nm_pace = rate.syllable_rate(samples, audio.SAMPLE_RATE, regions, language, speech)
    measurements |= m_rate | m_pace
    not_measurable |= nm_rate | nm_pace

    return SpeechReport(
        source_name=Path(path).name,
        model=model_name,
        transcript=transcript,
        asr_quality=asr.quality(transcript),
        speech_regions=regions,
        pauses=events,
        rate_windows=windows,
        measurements=measurements,
        not_measurable=not_measurable,
    )
