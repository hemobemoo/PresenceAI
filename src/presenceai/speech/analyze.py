"""Post-recording speech analysis: one audio file -> `SpeechReport`.

P1 slice 1 covers transcription and ASR quality. Pauses and rate (slice 2), fillers and
repetitions (slice 3) and loudness and pitch (slice 4) add their measurements here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from presenceai.schemas.speech import SpeechLanguage, SpeechReport
from presenceai.settings import AsrSettings
from presenceai.speech import asr


def analyze_file(
    path: str | Path,
    *,
    language: SpeechLanguage,
    model: Any,
    model_name: str,
    options: AsrSettings,
) -> SpeechReport:
    transcript = asr.transcribe_file(model, path, language=language, options=options)
    return SpeechReport(
        source_name=Path(path).name,
        model=model_name,
        transcript=transcript,
        asr_quality=asr.quality(transcript),
    )
