"""Post-recording speech analysis: one audio file -> `SpeechReport`.

Slice 1: transcription and ASR quality. Slice 2: VAD speech regions, pauses (with position),
speech rate, articulation rate and syllable pace (ADR 0011). Slice 3: pitch track (pYIN, once
per recording), fillers (lexicon + acoustic candidates), discourse markers and repetitions
(ADR 0012). Loudness and pitch variation (slice 4) add their measurements here and reuse the
pitch track.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from presenceai.schemas.speech import PitchTrack, SpeechLanguage, SpeechReport
from presenceai.settings import AsrSettings, PitchSettings, SpeechSettings
from presenceai.speech import asr, audio, fillers, pauses, rate, repetitions
from presenceai.speech import pitch as pitch_mod
from presenceai.speech import vad as vad_mod

PitchTracker = Callable[[np.ndarray, int, PitchSettings], PitchTrack]


def analyze_file(
    path: str | Path,
    *,
    language: SpeechLanguage,
    model: Any,
    model_name: str,
    options: AsrSettings,
    speech: SpeechSettings,
    vad: vad_mod.Vad | None = None,
    pitch_tracker: PitchTracker | None = None,
) -> SpeechReport:
    transcript = asr.transcribe_file(model, path, language=language, options=options)
    samples = audio.load_audio(path)
    regions = vad(samples) if vad is not None else vad_mod.detect_speech(samples, speech.vad)
    tracker = pitch_tracker or pitch_mod.track
    track = tracker(samples, audio.SAMPLE_RATE, speech.pitch)
    verbatim = asr.is_verbatim(model_name)

    events = pauses.find_pauses(regions, transcript, speech)
    measurements, not_measurable = pauses.summarise(events, regions, transcript, speech)
    m_rate, nm_rate, windows = rate.word_rate(transcript, regions, speech)
    m_pace, nm_pace = rate.syllable_rate(samples, audio.SAMPLE_RATE, regions, language, speech)

    lexicon = fillers.load_lexicon(language)
    lexical = fillers.find_lexical(transcript, lexicon)
    acoustic = fillers.acoustic_candidates(regions, transcript, track, speech, lexical)
    m_fill, nm_fill = fillers.summarise(
        lexical, acoustic, transcript, regions, speech, verbatim=verbatim
    )
    reps = repetitions.find_repetitions(transcript, lexicon, speech)
    m_rep, nm_rep = repetitions.summarise(reps, transcript, regions, speech, verbatim=verbatim)

    measurements |= m_rate | m_pace | m_fill | m_rep
    not_measurable |= nm_rate | nm_pace | nm_fill | nm_rep

    return SpeechReport(
        source_name=Path(path).name,
        model=model_name,
        transcript=transcript,
        asr_quality=asr.quality(transcript),
        speech_regions=regions,
        pauses=events,
        rate_windows=windows,
        fillers=lexical,
        acoustic_candidates=acoustic,
        repetitions=reps,
        pitch=track,
        measurements=measurements,
        not_measurable=not_measurable,
    )
