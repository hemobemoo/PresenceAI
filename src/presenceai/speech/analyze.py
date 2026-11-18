"""Post-recording speech analysis: one audio file -> `SpeechReport`.

Slice 1: transcription and ASR quality. Slice 2: VAD speech regions, pauses (with position),
speech rate, articulation rate and syllable pace (ADR 0011). Slice 3: pitch track (pYIN, once
per recording), fillers (lexicon + acoustic candidates), discourse markers and repetitions
(ADR 0012). Slice 4: relative loudness (clipping on the original samples, noise, browser
processing, optional baseline) and pitch variation from the same pitch track (ADR 0013).
Slice 6: all report texts come from lang/<lang>/report.yaml in the recording's language,
and a language check on the first speech adds a notice on a mismatch (ADR 0016).
Words Whisper invents after the last speech are removed right after VAD,
before any measure or quote (ADR 0015).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from presenceai.lang import text
from presenceai.schemas.speech import (
    CaptureProcessing,
    LoudnessProfile,
    PitchTrack,
    SpeechLanguage,
    SpeechReport,
)
from presenceai.settings import AsrSettings, PitchSettings, SpeechSettings
from presenceai.speech import (
    asr,
    audio,
    fillers,
    language_check,
    loudness,
    pauses,
    pitch_variation,
    rate,
    repetitions,
    trim,
)
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
    capture: CaptureProcessing | None = None,
    baseline: tuple[LoudnessProfile, CaptureProcessing, str] | None = None,
) -> SpeechReport:
    capture = capture or CaptureProcessing()  # uploaded files: unknown
    transcript = asr.transcribe_file(model, path, language=language, options=options)
    samples = audio.load_audio(path)
    regions = vad(samples) if vad is not None else vad_mod.detect_speech(samples, speech.vad)
    transcript, outside = trim.trim_to_speech(transcript, regions, speech.transcript)
    lang_check = language_check.check(model, samples, regions, language, speech.language_check)
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

    m_loud, nm_loud, profile = loudness.summarise(
        samples,
        audio.SAMPLE_RATE,
        audio.load_native(path),
        regions,
        capture=capture,
        baseline=baseline[:2] if baseline else None,
        language=language,
        cfg=speech,
    )
    m_pitch, nm_pitch, diag = pitch_variation.analyse(
        track, regions, lexical, acoustic, language=language, snr_db=profile.snr_db, cfg=speech
    )

    measurements |= m_rate | m_pace | m_fill | m_rep | m_loud | m_pitch
    if not regions:
        not_measurable["transcript"] = text(language, "common.no_speech_recording")
    not_measurable |= nm_rate | nm_pace | nm_fill | nm_rep | nm_loud | nm_pitch

    return SpeechReport(
        source_name=Path(path).name,
        model=model_name,
        transcript=transcript,
        asr_quality=asr.quality(transcript),
        language_check=lang_check,
        words_outside_speech=outside,
        speech_regions=regions,
        pauses=events,
        rate_windows=windows,
        fillers=lexical,
        acoustic_candidates=acoustic,
        repetitions=reps,
        pitch=track,
        capture=capture,
        loudness_profile=profile,
        pitch_diagnostics=diag,
        baseline_source=baseline[2] if baseline else None,
        measurements=measurements,
        not_measurable=not_measurable,
    )
