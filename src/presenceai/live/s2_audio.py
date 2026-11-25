"""Cutting an S2 session's stream into the explanation and the answers (P3-e).

The stream is 16 kHz float32 mono in memory. The explanation is samples [0, explanation end);
each kept answer take is [start, end). For each answer the voice detector also looks back
`preroll_s` before the take's mark, so speech already running when the window opened is not
taken as the answer's onset (`speech/latency.py: running_until`). The silence check of a full
window uses voice activity detection only, never the transcript (ADR 0022).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from presenceai.live.s2_session import S2Session, Take
from presenceai.pipeline.app_run import AnswerInput
from presenceai.schemas.speech import SpeechRegion
from presenceai.speech.audio import MemoryAudio
from presenceai.speech.latency import response_latency, running_until
from presenceai.storage.media import wav_bytes

Vad = Callable[[np.ndarray], Sequence[SpeechRegion]]


def is_silent(window: np.ndarray, vad: Vad) -> bool:
    """No speech region in the window's own audio (a measured non-answer if kept)."""
    return window.size == 0 or not vad(window)


def answer_is_silent(audio: np.ndarray, preroll: int, rate: int, vad: Vad) -> bool:
    """A window with no new speech: nothing starts after the speech that was already running
    when the window opened. `audio` begins `preroll` samples before the window's mark.

    The same rule as the latency (owner, P3-e: running speech is not an onset). Without it, the
    audio still in flight when the question appears (about 0.1-0.3 s, ADR 0022) would make a
    silent answer look answered (found by the browser journey test, 2026-11-25)."""
    if audio.size <= preroll:
        return True
    regions = vad(audio)
    run = running_until(regions, preroll / rate)
    window = [
        SpeechRegion(start_s=max(0.0, r.start_s - preroll / rate), end_s=r.end_s - preroll / rate)
        for r in regions
        if r.end_s > preroll / rate
    ]
    return response_latency(window, run) is None


def window_audio(stream: np.ndarray, take: Take) -> np.ndarray:
    return stream[take.start : take.end]


def answer_running_until(
    stream: np.ndarray, take: Take, *, rate: int, preroll_s: float, vad: Vad
) -> float:
    """Where speech that was already going at the take's mark ends, in seconds from the mark."""
    pre = min(round(preroll_s * rate), take.start)
    if pre <= 0:
        return 0.0
    regions = vad(stream[take.start - pre : take.end])
    return running_until(regions, pre / rate)


def session_inputs(
    stream: np.ndarray, session: S2Session, *, preroll_s: float, vad: Vad
) -> tuple[MemoryAudio, list[AnswerInput]]:
    """The explanation's audio and one input per question (None audio: not recorded)."""
    rate = session.rate
    start, end = session.explanation_span(len(stream))
    explanation = MemoryAudio("explanation.wav", wav_bytes(stream[start:end], rate))
    answers: list[AnswerInput] = []
    for i, question in enumerate(session.questions):
        take = session.takes.get(i)
        bank = bool(session.from_bank) and session.from_bank[i]
        if take is None or take.end <= take.start:
            answers.append(AnswerInput(question, None, session.window_s, relevance_scored=not bank))
            continue
        audio = MemoryAudio(f"answer_{i + 1}.wav", wav_bytes(window_audio(stream, take), rate))
        running = answer_running_until(stream, take, rate=rate, preroll_s=preroll_s, vad=vad)
        answers.append(
            AnswerInput(question, audio, session.window_s, running, relevance_scored=not bank)
        )
    return explanation, answers
