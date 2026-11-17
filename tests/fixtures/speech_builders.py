"""Builders for speech test data (transcripts, regions, pitch tracks). No models, no audio files.

Imported by tests through the `speech_builders` fixture in tests/conftest.py.
"""

from __future__ import annotations

from collections.abc import Sequence

from presenceai.schemas.speech import PitchTrack, Segment, SpeechRegion, Transcript, Word


def transcript(
    tokens: Sequence[tuple[str, float, float]],
    *,
    p: float = 0.95,
    language: str = "en",
    duration_s: float = 120.0,
) -> Transcript:
    """Tokens as (text, start_s, end_s); one segment; every word with probability `p`."""
    text, words = "", []
    for token, start, end in tokens:
        if text:
            text += " "
        s = len(text)
        text += token
        words.append(
            Word(
                text=token,
                start_s=start,
                end_s=end,
                probability=p,
                start_char=s,
                end_char=len(text),
            )
        )
    segs: tuple[Segment, ...] = ()
    if words:
        segs = (
            Segment(
                start_s=words[0].start_s,
                end_s=words[-1].end_s,
                words=tuple(words),
                avg_logprob=-0.1,
                no_speech_prob=0.0,
            ),
        )
    return Transcript(
        language=language,  # pyright: ignore[reportArgumentType]
        duration_s=duration_s,
        text=text,
        segments=segs,
    )


def spaced(text: str, *, step: float = 0.5, dur: float = 0.3, **kw: object) -> Transcript:
    """One word every `step` seconds, each lasting `dur`."""
    toks = [(t, i * step, i * step + dur) for i, t in enumerate(text.split())]
    return transcript(toks, **kw)  # pyright: ignore[reportArgumentType]


def region(a: float, b: float) -> SpeechRegion:
    return SpeechRegion(start_s=a, end_s=b)


def pitch_track(
    duration_s: float,
    *,
    hop_s: float = 0.01,
    segments: Sequence[tuple[float, float, float, float]] = (),
) -> PitchTrack:
    """Unvoiced everywhere except `segments`: (start_s, end_s, f0_start_hz, f0_end_hz), linear."""
    n = round(duration_s / hop_s)
    f0 = [0.0] * n
    voiced = [False] * n
    for start, end, f_a, f_b in segments:
        i0, i1 = round(start / hop_s), round(end / hop_s)
        for i in range(i0, min(i1, n)):
            frac = (i - i0) / max(1, i1 - i0 - 1)
            f0[i] = f_a + (f_b - f_a) * frac
            voiced[i] = True
    return PitchTrack(
        hop_s=hop_s, fmin_hz=65.0, fmax_hz=400.0, f0_hz=tuple(f0), voiced=tuple(voiced)
    )
