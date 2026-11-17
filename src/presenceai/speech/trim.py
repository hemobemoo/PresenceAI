"""Remove words Whisper invents before the first or after the last speech (ADR 0015).

Base Whisper sometimes writes text into silence, mostly at the end of a recording (C1 clip
s10_r0: "Thank you very much for joining us. Thank you." after the last sentence). Such words
would raise `rate.wpm` and could be quoted back as the user's own words (AGENTS.md rule 4).

Rule ("ends only"): a word is removed when it ends before the first VAD speech onset minus
`edge_tolerance_s`, or starts after the last speech offset plus `edge_tolerance_s`. Words in
pauses between speech regions are kept, because Whisper's word times often drift into pauses.
With no speech regions at all, every word is removed.

The kept words get a rebuilt `Transcript.text` and character offsets, so no evidence quote can
contain a removed word. Removed words are returned with their original offsets, for the report.
"""

from __future__ import annotations

from presenceai.schemas.speech import Segment, SpeechRegion, Transcript, Word
from presenceai.settings import TranscriptSettings


def trim_to_speech(
    transcript: Transcript, regions: tuple[SpeechRegion, ...], cfg: TranscriptSettings
) -> tuple[Transcript, tuple[Word, ...]]:
    """(transcript without words outside speech, the removed words in transcript order)."""
    if regions:
        first = regions[0].start_s - cfg.edge_tolerance_s
        last = regions[-1].end_s + cfg.edge_tolerance_s

        def inside(w: Word) -> bool:
            return w.end_s >= first and w.start_s <= last
    else:

        def inside(w: Word) -> bool:
            return False

    dropped = tuple(w for w in transcript.words if not inside(w))
    if not dropped:
        return transcript, ()

    text = ""
    segments: list[Segment] = []
    for seg in transcript.segments:
        kept: list[Word] = []
        for w in seg.words:
            if not inside(w):
                continue
            if text:
                text += " "
            start = len(text)
            text += w.text
            kept.append(w.model_copy(update={"start_char": start, "end_char": len(text)}))
        if not kept:
            continue
        if len(kept) < len(seg.words):  # narrow the segment to the words it still holds
            seg = seg.model_copy(update={"start_s": kept[0].start_s, "end_s": kept[-1].end_s})
        segments.append(seg.model_copy(update={"words": tuple(kept)}))
    trimmed = Transcript(
        language=transcript.language,
        duration_s=transcript.duration_s,
        text=text,
        segments=tuple(segments),
    )
    return trimmed, dropped
