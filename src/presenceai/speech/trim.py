"""Remove words Whisper invents after the last speech (ADR 0015).

Base Whisper sometimes writes text into silence at the end of a recording (C1 clip s10_r0:
"Thank you very much for joining us. Thank you." after the last sentence). Such words would
raise `rate.wpm` and could be quoted back as the user's own words (AGENTS.md rule 4).

Rule ("end only"): a word is removed when it starts after the last VAD speech offset plus
`end_tolerance_s`. Words before the first speech are kept: Whisper often squeezes the real first
word into the start of the file (0.0-0.2 s) although speech starts later, and a start rule
removed such real words in the E2 test part. Words in pauses between speech regions are kept,
because Whisper's word times drift into pauses. With no speech regions at all, every word is
removed.

The kept words get a rebuilt `Transcript.text` and character offsets, so no evidence quote can
contain a removed word. Removed words are returned with their original offsets, for the report.
"""

from __future__ import annotations

from presenceai.schemas.speech import Segment, SpeechRegion, Transcript, Word
from presenceai.settings import TranscriptSettings


def trim_to_speech(
    transcript: Transcript, regions: tuple[SpeechRegion, ...], cfg: TranscriptSettings
) -> tuple[Transcript, tuple[Word, ...]]:
    """(transcript without words after the last speech, the removed words in order)."""
    if regions:
        last = regions[-1].end_s + cfg.end_tolerance_s

        def keep(w: Word) -> bool:
            return w.start_s <= last
    else:

        def keep(w: Word) -> bool:
            return False

    dropped = tuple(w for w in transcript.words if not keep(w))
    if not dropped:
        return transcript, ()

    text = ""
    segments: list[Segment] = []
    for seg in transcript.segments:
        kept: list[Word] = []
        for w in seg.words:
            if not keep(w):
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
