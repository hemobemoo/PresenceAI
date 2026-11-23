"""One S2 follow-up answer from its recording (ADR 0022): status, latency, relevance, limitation.

"No answer given" is decided from the audio, never from an empty transcript (owner,
2026-11-23): the recording passed its checks, the full answer window elapsed and the voice
detector found no speech. Then relevance is 0.0 and the latency is the whole window, which is
longer than every latency limit, so the answer fails every follow-up condition and the dimension
scores level 1. Speech without a transcript, an early stop or a failed recording give missing
values instead (the partial-evidence wording applies, ADR 0019). P3 offers a retry before a
non-answer is scored.
"""

from __future__ import annotations

from collections.abc import Sequence

from presenceai.knowledge.embed import Embedder
from presenceai.schemas.followup import AnswerMeasures
from presenceai.schemas.speech import SpeechRegion
from presenceai.speech.latency import response_latency
from presenceai.text.followup import limitation_cue, relevance


def measure_answer(
    index: int,
    *,
    question: str,
    transcript: str,
    regions: Sequence[SpeechRegion],
    recording_ok: bool,
    window_s: float,
    elapsed_s: float,
    embedder: Embedder | None,
) -> AnswerMeasures:
    """`embedder` None: relevance is not computed (e.g. Arabic, ADR 0022 item 6)."""
    if not recording_ok:
        return AnswerMeasures(index=index, window_s=window_s, status="recording_failed")
    if not regions:
        if elapsed_s >= window_s:
            return AnswerMeasures(
                index=index, window_s=window_s, status="no_answer", latency_s=window_s,
                relevance=0.0,
            )  # fmt: skip
        return AnswerMeasures(index=index, window_s=window_s, status="incomplete")
    latency = response_latency(regions)
    if not transcript.strip():
        return AnswerMeasures(
            index=index, window_s=window_s, status="speech_untranscribed", latency_s=latency
        )
    return AnswerMeasures(
        index=index,
        window_s=window_s,
        status="answered",
        latency_s=latency,
        relevance=relevance(question, transcript, embedder) if embedder is not None else None,
        limitation=limitation_cue(transcript),
    )
