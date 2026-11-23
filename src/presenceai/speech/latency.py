"""Response latency of an S2 follow-up answer (ADR 0022 §8).

The answer recording starts automatically when the question ends, so the latency is the start
of the first speech region found by voice activity detection. If the question was shown as
text, the reading time is included. No speech region: no latency (None), so the condition is
dropped, never scored low.
"""

from __future__ import annotations

from collections.abc import Sequence

from presenceai.schemas.speech import SpeechRegion


def response_latency(regions: Sequence[SpeechRegion]) -> float | None:
    """Seconds from the start of the answer recording to the first speech onset."""
    if not regions:
        return None
    return min(r.start_s for r in regions)
