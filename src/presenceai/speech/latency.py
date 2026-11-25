"""Response latency of an S2 follow-up answer (ADR 0022 §8; P3-e amendment).

The answer recording starts automatically when the question appears, so the latency is the start
of the first speech region found by voice activity detection. If the question was shown as
text, the reading time is included. No speech region: no latency (None), so the condition is
dropped, never scored low.

**Speech already going when the answer window opens is not an onset** (owner, P3-e). In the
app one microphone stream runs through the whole S2 session, so the user may still be talking
(the end of the explanation, or of the previous answer) when the next question appears. The app
looks back `app.answer_preroll_s` before the window's mark: a speech region that began before the
mark and continues after it is *running speech*, and `running_until` returns where it ends,
in seconds from the mark. The latency is then the first speech start at or after that point.
Answers recorded as separate files (the `assess` command) have no running speech (0.0).
"""

from __future__ import annotations

from collections.abc import Sequence

from presenceai.schemas.speech import SpeechRegion

_EPS = 1e-6


def running_until(regions: Sequence[SpeechRegion], preroll_s: float) -> float:
    """Seconds after the mark where speech that began before the mark ends (0.0 if none).

    `regions` come from voice activity detection on the audio from `preroll_s` before the mark
    to the end of the answer window, so the mark is at `preroll_s`."""
    end = 0.0
    for r in regions:
        if r.start_s < preroll_s - _EPS and r.end_s > preroll_s + _EPS:
            end = max(end, r.end_s - preroll_s)
    return end


def response_latency(regions: Sequence[SpeechRegion], running_until_s: float = 0.0) -> float | None:
    """Seconds from the start of the answer recording to the first speech onset, ignoring speech
    that was already running when the recording started (it ends at `running_until_s`)."""
    if running_until_s <= 0:
        return min((r.start_s for r in regions), default=None)
    # a region of the window audio that starts at the cut is the running speech, not an onset
    onsets = [r.start_s for r in regions if r.start_s >= running_until_s - _EPS]
    return min(onsets, default=None)
