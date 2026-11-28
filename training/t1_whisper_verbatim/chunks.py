"""Per-window chunks for long audio (ADR 0028 §12, details in ADR 0029 §8). Pure functions.

Speech regions come from Silero VAD with the app's settings. Every chunk spans at most `max_s`,
cuts lie in pauses, and the chunks cover the whole recording, so no audio is dropped.
"""

from __future__ import annotations

from collections.abc import Sequence


def _pieces(regions: Sequence[tuple[float, float]], duration: float, max_s: float):
    out = []
    for s, e in sorted(regions):
        s, e = max(0.0, s), min(duration, e)
        while e - s > max_s:  # a region over max_s is cut every max_s
            out.append((s, s + max_s))
            s += max_s
        if e > s:
            out.append((s, e))
    return out


def _groups(pieces, max_s: float):
    groups: list[list[float]] = []
    for s, e in pieces:
        if groups and e - groups[-1][0] <= max_s:
            groups[-1][1] = max(groups[-1][1], e)
        else:
            groups.append([s, e])
    return groups


def _fill(a: float, b: float, max_s: float) -> list[tuple[float, float]]:
    out = []
    while b - a > 1e-9:
        out.append((a, min(b, a + max_s)))
        a = out[-1][1]
    return out


def plan_chunks(
    regions: Sequence[tuple[float, float]], duration: float, max_s: float = 30.0
) -> list[tuple[float, float]]:
    """Chunks (start, end) in seconds, in order, contiguous from 0 to `duration`."""
    if duration <= 0:
        return []
    groups = _groups(_pieces(regions, duration, max_s), max_s)
    if not groups:
        return _fill(0.0, duration, max_s)
    spans = []
    for i, (gs, ge) in enumerate(groups):
        prev_end = groups[i - 1][1] if i else None
        next_start = groups[i + 1][0] if i + 1 < len(groups) else None
        want_l = gs if prev_end is None else (gs - prev_end) / 2
        want_r = duration - ge if next_start is None else (next_start - ge) / 2
        slack = max_s - (ge - gs)
        left = min(want_l, max(slack / 2, slack - want_r))
        right = min(want_r, slack - left)
        spans.append((gs - left, ge + right))
    chunks: list[tuple[float, float]] = []
    pos = 0.0
    for s, e in spans:
        chunks += _fill(pos, s, max_s)  # a pause part no neighbour can take within max_s
        chunks.append((s, e))
        pos = e
    chunks += _fill(pos, duration, max_s)
    return chunks
