"""Which received frames go to the landmark worker (spike S1, ADR 0002).

Two samplers with the same interface, ``take(now_s) -> bool`` and ``count``:

- ``EveryNthSampler``: every Nth received frame, whatever the camera rate. Used for the stand-in
  (mode ``none``) and for the ``--process-every N`` override.
- ``IntervalSampler``: time-based. Takes a frame when at least one target interval has passed
  since the last one, so the landmark rate stays near ``target_fps`` when the camera rate
  changes. A 30 fps camera gives about every 2nd frame at 15 fps; a camera at or below the
  target (e.g. 15 fps in low light) gives every frame.

Neither sampler holds frames, so neither can build a backlog. After a long gap
the interval schedule restarts from the current frame instead of catching up with a burst.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Sampler(Protocol):
    count: int

    def take(self, now_s: float) -> bool: ...


class EveryNthSampler:
    """Every Nth received frame (the first frame is always taken)."""

    def __init__(self, every: int) -> None:
        self.every = max(1, every)
        self.count = 0

    def take(self, now_s: float) -> bool:
        self.count += 1
        return (self.count - 1) % self.every == 0


class IntervalSampler:
    """Time-based: at most about ``target_fps`` frames per second, every frame if slower.

    ``tolerance`` is a fraction of the interval by which a frame may arrive early and still be
    taken. Without it, frame-arrival jitter would make a 30 fps camera at a 15 fps target skip
    two frames instead of one now and then. The schedule advances by exactly one interval per
    taken frame, so the long-run rate cannot exceed the target by more than the tolerance allows.
    """

    def __init__(self, target_fps: float, tolerance: float) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if not 0 <= tolerance < 0.5:
            raise ValueError("tolerance must be in [0, 0.5)")
        self.interval_s = 1.0 / target_fps
        self.slack_s = tolerance * self.interval_s
        self.next_due: float | None = None
        self.count = 0

    def take(self, now_s: float) -> bool:
        self.count += 1
        if self.next_due is None:
            self.next_due = now_s + self.interval_s
            return True
        if now_s < self.next_due - self.slack_s:
            return False
        if now_s - self.next_due > self.interval_s:
            # Long gap (camera stalled or slower than the target): restart from this frame.
            self.next_due = now_s + self.interval_s
        else:
            self.next_due += self.interval_s
        return True


@dataclass(frozen=True)
class SamplingSpec:
    """How one run samples frames; ``describe()`` goes into the summary's settings column."""

    every: int | None = None
    target_fps: float | None = None
    tolerance: float = 0.0

    def make(self) -> Sampler:
        if self.target_fps is not None:
            return IntervalSampler(self.target_fps, self.tolerance)
        return EveryNthSampler(self.every or 1)

    def describe(self) -> str:
        if self.target_fps is not None:
            return f"sampling=time target_fps={self.target_fps:g} tolerance={self.tolerance:g}"
        return f"process_every={self.every or 1}"


def choose_sampling(
    *,
    mode: str,
    process_every_override: int | None,
    stand_in_every: int,
    target_fps: float,
    tolerance: float,
) -> SamplingSpec:
    """--process-every wins; the stand-in uses every-Nth; real landmarks use time-based."""
    if process_every_override is not None:
        return SamplingSpec(every=process_every_override)
    if mode == "none":
        return SamplingSpec(every=stand_in_every)
    return SamplingSpec(target_fps=target_fps, tolerance=tolerance)
