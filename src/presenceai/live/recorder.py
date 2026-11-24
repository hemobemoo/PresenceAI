"""In-app recording (P3-d, ADR 0024): microphone frames from FastRTC into memory.

- Frames arrive as (sample_rate, int16 or float samples). They are kept **in memory only**; the
  caller encrypts the finished recording (ADR 0023), so no clear audio is written to disk.
- **Elapsed time** is wall-clock time since the first frame (the page shows it as a timer).
- **Hard stop:** frames are ignored once the elapsed time reaches the scenario limit plus
  `app.recording_extra_s` (60 s, provisional), so overrun stays measurable (ADR 0019) and a
  forgotten recording cannot grow without end.
- **Completeness:** received audio seconds divided by wall-clock seconds. Below
  `app.completeness_min_ratio` (0.95, provisional) the page warns that the recording looks
  incomplete (frames were dropped on the way).

A take starts with its first frame (`start`, called by the FastRTC handler) and stays in memory
when the microphone stream closes (`end_stream`), until "Stop and analyse" takes it (`finish`).
One recording at a time: the local app has one user. The clock is injectable for tests.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, repr=False)
class Recording:
    samples: np.ndarray  # float32 mono in [-1, 1]
    sample_rate: int
    received_s: float
    wall_s: float
    ratio: float  # received_s / wall_s (1.0 when nothing was lost)
    complete: bool
    hard_stopped: bool

    def __repr__(self) -> str:  # never the samples
        return f"Recording(received_s={self.received_s:.1f}, wall_s={self.wall_s:.1f})"


def _mono_float(samples: np.ndarray) -> np.ndarray:
    """FastRTC frames are (channels, n) int16; scale to [-1, 1] first, then average channels."""
    a = np.asarray(samples)
    a = (
        a.astype(np.float32) / 32768.0
        if np.issubdtype(a.dtype, np.integer)
        else a.astype(np.float32)
    )
    if a.ndim == 2:
        a = a.mean(axis=0) if a.shape[0] <= a.shape[1] else a.mean(axis=1)
    return a.reshape(-1).astype(np.float32, copy=False)


class RecordingBuffer:
    def __init__(
        self,
        *,
        limit_s: float,
        extra_s: float,
        min_ratio: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit_s = limit_s
        self.hard_stop_s = limit_s + extra_s
        self.min_ratio = min_ratio
        self._clock = clock
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._rate: int | None = None
        self._first: float | None = None
        self._last: float | None = None
        self._hard_stopped = False
        self._active = False

    def start(self) -> None:
        with self._lock:
            self._chunks, self._rate = [], None
            self._first = self._last = None
            self._hard_stopped, self._active = False, True

    @property
    def active(self) -> bool:
        return self._active

    def end_stream(self) -> None:
        """The microphone stream closed (the user stopped the microphone box). The take is
        kept for "Stop and analyse"; no frame is added after this and the timer stops."""
        with self._lock:
            self._active = False

    def has_audio(self) -> bool:
        with self._lock:
            return bool(self._chunks)

    def add(self, sample_rate: int, samples: np.ndarray) -> bool:
        """Keep a frame; False when it is ignored (not recording, or past the hard stop)."""
        now = self._clock()
        with self._lock:
            if not self._active:
                return False
            if self._first is None:
                self._first = now
            if now - self._first >= self.hard_stop_s:
                self._hard_stopped = True
                return False
            if self._rate is None:
                self._rate = int(sample_rate)
            elif int(sample_rate) != self._rate:
                return False  # a rate change mid-recording: drop the frame, never mix rates
            self._chunks.append(_mono_float(samples))
            self._last = now
            return True

    def elapsed_s(self) -> float:
        with self._lock:
            if self._first is None:
                return 0.0
            end = self._clock() if self._active else (self._last or self._first)
            return min(end - self._first, self.hard_stop_s)

    @property
    def hard_stopped(self) -> bool:
        return self._hard_stopped or (self._active and self.elapsed_s() >= self.hard_stop_s)

    def finish(self) -> Recording | None:
        """Stop and return the recording, or None when no frame arrived."""
        with self._lock:
            self._active = False
            if not self._chunks or self._rate is None or self._first is None:
                return None
            samples = np.concatenate(self._chunks)
            received = len(samples) / self._rate
            # Wall time the frames cover: arrival of the first to arrival of the last frame, plus
            # the first frame's own length (it arrived when it ended). Idle time after the stream
            # stopped is not counted, so pressing Stop late never looks like lost audio.
            first_len = len(self._chunks[0]) / self._rate
            span = (self._last or self._first) - self._first + first_len
            wall = max(span, 1e-9)
            ratio = min(received / wall, 1.0)
            self._chunks = []
            return Recording(
                samples=samples,
                sample_rate=self._rate,
                received_s=received,
                wall_s=wall,
                ratio=ratio,
                complete=ratio >= self.min_ratio,
                hard_stopped=self._hard_stopped,  # frames were refused at the hard stop
            )
