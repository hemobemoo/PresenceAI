"""An S2 project-defense session on one microphone stream (P3-e, ADR 0022 P3-e amendment).

The user presses Record once. The session moves through:

    explanation -> answer 1 -> (retry offer) -> answer 2 -> (retry offer) -> done

- **Marks are received-audio sample positions**, never clock times (owner): "Next: question 1"
  and "Done answering" set a mark at the number of samples received so far. A window's length
  is counted in samples too, so a window is exactly `window_s` of received audio whatever the
  network did. The mark is late by the audio still in flight when the button is pressed (ADR
  0022 records the expected error).
- **The explanation is not cut at 90 s**: "Next: question 1" ends it, with a hard stop at the
  limit + `app.recording_extra_s`, so overrun stays measurable (ADR 0019).
- **Answer windows** open when a question appears (reading time is part of the latency) and
  close after `app.answer_window_s` of audio or on "Done answering". A full window is checked
  for silence by voice activity detection only; a silent full window gets **one** retry offer
  ("We didn't hear an answer"): retrying discards that take and reopens the window for the same
  question; keeping it scores it as no answer (ADR 0022). An early "Done" is never a non-answer.

Pure state: no audio, no clock, no UI. The page feeds it sample positions.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum


def pick_questions(
    bank: Sequence[str], own: Sequence[str | None], *, count: int, rng: random.Random
) -> tuple[str, ...]:
    """The user's own questions where given, the rest drawn at random from the bank without
    repeats (owner, P3-e: 2 at random from about 6, or the user's own)."""
    mine = [q.strip() for q in own if q and q.strip()][:count]
    pool = [q for q in bank if q not in mine]
    return (*mine, *rng.sample(pool, count - len(mine)))


class Phase(StrEnum):
    READY = "ready"
    EXPLANATION = "explanation"
    ANSWER = "answer"
    RETRY_OFFER = "retry_offer"
    DONE = "done"


@dataclass(frozen=True)
class Take:
    """One answer attempt: samples [start, end) of the stream."""

    question: int  # 0-based
    start: int
    end: int
    full: bool  # the whole window elapsed (False: "Done answering" before it ended)


@dataclass
class S2Session:
    questions: tuple[str, ...]
    rate: int
    explanation_limit_s: float
    extra_s: float
    window_s: float
    retries: int = 1
    # True for a question from the generic bank: its relevance is reported, not scored (owner,
    # 2026-11-25); the user's own questions keep scoring. Empty: all the user's own.
    from_bank: tuple[bool, ...] = ()
    phase: Phase = Phase.READY
    explanation_end: int | None = None
    explanation_hard_stopped: bool = False
    current: int = 0  # the question being answered
    _window_start: int = 0
    _attempts: dict[int, int] = field(default_factory=dict)
    _pending: Take | None = None  # a full window waiting for its silence check
    _offered: Take | None = None  # a silent full window with a retry on offer
    takes: dict[int, Take] = field(default_factory=dict)  # the kept take per question

    def __post_init__(self) -> None:
        if not self.questions:
            raise ValueError("an S2 session needs at least one question")

    # --- sizes -----------------------------------------------------------------------------

    @property
    def explanation_hard_stop(self) -> int:
        return round((self.explanation_limit_s + self.extra_s) * self.rate)

    @property
    def window(self) -> int:
        return round(self.window_s * self.rate)

    @property
    def stream_limit_s(self) -> float:
        """The stream's time limit for the whole session, retries included, with one extra
        margin for reading the retry prompts and pressing buttons."""
        answers = len(self.questions) * (1 + self.retries) * self.window_s
        return self.explanation_limit_s + self.extra_s + answers + self.extra_s

    # --- transitions -------------------------------------------------------------------------

    def begin(self) -> None:
        """Recording started: the explanation runs from sample 0."""
        if self.phase is not Phase.READY:
            raise ValueError("the session has already begun")
        self.phase = Phase.EXPLANATION

    def next_question(self, position: int) -> None:
        """ "Next: question 1": end the explanation at `position` and open the first window."""
        if self.phase is not Phase.EXPLANATION:
            raise ValueError("not in the explanation")
        self.explanation_end = min(position, self.explanation_hard_stop)
        self._open(0, self.explanation_end)

    def done_answering(self, position: int) -> None:
        """ "Done answering": close the window early. Never a non-answer, so no silence check."""
        if self.phase is not Phase.ANSWER:
            raise ValueError("no answer window is open")
        end = min(position, self._window_start + self.window)
        full = end - self._window_start >= self.window
        self._keep(Take(self.current, self._window_start, end, full=full), open_at=end)

    def tick(self, position: int) -> Take | None:
        """Advance by received audio. Returns a full window that needs its silence check
        (the caller then calls `checked`), else None."""
        if self.phase is Phase.EXPLANATION and position >= self.explanation_hard_stop:
            self.explanation_hard_stopped = True
            self.explanation_end = self.explanation_hard_stop
            self._open(0, self.explanation_end)
        if self.phase is Phase.ANSWER and position - self._window_start >= self.window:
            take = Take(self.current, self._window_start, self._window_start + self.window, True)
            self._pending = take
            return take
        return None

    def checked(self, silent: bool) -> None:
        """The silence check of the pending full window: offer a retry, or keep it."""
        take = self._pending
        if take is None:
            raise ValueError("no window is waiting for its silence check")
        self._pending = None
        if silent and self._attempts.get(take.question, 1) <= self.retries:
            self.phase = Phase.RETRY_OFFER
            self._offered = take
            return
        self._keep(take, open_at=take.end)

    def retry(self, position: int) -> None:
        """ "Try again": discard the silent take and reopen the same question at `position`."""
        if self.phase is not Phase.RETRY_OFFER:
            raise ValueError("no retry is offered")
        self._offered = None
        self._open(self.current, position)

    def keep_silent(self, position: int) -> None:
        """ "Keep (no answer)": the silent full window is scored as no answer; the next
        question appears now, so its window opens at `position`."""
        if self.phase is not Phase.RETRY_OFFER or self._offered is None:
            raise ValueError("no retry is offered")
        take, self._offered = self._offered, None
        self._keep(take, open_at=position)

    # --- views -------------------------------------------------------------------------------

    def window_elapsed_s(self, position: int) -> float:
        if self.phase is not Phase.ANSWER:
            return 0.0
        return min(position - self._window_start, self.window) / self.rate

    def window_start(self) -> int:
        return self._window_start

    def explanation_span(self, position: int) -> tuple[int, int]:
        end = self.explanation_end if self.explanation_end is not None else position
        return 0, min(end, self.explanation_hard_stop)

    def attempts(self, question: int) -> int:
        return self._attempts.get(question, 0)

    # --- internals ---------------------------------------------------------------------------

    def _open(self, question: int, position: int) -> None:
        self.current = question
        self._window_start = position
        self._attempts[question] = self._attempts.get(question, 0) + 1
        self.phase = Phase.ANSWER

    def _keep(self, take: Take, *, open_at: int) -> None:
        self.takes[take.question] = take
        nxt = take.question + 1
        if nxt < len(self.questions):
            self._open(nxt, open_at)  # the next question appears: its window opens
        else:
            self.phase = Phase.DONE
