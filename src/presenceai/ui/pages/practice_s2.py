"""The S2 project-defense controls on the Practice page (P3-e).

`S2Controller` joins the one microphone stream (`RecordingBuffer`) and the session state
(`live/s2_session.py`): buttons place marks at received-audio sample positions, the 1 s timer
advances the session and runs the voice-only silence check of a full window. `view` turns the
state into what the page shows, as plain data, so it is tested without a browser.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from presenceai.lang import Language, followup_questions, ui_text
from presenceai.live.recorder import RecordingBuffer
from presenceai.live.s2_audio import Vad, is_silent
from presenceai.live.s2_session import Phase, S2Session, pick_questions
from presenceai.ui.pages.report import clock

SCENARIO = "s2_project_defense"
QUESTIONS = 2  # scenarios/s2_project_defense.yaml: followups


@dataclass(frozen=True)
class S2View:
    status: str  # elapsed time and what to do now
    question: str  # the question on screen (Markdown-escaped by the page), or ""
    next_on: bool
    done_on: bool
    retry_on: bool


class S2Controller:
    def __init__(
        self,
        buffer: RecordingBuffer,
        *,
        explanation_limit_s: float,
        extra_s: float,
        window_s: float,
        vad: Vad,
        rng: random.Random | None = None,
    ) -> None:
        self.buffer = buffer
        self.explanation_limit_s = explanation_limit_s
        self.extra_s = extra_s
        self.window_s = window_s
        self.vad = vad
        self.rng = rng or random.Random()  # noqa: S311  (picking questions, not security)
        self.session: S2Session | None = None

    def start(self, language: Language, own: Sequence[str | None], rate: int) -> S2Session:
        """Recording started: pick the questions and size the stream for the whole session."""
        questions = pick_questions(followup_questions(language), own, count=QUESTIONS, rng=self.rng)
        self.session = S2Session(
            questions, rate=rate, explanation_limit_s=self.explanation_limit_s,
            extra_s=self.extra_s, window_s=self.window_s,
        )  # fmt: skip
        self.session.begin()
        self.buffer.set_limits(limit_s=self.session.stream_limit_s, extra_s=0)
        return self.session

    def tick(self) -> None:
        s = self.session
        if s is None:
            return
        take = s.tick(self.buffer.position())
        if take is not None:
            s.checked(is_silent(self.buffer.slice(take.start, take.end), self.vad))

    def next_question(self) -> None:
        if self.session is not None and self.session.phase is Phase.EXPLANATION:
            self.session.next_question(self.buffer.position())

    def done_answering(self) -> None:
        if self.session is not None and self.session.phase is Phase.ANSWER:
            self.session.done_answering(self.buffer.position())

    def retry(self) -> None:
        if self.session is not None and self.session.phase is Phase.RETRY_OFFER:
            self.session.retry(self.buffer.position())

    def keep_silent(self) -> None:
        if self.session is not None and self.session.phase is Phase.RETRY_OFFER:
            self.session.keep_silent(self.buffer.position())

    def view(self, lang: Language) -> S2View:
        return view(self.session, self.buffer.position(), lang)


def view(session: S2Session | None, position: int, lang: Language) -> S2View:
    if session is None or session.phase is Phase.READY:
        return S2View(ui_text(lang, "practice.s2.not_started"), "", False, False, False)
    rate = session.rate
    total = len(session.questions)
    if session.phase is Phase.EXPLANATION:
        status = ui_text(
            lang, "practice.s2.explanation_elapsed", clock=clock(position / rate),
            limit=clock(session.explanation_limit_s),
        )  # fmt: skip
        hint = ui_text(lang, "practice.s2.explain", limit=round(session.explanation_limit_s))
        return S2View(f"{status}\n\n{hint}", "", True, False, False)
    notes: list[str] = []
    if session.explanation_hard_stopped:
        hard = round(session.explanation_limit_s + session.extra_s)
        notes.append(ui_text(lang, "practice.s2.hard_stopped", hard=hard))
    n = session.current + 1
    heading = ui_text(lang, "practice.s2.question", n=n, total=total)
    question = f"{heading}\n\n{session.questions[session.current]}"
    if session.phase is Phase.ANSWER:
        notes.append(
            ui_text(
                lang,
                "practice.s2.answer_elapsed",
                n=n,
                clock=clock(session.window_elapsed_s(position)),
                window=clock(session.window_s),
            )
        )
        return S2View("\n\n".join(notes), question, False, True, False)
    if session.phase is Phase.RETRY_OFFER:
        notes.append(ui_text(lang, "practice.s2.no_answer", n=n))
        return S2View("\n\n".join(notes), question, False, False, True)
    notes.append(ui_text(lang, "practice.s2.done"))
    return S2View("\n\n".join(notes), "", False, False, False)
