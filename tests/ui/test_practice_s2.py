"""S2 on the Practice page (P3-e): one stream, sample marks, silence check and retry, running
speech, and the analysis with follow-ups. Fakes only: no models, no microphone, no browser."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from presenceai.live.recorder import RecordingBuffer
from presenceai.live.s2_audio import session_inputs
from presenceai.live.s2_session import Phase
from presenceai.pipeline import app_run
from presenceai.schemas.speech import AsrQuality, SpeechRegion, SpeechReport
from presenceai.schemas.storage import Consent
from presenceai.settings import Settings
from presenceai.speech.audio import load_audio
from presenceai.ui.context import AppContext
from presenceai.ui.pages import practice
from presenceai.ui.pages import practice_s2 as s2_page


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (reproducible test picks, not security)


RATE = 16_000
T0 = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)


def fake_vad(samples: np.ndarray) -> tuple[SpeechRegion, ...]:
    """Speech = runs of non-zero samples (a stand-in for Silero; tests only)."""
    on = np.abs(samples) > 1e-4
    regions, start = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            regions.append(SpeechRegion(start_s=start / RATE, end_s=i / RATE))
            start = None
    return tuple(regions)


def _feed(buffer: RecordingBuffer, seconds: float, *, speech: bool) -> None:
    n = round(seconds * RATE)
    value = 3000 if speech else 0
    for i in range(0, n, 1600):
        buffer.add(RATE, np.full((1, min(1600, n - i)), value, dtype=np.int16))


@pytest.fixture
def controller() -> s2_page.S2Controller:
    buffer = RecordingBuffer(limit_s=180, extra_s=60, min_ratio=0.95, clock=lambda: 0.0)
    c = s2_page.S2Controller(
        buffer, explanation_limit_s=90, extra_s=60, window_s=60, vad=fake_vad,
        rng=seeded(7),
    )  # fmt: skip
    buffer.start()
    c.start("en", [None, None], RATE)
    return c


def test_session_on_one_stream_with_a_retry(controller: s2_page.S2Controller):
    c, b = controller, controller.buffer
    assert b.hard_stop_s == 450  # the stream is sized for the whole session, retries included
    assert c.view("en").next_on and "Explanation: 00:00 of 01:30" in c.view("en").status
    _feed(b, 20, speech=True)
    c.next_question()  # mark at 20 s of received audio
    v = c.view("en")
    assert v.done_on and v.question.startswith("Question 1 of 2") and c.session is not None
    _feed(b, 60, speech=False)  # a silent full window
    c.tick()
    v = c.view("en")
    assert v.retry_on and "We didn't hear an answer to question 1" in v.status
    _feed(b, 3, speech=False)  # reading the offer
    c.retry()  # the silent take is discarded; question 1 again, from 83 s
    assert c.session.phase is Phase.ANSWER and c.session.window_start() == 83 * RATE
    _feed(b, 1, speech=False)
    _feed(b, 10, speech=True)
    c.done_answering()
    assert c.session.takes[0].start == 83 * RATE and c.session.current == 1
    _feed(b, 2, speech=False)
    _feed(b, 58, speech=True)
    c.tick()  # full window with speech: kept, done
    assert c.session.phase is Phase.DONE and "Both questions are done" in c.view("en").status


def test_answer_1_running_into_question_2_is_not_its_onset(controller: s2_page.S2Controller):
    c, b = controller, controller.buffer
    _feed(b, 5, speech=True)
    c.next_question()  # question 1 at 5 s
    _feed(b, 1, speech=False)
    _feed(b, 10, speech=True)
    c.done_answering()  # "Done" at 16 s while still talking: question 2 appears at 16 s
    _feed(b, 2, speech=True)  # the end of answer 1 runs 2 s into question 2's window
    _feed(b, 3, speech=False)  # reading question 2
    _feed(b, 20, speech=True)  # answer 2 starts 5 s after the mark
    c.done_answering()
    assert c.session is not None
    rec = b.finish()
    assert rec is not None
    explanation, answers = session_inputs(rec.samples, c.session, preroll_s=1.0, vad=fake_vad)
    assert len(load_audio(explanation)) == 5 * RATE
    assert answers[0].running_until_s == pytest.approx(0.0)  # silence at question 1's mark
    assert answers[1].running_until_s == pytest.approx(2.0)  # answer 1 ran 2 s past the mark
    a2 = answers[1].audio
    assert a2 is not None and len(load_audio(a2)) == 25 * RATE


def test_the_explanation_hard_stop_shows_question_1(controller: s2_page.S2Controller):
    c, b = controller, controller.buffer
    _feed(b, 151, speech=True)  # still explaining past 90 + 60 s
    c.tick()
    assert c.session is not None and c.session.explanation_hard_stopped
    v = c.view("en")
    assert v.question.startswith("Question 1 of 2") and "stopped at 150 s" in v.status


# --- the analysis -----------------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path: Path):
    c = AppContext.open(
        tmp_path / "home", SecretStr(Fernet.generate_key().decode()), raw_media_days=30, now=T0
    )
    c.db.set_consent(Consent(use_app=True, keep_recordings=True), now=T0)
    yield c
    c.close()


class BagOfWords:
    """A tiny embedder: word overlap, so relevance is computed without e5."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: Any, kind: Any) -> np.ndarray:
        import zlib

        m = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):  # the same fixed space for every call
            for w in t.lower().split():
                m[i, zlib.crc32(w.encode()) % 256] += 1
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def _answer(sb: Any, text: str, onset: float, duration: float) -> SpeechReport:
    t = sb.spaced(text, language="en", duration_s=duration)
    t = sb.transcript([(w.text, w.start_s + onset, w.end_s + onset) for w in t.words],
                      language="en", duration_s=duration)  # fmt: skip
    return SpeechReport(
        source_name="a.wav", model="whisper_small", transcript=t,
        asr_quality=AsrQuality(word_count=len(t.words)),
        speech_regions=(SpeechRegion(start_s=onset, end_s=t.words[-1].end_s),),
    )  # fmt: skip


def test_s2_analysis_end_to_end_with_fakes(
    ctx: AppContext, controller: s2_page.S2Controller, sb: Any, fake_assessment: Any,
    fake_parser: Any, caplog: pytest.LogCaptureFixture,
):  # fmt: skip
    c, b = controller, controller.buffer
    _feed(b, 30, speech=True)
    c.next_question()
    _feed(b, 1.5, speech=False)
    _feed(b, 8.5, speech=True)
    c.done_answering()
    _feed(b, 3, speech=False)
    _feed(b, 7, speech=True)
    c.done_answering()
    assert c.session is not None
    q1, q2 = c.session.questions
    loads: list[str] = []

    def analyze(audio: Any, **_: Any) -> SpeechReport:
        if audio.name == "explanation.wav":
            return fake_assessment.speech
        if audio.name == "answer_1.wav":
            return _answer(sb, f"{q1} because the users asked for it", 1.5, 30.0)
        return _answer(sb, f"{q2} we measured it twice", 3.0, 30.0)

    analyzer = app_run.Analyzer(
        Settings(_env_file=None),  # pyright: ignore[reportCallIssue]
        load_whisper=lambda name, options: loads.append("whisper") or object(),
        load_parser=lambda: fake_parser,
        analyze=analyze,
        load_embedder=lambda: loads.append("e5") or BagOfWords(),
    )
    with caplog.at_level("INFO", logger="presenceai"):
        updates = list(practice.run_s2(ctx, analyzer, "en", b.finish(), c.session,
                                       language="en", vad=fake_vad))  # fmt: skip
    steps = [u.status for u in updates if u.busy and "step" in u.status]
    assert [s.split(":")[0] for s in steps] == [f"Analysing… step {i} of 5" for i in range(1, 6)]
    final = updates[-1]
    assert not final.busy and r"### Follow\-up handling" in final.report
    assert "could not be checked" not in final.report
    assert "**Questions**" in final.report
    assert loads == ["whisper", "e5"] and not analyzer.models_loaded  # e5 for English S2 only
    (s,) = ctx.db.list_sessions()
    assert s.scenario == "s2_project_defense" and s.status == "analysed"
    assert ctx.store is not None
    assert ctx.store.media_names(s.id) == ("answer_1.wav", "answer_2.wav", "explanation.wav")
    values = {m.measure: m.value for m in ctx.db.summary(s.id).measures}
    assert values["followup.latency_max_s"] == pytest.approx(3.0)  # the slower answer
    assert "followup.relevance_min" in values
    assert not any("users asked" in r.getMessage() for r in caplog.records)
