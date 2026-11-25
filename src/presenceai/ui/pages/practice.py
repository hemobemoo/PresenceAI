"""Practice page (P3-d): record in the app (FastRTC, audio only) or upload a file, then analyse.

- **Recording:** FastRTC `WebRTC(modality="audio", mode="send", full_screen=False)`, frames
  resampled to 16 kHz and kept in memory (`live/recorder.py`). `full_screen=False` matters:
  FastRTC 0.0.34's default draws a full-screen waveform layer that swallows every click on the
  page (owner's round-1 test: neither analyse button reacted). **No STUN or TURN server**:
  `{"iceServers": []}` on the browser and on the server (aiortc would otherwise contact
  stun.l.google.com). A take starts with its first frame and survives stopping the microphone
  box until "Stop and analyse" takes it.
- **Timer:** a 1 s `gr.Timer`, active only while the microphone records (FastRTC's
  `start_recording` / `stop_recording` events) and sent outside Gradio's queue, so it no longer
  floods the server with queued requests (round 1: hundreds of Windows 10054 resets).
- **Upload:** read into memory, Gradio's temp copy deleted, limits checked
  (`pipeline/app_run.py: check_upload`). Uploads have no time limit (overrun is missing). WMA
  gets its own message.
- **Pressing either analyse button always ends in a report or a visible message:** the handler
  is a generator that first shows "Preparing…", then "Analysing… step N of 4", disables both
  buttons while it runs, and turns every failure into a message on the page.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from presenceai.evalconfig import REPO_ROOT
from presenceai.lang import Language, ui_text
from presenceai.live.recorder import Recording, RecordingBuffer
from presenceai.live.s2_audio import Vad, session_inputs
from presenceai.live.s2_session import S2Session
from presenceai.pipeline import app_run
from presenceai.schemas.storage import MediaSource
from presenceai.speech.audio import MemoryAudio
from presenceai.storage import media, reports
from presenceai.ui.context import AppContext
from presenceai.ui.pages import practice_s2 as s2_page
from presenceai.ui.pages import report as report_page

log = logging.getLogger(__name__)

SCENARIOS = ("s1_presentation", "s2_project_defense")  # S2: follow-ups on one stream (P3-e)
NO_ICE: dict[str, Any] = {"iceServers": []}  # no STUN/TURN in Stage A
INPUT_RATE = 16_000


def scenario_limit_s(scenario: str) -> float:
    """The scenario's time limit: the upper end of `duration_s` in scenarios/<id>.yaml."""
    data = yaml.safe_load((REPO_ROOT / "scenarios" / f"{scenario}.yaml").read_text("utf-8"))
    return float(data["duration_s"][-1])


@dataclass(frozen=True)
class Update:
    """One state of the page while an analysis runs (or after it)."""

    status: str  # Markdown: progress, notices and errors, in the page language
    report: str = ""  # Markdown report
    transcript: str = ""  # plain text
    busy: bool = False  # True while running: the analyse buttons are disabled


Outcome = Update  # the final state


def _keep_days(ctx: AppContext) -> int | None:
    return ctx.raw_media_days if ctx.consent().keep_recordings else None


def _run(
    ctx: AppContext,
    analyzer: app_run.Analyzer,
    lang: Language,
    *,
    language: Language,
    scenario: str,
    source: MediaSource,
    audio: MemoryAudio,
    duration_s: float,
    limit_s: float | None,
    notices: list[str],
    completeness: float | None = None,
    answers: Sequence[app_run.AnswerInput] | None = None,
) -> Iterator[Update]:
    if ctx.store is None:
        yield Update(ui_text(lang, "practice.key_missing"))
        return
    if analyzer.busy():
        yield Update(ui_text(lang, "practice.busy"))
        return
    keep = _keep_days(ctx)
    session = ctx.db.new_session(scenario, language, source, now=datetime.now(UTC), keep_days=keep)
    ctx.store.write_bytes(session.id, audio.name, audio.data)  # encrypted at write time
    for a in answers or ():
        if a.audio is not None:  # only the kept take of each question; a retried one is gone
            ctx.store.write_bytes(session.id, a.audio.name, a.audio.data)
    steps = analyzer.steps(
        ctx.db, ctx.store, session, audio, limit_s=limit_s, keep=keep is not None,
        duration_s=duration_s, completeness=completeness, answers=answers,
    )  # fmt: skip
    try:
        while True:
            try:
                step = next(steps)
            except StopIteration:
                break
            name = ui_text(lang, f"practice.steps.{step.name}")
            text = ui_text(lang, "practice.step_status", n=step.index, total=step.total, step=name)
            yield Update("\n\n".join([*notices, text]), busy=True)
    except app_run.AnalysisBusy:
        yield Update(ui_text(lang, "practice.busy"))
        return
    except app_run.AnalysisFailed as e:
        yield Update("\n\n".join([*notices, ui_text(lang, "practice.failed", error=str(e))]))
        return
    view = reports.open_report(ctx.db, ctx.store, session.id, lang=lang)
    yield Update(
        "\n\n".join([*notices, ui_text(lang, "practice.done")]),
        report_page.render(view, lang, scenario),
        report_page.transcript_text(view),
    )


def _guarded(lang: Language, updates: Iterator[Update]) -> Iterator[Update]:
    """Every press ends in a report or a visible message, never a silent stop."""
    last: Update | None = None
    try:
        for last in updates:
            yield last
    except Exception as e:  # anything unexpected: say so on the page, log the type only
        log.warning("practice handler failed: error=%s", type(e).__name__)
        yield Update(ui_text(lang, "practice.failed", error=type(e).__name__))
        return
    if last is None or last.busy:
        yield Update(ui_text(lang, "practice.failed", error="no result"))


def run_recording(
    ctx: AppContext,
    analyzer: app_run.Analyzer,
    lang: Language,
    recording: Recording | None,
    *,
    language: Language,
    scenario: str,
    extra_s: float,
) -> Iterator[Update]:
    def inner() -> Iterator[Update]:
        yield Update(ui_text(lang, "practice.preparing"), busy=True)
        if recording is None:
            yield Update(ui_text(lang, "practice.no_audio"))
            return
        limit = scenario_limit_s(scenario)
        notices: list[str] = []
        if not recording.complete:
            notices.append(
                ui_text(
                    lang,
                    "practice.incomplete",
                    received=recording.received_s,
                    wall=recording.wall_s,
                )
            )
        if recording.hard_stopped:
            hard, extra = round(limit + extra_s), round(extra_s)
            notices.append(ui_text(lang, "practice.hard_stopped", hard=hard, extra=extra))
        audio = MemoryAudio("talk.wav", media.wav_bytes(recording.samples, recording.sample_rate))
        yield from _run(
            ctx, analyzer, lang, language=language, scenario=scenario, source="app", audio=audio,
            duration_s=recording.received_s, limit_s=limit, notices=notices,
            completeness=recording.ratio,
        )  # fmt: skip

    return _guarded(lang, inner())


def run_s2(
    ctx: AppContext,
    analyzer: app_run.Analyzer,
    lang: Language,
    recording: Recording | None,
    session: S2Session | None,
    *,
    language: Language,
    vad: Vad,
) -> Iterator[Update]:
    """An S2 session from its one stream: the explanation and each kept answer (P3-e)."""

    def inner() -> Iterator[Update]:
        yield Update(ui_text(lang, "practice.preparing"), busy=True)
        if recording is None or session is None:
            yield Update(ui_text(lang, "practice.no_audio"))
            return
        explanation, answers = session_inputs(
            recording.samples, session, preroll_s=analyzer.settings.app.answer_preroll_s, vad=vad
        )
        notices: list[str] = []
        if not recording.complete:
            notices.append(
                ui_text(
                    lang, "practice.incomplete", received=recording.received_s,
                    wall=recording.wall_s,
                )
            )  # fmt: skip
        if session.explanation_hard_stopped:
            hard = round(session.explanation_limit_s + session.extra_s)
            notices.append(ui_text(lang, "practice.s2.hard_stopped", hard=hard))
        yield from _run(
            ctx, analyzer, lang, language=language, scenario=s2_page.SCENARIO, source="app",
            audio=explanation, duration_s=recording.received_s,
            limit_s=session.explanation_limit_s, notices=notices,
            completeness=recording.ratio, answers=answers,
        )  # fmt: skip

    return _guarded(lang, inner())


def run_upload(
    ctx: AppContext,
    analyzer: app_run.Analyzer,
    lang: Language,
    path: str | None,
    *,
    language: Language,
    scenario: str,
) -> Iterator[Update]:
    def inner() -> Iterator[Update]:
        yield Update(ui_text(lang, "practice.preparing"), busy=True)
        if not path:
            yield Update(ui_text(lang, "practice.no_file"))
            return
        check = app_run.check_upload(Path(path), analyzer.settings.app)
        if check.audio is None:
            key = check.reason or "practice.upload_unreadable"
            yield Update(ui_text(lang, key, **dict(check.values)))
            return
        yield from _run(
            ctx, analyzer, lang, language=language, scenario=scenario, source="upload",
            audio=check.audio, duration_s=check.duration_s or 0.0, limit_s=None, notices=[],
        )  # fmt: skip

    return _guarded(lang, inner())


def _last(updates: Iterator[Update]) -> Update:
    final = Update("")
    for update in updates:
        final = update
    return final


def analyse_recording(
    ctx: AppContext,
    analyzer: app_run.Analyzer,
    lang: Language,
    recording: Recording | None,
    *,
    language: Language,
    scenario: str,
    extra_s: float,
) -> Outcome:
    """The final state of `run_recording` (tests and callers without progress)."""
    return _last(
        run_recording(
            ctx, analyzer, lang, recording, language=language, scenario=scenario, extra_s=extra_s
        )
    )


def analyse_upload(
    ctx: AppContext,
    analyzer: app_run.Analyzer,
    lang: Language,
    path: str | None,
    *,
    language: Language,
    scenario: str,
) -> Outcome:
    return _last(run_upload(ctx, analyzer, lang, path, language=language, scenario=scenario))


def elapsed_text(buffer: RecordingBuffer, lang: Language) -> str:
    if not buffer.active:
        return ui_text(lang, "practice.elapsed_idle")
    clock = report_page.clock
    return ui_text(
        lang, "practice.elapsed", clock=clock(buffer.elapsed_s()), limit=clock(buffer.limit_s)
    )


def make_handler(buffer: RecordingBuffer) -> Any:
    """A FastRTC handler that keeps microphone frames in `buffer` (send-only: nothing back).

    FastRTC makes one copy per microphone connection. The take starts with that connection's
    first frame, not in `start_up`, which FastRTC runs in a thread at the same time as the first
    frames (a reset there could drop them). `shutdown` (the microphone box stopped) keeps the
    take for "Stop and analyse"."""
    from fastrtc import StreamHandler

    class _Recorder(StreamHandler):
        def __init__(self) -> None:
            super().__init__(input_sample_rate=INPUT_RATE)
            self._started = False

        def start_up(self) -> None:
            return None

        def receive(self, frame: tuple[int, Any]) -> None:
            if not self._started:
                buffer.start()
                self._started = True
            rate, samples = frame
            buffer.add(rate, samples)

        def emit(self) -> None:
            return None

        def shutdown(self) -> None:
            buffer.end_stream()

        def copy(self) -> _Recorder:
            return _Recorder()

    return _Recorder()


def build(ctx: AppContext, lang: Language, analyzer: app_run.Analyzer) -> None:
    import gradio as gr
    from fastrtc import WebRTC

    rtl = lang == "ar"
    app_cfg = analyzer.settings.app
    if not ctx.key_ok:
        gr.Markdown(ui_text(lang, "practice.key_missing"), rtl=rtl)
    scenario = gr.Dropdown(
        choices=[(ui_text(lang, f"data.scenarios.{s}"), s) for s in SCENARIOS],
        value=SCENARIOS[0],
        label=ui_text(lang, "practice.scenario"),
    )
    language = gr.Radio(
        choices=[(ui_text(lang, f"data.languages.{x}"), x) for x in ("en", "ar")],
        value=lang,
        label=ui_text(lang, "practice.language"),
    )
    limit = scenario_limit_s(SCENARIOS[0])
    buffer = RecordingBuffer(
        limit_s=limit, extra_s=app_cfg.recording_extra_s, min_ratio=app_cfg.completeness_min_ratio
    )
    hard = round(limit + app_cfg.recording_extra_s)
    limit_note = gr.Markdown(
        ui_text(lang, "practice.limit_note", limit=round(limit), hard=hard), rtl=rtl
    )

    def vad(samples: Any) -> Any:
        from presenceai.speech.vad import detect_speech

        return detect_speech(samples, analyzer.settings.speech.vad)

    s2 = s2_page.S2Controller(
        buffer, explanation_limit_s=scenario_limit_s(s2_page.SCENARIO),
        extra_s=app_cfg.recording_extra_s, window_s=app_cfg.answer_window_s, vad=vad,
        preroll_s=app_cfg.answer_preroll_s,
    )  # fmt: skip
    with gr.Group(visible=False) as s2_box:
        gr.Markdown(ui_text(lang, "practice.s2.own_help"), rtl=rtl)
        own = [
            gr.Textbox(label=ui_text(lang, "practice.s2.own_question", n=i), rtl=rtl)
            for i in (1, 2)
        ]

    gr.Markdown(f"### {ui_text(lang, 'practice.record_heading')}", rtl=rtl)
    gr.Markdown(ui_text(lang, "practice.record_help"), rtl=rtl)
    mic = WebRTC(
        label=ui_text(lang, "practice.microphone"),
        modality="audio",
        mode="send",
        full_screen=False,  # the default full-screen layer swallows every click on the page
        rtc_configuration=NO_ICE,
        server_rtc_configuration=NO_ICE,
    )
    # one stream limit for both scenarios: S1's hard stop or a whole S2 session, retries included
    probe = S2Session(("", ""), rate=INPUT_RATE, explanation_limit_s=s2.explanation_limit_s,
                      extra_s=s2.extra_s, window_s=s2.window_s)  # fmt: skip
    stream_limit = max(hard, round(probe.stream_limit_s))
    mic.stream(make_handler(buffer), inputs=[mic], outputs=[mic], time_limit=stream_limit)
    elapsed = gr.Markdown(elapsed_text(buffer, lang), rtl=rtl)
    with gr.Group(visible=False) as s2_live:
        s2_question = gr.Markdown(rtl=rtl)
        s2_status = gr.Markdown(s2_page.view(None, 0, lang).status, rtl=rtl)
        with gr.Row():
            next_btn = gr.Button(ui_text(lang, "practice.s2.next"), interactive=False)
            done_btn = gr.Button(ui_text(lang, "practice.s2.done_answering"), interactive=False)
            retry_btn = gr.Button(ui_text(lang, "practice.s2.retry"), interactive=False)
            keep_btn = gr.Button(ui_text(lang, "practice.s2.keep"), interactive=False)
    timer = gr.Timer(1.0, active=False)
    s2_outputs = [s2_status, s2_question, next_btn, done_btn, retry_btn, keep_btn]

    def s2_render() -> tuple[Any, ...]:
        v = s2.view(lang)
        question = report_page.escape(v.question).replace("\\\n", "\n") if v.question else ""
        return (
            v.status, question, gr.Button(interactive=v.next_on),
            gr.Button(interactive=v.done_on), gr.Button(interactive=v.retry_on),
            gr.Button(interactive=v.retry_on),
        )  # fmt: skip

    def on_tick() -> tuple[Any, ...]:
        # no inputs: the timer event sends none, and an input here made the browser warn
        # "Too many arguments" on every tick; the scenario is known from the session
        if s2.session is None:
            return (elapsed_text(buffer, lang), *[gr.skip() for _ in s2_outputs])
        s2.tick()
        return ("", *s2_render())  # S2 shows its own timers, not the stream's limit

    def on_start(sc: str, lg: Language, q1: str, q2: str) -> tuple[Any, ...]:
        if sc == s2_page.SCENARIO:
            s2.start(lg, [q1, q2], INPUT_RATE)
        else:
            s2.session = None
            buffer.set_limits(limit_s=limit, extra_s=app_cfg.recording_extra_s)
        return (gr.Timer(active=True), *s2_render())

    def on_scenario(sc: str) -> tuple[Any, ...]:
        is_s2 = sc == s2_page.SCENARIO
        if is_s2:
            s2_limit = scenario_limit_s(s2_page.SCENARIO)
            s2_hard = round(s2_limit + app_cfg.recording_extra_s)
            note = ui_text(lang, "practice.limit_note", limit=round(s2_limit), hard=s2_hard)
        else:
            note = ui_text(lang, "practice.limit_note", limit=round(limit), hard=hard)
        return note, gr.Group(visible=is_s2), gr.Group(visible=is_s2)

    def s2_action(action: str) -> Any:
        def act() -> tuple[Any, ...]:
            getattr(s2, action)()
            return s2_render()

        return act

    timer.tick(on_tick, None, [elapsed, *s2_outputs], queue=False, show_progress="hidden")
    mic.start_recording(on_start, [scenario, language, *own], [timer, *s2_outputs], queue=False)
    mic.stop_recording(
        lambda: (gr.Timer(active=False), elapsed_text(buffer, lang)),
        outputs=[timer, elapsed],
        queue=False,
    )
    scenario.change(on_scenario, [scenario], [limit_note, s2_box, s2_live], queue=False)
    for btn, action in ((next_btn, "next_question"), (done_btn, "done_answering"),
                        (retry_btn, "retry"), (keep_btn, "keep_silent")):  # fmt: skip
        btn.click(s2_action(action), None, s2_outputs, queue=False, show_progress="hidden")
    stop = gr.Button(
        ui_text(lang, "practice.stop_analyse"), variant="primary", interactive=ctx.key_ok
    )

    gr.Markdown(f"### {ui_text(lang, 'practice.upload_heading')}", rtl=rtl)
    minutes = round(app_cfg.upload_max_audio_s / 60)
    gr.Markdown(
        ui_text(lang, "practice.upload_help", mb=round(app_cfg.upload_max_mb), minutes=minutes),
        rtl=rtl,
    )
    upload = gr.File(
        label=ui_text(lang, "practice.upload"),
        file_types=[*app_run.ACCEPTED_SUFFIXES, ".wma"],  # .wma is selectable, then explained
        type="filepath",
    )
    analyse = gr.Button(ui_text(lang, "practice.analyse_upload"), interactive=ctx.key_ok)

    status = gr.Markdown(rtl=rtl)
    result = gr.Markdown(rtl=rtl)
    with gr.Accordion(ui_text(lang, "practice.transcript"), open=False):
        transcript = gr.Textbox(show_label=False, interactive=False, lines=8, rtl=rtl)

    def show(u: Update) -> tuple[Any, ...]:
        enabled = ctx.key_ok and not u.busy
        return (
            u.status, u.report, u.transcript,
            gr.Button(interactive=enabled), gr.Button(interactive=enabled),
        )  # fmt: skip

    def on_stop(sc: str, lg: Language) -> Iterator[tuple[Any, ...]]:
        recording = buffer.finish()
        if sc == s2_page.SCENARIO:
            session, s2.session = s2.session, None
            updates = run_s2(ctx, analyzer, lang, recording, session, language=lg, vad=vad)
        else:
            updates = run_recording(
                ctx, analyzer, lang, recording, language=lg, scenario=sc,
                extra_s=app_cfg.recording_extra_s,
            )  # fmt: skip
        for u in updates:
            yield (*show(u), gr.Timer(active=False))

    def on_upload(path: str | None, sc: str, lg: Language) -> Iterator[tuple[Any, ...]]:
        for u in run_upload(ctx, analyzer, lang, path, language=lg, scenario=sc):
            yield (*show(u), None if not u.busy else gr.File())

    def pressed() -> tuple[Any, ...]:
        # instant feedback, outside the queue, before Gradio processes the inputs (an upload
        # is copied first, which took ~3 s in the browser test)
        return show(Update(ui_text(lang, "practice.preparing"), busy=True))

    first = [status, result, transcript, stop, analyse]
    stop.click(pressed, None, first, queue=False, show_progress="hidden").then(
        on_stop, [scenario, language], [*first, timer],
        concurrency_limit=1, show_progress="hidden",
    )  # fmt: skip
    analyse.click(pressed, None, first, queue=False, show_progress="hidden").then(
        on_upload, [upload, scenario, language], [*first, upload],
        concurrency_limit=1, show_progress="hidden",
    )  # fmt: skip
