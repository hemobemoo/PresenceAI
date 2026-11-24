"""Practice page (P3-d): record in the app (FastRTC, audio only) or upload a file, then analyse.

- **Recording:** FastRTC `WebRTC(modality="audio", mode="send")`, frames resampled to 16 kHz
  and kept in memory (`live/recorder.py`). **No STUN or TURN server**: `{"iceServers": []}` on
  the browser and on the server, so the connection stays on this computer (aiortc would
  otherwise contact stun.l.google.com; owner, P3 plan). A timer shows the elapsed time; frames
  stop being kept at the limit + `app.recording_extra_s`; a recording whose received audio is
  below `app.completeness_min_ratio` of its wall-clock time gets a warning.
- **Upload:** read into memory, Gradio's temp copy deleted, limits checked
  (`pipeline/app_run.py: check_upload`). Uploads have no time limit (overrun is missing).
- Either way the audio is encrypted before it is stored, analysed from memory, and the report is
  shown with its N-of-M and capped badges (`ui/pages/report.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from presenceai.evalconfig import REPO_ROOT
from presenceai.lang import Language, ui_text
from presenceai.live.recorder import Recording, RecordingBuffer
from presenceai.pipeline import app_run
from presenceai.schemas.storage import MediaSource
from presenceai.speech.audio import MemoryAudio
from presenceai.storage import media, reports
from presenceai.ui.context import AppContext
from presenceai.ui.pages import report as report_page

SCENARIOS = ("s1_presentation",)  # S2 comes with its follow-up flow (P3-e)
NO_ICE: dict[str, Any] = {"iceServers": []}  # no STUN/TURN in Stage A
INPUT_RATE = 16_000


def scenario_limit_s(scenario: str) -> float:
    """The scenario's time limit: the upper end of `duration_s` in scenarios/<id>.yaml."""
    data = yaml.safe_load((REPO_ROOT / "scenarios" / f"{scenario}.yaml").read_text("utf-8"))
    return float(data["duration_s"][-1])


@dataclass(frozen=True)
class Outcome:
    status: str  # Markdown: notices and errors, in the page language
    report: str = ""  # Markdown report
    transcript: str = ""  # plain text


def _keep_days(ctx: AppContext) -> int | None:
    return ctx.raw_media_days if ctx.consent().keep_recordings else None


def _analyse(
    ctx: AppContext,
    analyzer: app_run.Analyzer,
    lang: Language,
    *,
    language: Language,
    scenario: str,
    source: MediaSource,
    audio: MemoryAudio,
    limit_s: float | None,
    notices: list[str],
) -> Outcome:
    if ctx.store is None:
        return Outcome(ui_text(lang, "practice.key_missing"))
    if analyzer.busy():
        return Outcome(ui_text(lang, "practice.busy"))
    keep = _keep_days(ctx)
    session = ctx.db.new_session(scenario, language, source, now=datetime.now(UTC), keep_days=keep)
    ctx.store.write_bytes(session.id, audio.name, audio.data)  # encrypted at write time
    try:
        analyzer.run(ctx.db, ctx.store, session, audio, limit_s=limit_s, keep=keep is not None)
    except app_run.AnalysisFailed:
        return Outcome("\n\n".join([*notices, ui_text(lang, "practice.failed")]))
    view = reports.open_report(ctx.db, ctx.store, session.id, lang=lang)
    status = "\n\n".join([*notices, ui_text(lang, "practice.done")])
    return Outcome(
        status, report_page.render(view, lang, scenario), report_page.transcript_text(view)
    )


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
    if recording is None:
        return Outcome(ui_text(lang, "practice.no_audio"))
    limit = scenario_limit_s(scenario)
    notices: list[str] = []
    if not recording.complete:
        notices.append(
            ui_text(
                lang, "practice.incomplete", received=recording.received_s, wall=recording.wall_s
            )
        )
    if recording.hard_stopped:
        notices.append(
            ui_text(
                lang, "practice.hard_stopped", hard=round(limit + extra_s), extra=round(extra_s)
            )
        )
    audio = MemoryAudio("talk.wav", media.wav_bytes(recording.samples, recording.sample_rate))
    return _analyse(
        ctx, analyzer, lang, language=language, scenario=scenario, source="app", audio=audio,
        limit_s=limit, notices=notices,
    )  # fmt: skip


def analyse_upload(
    ctx: AppContext,
    analyzer: app_run.Analyzer,
    lang: Language,
    path: str | None,
    *,
    language: Language,
    scenario: str,
) -> Outcome:
    if not path:
        return Outcome(ui_text(lang, "practice.no_file"))
    check = app_run.check_upload(Path(path), analyzer.settings.app)
    if check.audio is None:
        return Outcome(
            ui_text(lang, check.reason or "practice.upload_unreadable", **dict(check.values))
        )
    return _analyse(
        ctx, analyzer, lang, language=language, scenario=scenario, source="upload",
        audio=check.audio, limit_s=None, notices=[],
    )  # fmt: skip


def elapsed_text(buffer: RecordingBuffer, lang: Language) -> str:
    if not buffer.active:
        return ui_text(lang, "practice.elapsed_idle")
    clock = report_page.clock
    return ui_text(
        lang, "practice.elapsed", clock=clock(buffer.elapsed_s()), limit=clock(buffer.limit_s)
    )


def make_handler(buffer: RecordingBuffer) -> Any:
    """A FastRTC handler that keeps microphone frames in `buffer` (send-only: nothing back)."""
    from fastrtc import StreamHandler

    class _Recorder(StreamHandler):
        def __init__(self) -> None:
            super().__init__(input_sample_rate=INPUT_RATE)

        def start_up(self) -> None:
            buffer.start()

        def receive(self, frame: tuple[int, Any]) -> None:
            rate, samples = frame
            buffer.add(rate, samples)

        def emit(self) -> None:
            return None

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
    gr.Markdown(ui_text(lang, "practice.limit_note", limit=round(limit), hard=hard), rtl=rtl)

    gr.Markdown(f"### {ui_text(lang, 'practice.record_heading')}", rtl=rtl)
    gr.Markdown(ui_text(lang, "practice.record_help"), rtl=rtl)
    mic = WebRTC(
        label=ui_text(lang, "practice.microphone"),
        modality="audio",
        mode="send",
        rtc_configuration=NO_ICE,
        server_rtc_configuration=NO_ICE,
    )
    mic.stream(make_handler(buffer), inputs=[mic], outputs=[mic], time_limit=hard)
    elapsed = gr.Markdown(elapsed_text(buffer, lang), rtl=rtl)
    gr.Timer(0.5).tick(
        lambda: elapsed_text(buffer, lang), outputs=[elapsed], show_progress="hidden"
    )
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
        file_types=list(app_run.ACCEPTED_SUFFIXES),
        type="filepath",
    )
    analyse = gr.Button(ui_text(lang, "practice.analyse_upload"), interactive=ctx.key_ok)

    status = gr.Markdown(rtl=rtl)
    result = gr.Markdown(rtl=rtl)
    with gr.Accordion(ui_text(lang, "practice.transcript"), open=False):
        transcript = gr.Textbox(show_label=False, interactive=False, lines=8, rtl=rtl)

    def on_stop(sc: str, lg: Language) -> tuple[str, str, str]:
        o = analyse_recording(
            ctx, analyzer, lang, buffer.finish(), language=lg, scenario=sc,
            extra_s=app_cfg.recording_extra_s,
        )  # fmt: skip
        return o.status, o.report, o.transcript

    def on_upload(path: str | None, sc: str, lg: Language) -> tuple[str, str, str, None]:
        o = analyse_upload(ctx, analyzer, lang, path, language=lg, scenario=sc)
        return o.status, o.report, o.transcript, None  # clear the file box

    stop.click(on_stop, [scenario, language], [status, result, transcript], concurrency_limit=1)
    analyse.click(
        on_upload, [upload, scenario, language], [status, result, transcript, upload],
        concurrency_limit=1,
    )  # fmt: skip
