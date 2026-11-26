"""Practice page and report rendering (P3-d). Fakes only: no models, no microphone, no browser."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from presenceai.live.recorder import Recording
from presenceai.pipeline import app_run
from presenceai.schemas.storage import Consent, DimensionSummary, MeasureValue, SessionSummary
from presenceai.settings import Settings
from presenceai.storage import media, reports
from presenceai.storage.reports import ReportView
from presenceai.ui.context import AppContext
from presenceai.ui.pages import practice
from presenceai.ui.pages import report as report_page

T0 = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def ctx(tmp_path: Path):
    c = AppContext.open(
        tmp_path / "home", SecretStr(Fernet.generate_key().decode()), raw_media_days=30, now=T0
    )
    c.db.set_consent(Consent(use_app=True), now=T0)
    yield c
    c.close()


@pytest.fixture
def analyzer(fake_assessment: Any, fake_parser: Any) -> app_run.Analyzer:
    return app_run.Analyzer(
        Settings(_env_file=None),  # pyright: ignore[reportCallIssue]
        load_whisper=lambda name, options: object(),
        load_parser=lambda: fake_parser,
        analyze=lambda audio, **_: fake_assessment.speech,
    )


def _recording(*, complete: bool = True, hard: bool = False) -> Recording:
    samples = np.zeros(16_000, dtype=np.float32)
    return Recording(
        samples=samples, sample_rate=16_000, received_s=1.0, wall_s=1.0 if complete else 1.5,
        ratio=1.0 if complete else 0.67, complete=complete, hard_stopped=hard,
    )  # fmt: skip


# --- report rendering -------------------------------------------------------------------------


def test_report_cards_show_level_n_of_m_confidence_and_evidence(ctx: AppContext,
                                                                fake_assessment: Any):  # fmt: skip
    assert ctx.store is not None
    s = ctx.db.new_session("s1_presentation", "en", "app", now=T0, keep_days=None)
    reports.save(ctx.db, ctx.store, s.id, fake_assessment)
    md = report_page.render(reports.open_report(ctx.db, ctx.store, s.id, lang="en"), "en",
                            "s1_presentation")  # fmt: skip
    assert "### Delivery" in md and "### Central message" in md
    info = {r.dimension for r in fake_assessment.assessment.dimensions if r.information_only}
    assert info == {"central_message"}  # ADR 0025
    for f in fake_assessment.assessment.findings:
        if f.dimension not in info:
            assert f"based on {f.measured} of {f.total} measures" in md
    assert "Levels based on 4 of 5 dimensions" in md
    assert r"Not scored: Central message \(information only\)" in md
    assert "**Information only, not scored**" in md
    assert "Level " in md and "confidence: " in md
    assert "zebra lanterns glow" in md and "(00:" in md  # a quote and its time
    assert md.index("### Central message") < md.index("### Delivery")  # rubric order


def test_capped_badge_and_text_unavailable_fallback():
    summary = SessionSummary(
        session_id="0" * 32,
        dimensions=(
            DimensionSummary(dimension="delivery", level=3, capped_from=4, score=2 / 3,
                             confidence="low", measured=1, total=3),
            DimensionSummary(dimension="language", measured=0, total=2),
            DimensionSummary(dimension="organization", measured=0, total=2, failed=True),
        ),
    )  # fmt: skip
    view = ReportView(summary=summary, text_note="Report text unavailable: test.")
    md = report_page.render(view, "en", "s1_presentation")
    assert "Report text unavailable" in md
    assert "`limited to level 3 (reached 4)`" in md and "`based on 1 of 3 measures`" in md
    assert "**Not measurable**" in md and "**Could not be checked**" in md
    ar = report_page.render(view, "ar", "s1_presentation")
    assert "بناءً على 1 من 3 مقاييس" in ar and "الإلقاء" in ar


def test_user_words_cannot_add_markdown():
    risky = "see [here](http://203.0.113.5/x) and ![img](http://203.0.113.5/i.png) **bold**"
    out = report_page.escape(risky)
    assert "](http" not in out and "![" not in out and "**" not in out
    assert report_page.clock(65.4) == "01:05"


# --- practice handlers ------------------------------------------------------------------------


def test_m4a_upload_end_to_end_with_fakes(ctx: AppContext, analyzer: Any, m4a_file: Path):
    o = practice.analyse_upload(ctx, analyzer, "en", str(m4a_file), language="en",
                                scenario="s1_presentation")  # fmt: skip
    assert "Analysis finished." in o.status and "### Delivery" in o.report
    assert "zebra lanterns glow" in o.transcript
    (s,) = ctx.db.list_sessions()
    assert s.source == "upload" and s.status == "analysed"
    assert ctx.store is not None and ctx.store.media_names(s.id) == ()  # no keep consent
    assert not m4a_file.exists()
    over = {m.measure for m in ctx.db.summary(s.id).measures}
    assert "timing.overrun_s" not in over  # an upload has no time limit


def test_recording_with_keep_consent_is_kept_encrypted_with_notices(ctx: AppContext,
                                                                    analyzer: Any):  # fmt: skip
    ctx.db.set_consent(Consent(use_app=True, keep_recordings=True), now=T0)
    rec = _recording(complete=False, hard=True)
    o = practice.analyse_recording(
        ctx, analyzer, "en", rec, language="en", scenario="s1_presentation", extra_s=60
    )
    assert "looks incomplete" in o.status and "stopped at 240 s" in o.status
    (s,) = ctx.db.list_sessions()
    assert s.source == "app" and s.media_expires_at is not None
    assert ctx.store is not None and ctx.store.media_names(s.id) == ("talk.wav",)
    raw = (ctx.layout.media_dir(s.id) / "talk.wav.enc").read_bytes()
    assert b"RIFF" not in raw


def test_practice_refusals(ctx: AppContext, analyzer: Any, tmp_path: Path):
    o = practice.analyse_recording(ctx, analyzer, "en", None, language="en",
                                   scenario="s1_presentation", extra_s=60)  # fmt: skip
    assert o.status == "No audio arrived. Check the microphone and record again."
    o = practice.analyse_upload(ctx, analyzer, "ar", None, language="ar",
                                scenario="s1_presentation")  # fmt: skip
    assert o.status == "اختر ملفًا أولًا."
    big = tmp_path / "long.wav"
    big.write_bytes(media.wav_bytes(np.zeros(16_000 * 2, dtype=np.float32), 16_000))
    analyzer.settings = analyzer.settings.model_copy(
        update={"app": analyzer.settings.app.model_copy(update={"upload_max_audio_s": 1.0})}
    )
    o = practice.analyse_upload(ctx, analyzer, "en", str(big), language="en",
                                scenario="s1_presentation")  # fmt: skip
    assert o.status.startswith("The recording is longer than")
    assert ctx.db.list_sessions() == ()  # nothing stored for a refused upload


def test_scenario_limit_and_timer_text():
    assert practice.scenario_limit_s("s1_presentation") == 180.0
    from presenceai.live.recorder import RecordingBuffer

    b = RecordingBuffer(limit_s=180, extra_s=60, min_ratio=0.95, clock=lambda: 10.0)
    assert practice.elapsed_text(b, "en") == "Not recording."
    b.start()
    b.add(16_000, np.zeros(1600, dtype=np.int16))
    assert practice.elapsed_text(b, "en") == "Elapsed: 00:00 of 03:00"


# --- round-1 fixes (owner's real S1 test, 2026-11-25) -----------------------------------------


def _frames(handler: Any, n: int) -> None:
    for _ in range(n):
        handler.receive((16_000, np.full((1, 1600), 500, dtype=np.int16)))


def test_the_take_survives_stopping_the_microphone_box():
    from presenceai.live.recorder import RecordingBuffer

    buffer = RecordingBuffer(limit_s=180, extra_s=60, min_ratio=0.95)
    template = practice.make_handler(buffer)
    handler = template.copy()  # FastRTC uses one copy per microphone connection
    _frames(handler, 10)
    handler.start_up()  # FastRTC runs this alongside the first frames: it must not reset them
    _frames(handler, 10)
    handler.shutdown()  # the user stops the microphone box
    assert not buffer.active and buffer.has_audio()
    assert practice.elapsed_text(buffer, "en") == "Not recording."  # the timer stops
    rec = buffer.finish()  # later: "Stop and analyse"
    assert rec is not None and rec.received_s == pytest.approx(2.0)
    second = template.copy()  # a new take starts fresh with its own first frame
    _frames(second, 5)
    rec2 = buffer.finish()
    assert rec2 is not None and rec2.received_s == pytest.approx(0.5)


def test_pressing_analyse_shows_progress_then_a_report(ctx: AppContext, analyzer: Any,
                                                       m4a_file: Path):  # fmt: skip
    updates = list(practice.run_upload(ctx, analyzer, "en", str(m4a_file), language="en",
                                       scenario="s1_presentation"))  # fmt: skip
    assert updates[0].busy and updates[0].status == "Preparing the audio…"
    steps = [u.status for u in updates if u.busy and "step" in u.status]
    assert [s.split(":")[0] for s in steps] == [f"Analysing… step {i} of 4" for i in (1, 2, 3, 4)]
    final = updates[-1]
    assert (
        not final.busy and "### Delivery" in final.report and "Analysis finished." in final.status
    )


@pytest.mark.parametrize("where", ["analysis", "storage", "check"])
def test_pressing_analyse_always_ends_in_a_visible_message(
    ctx: AppContext, analyzer: Any, m4a_file: Path, monkeypatch: pytest.MonkeyPatch, where: str
):
    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("zebra lanterns glow")  # text that must never reach the page or log

    if where == "analysis":
        analyzer.analyze = boom
    elif where == "storage":
        assert ctx.store is not None
        monkeypatch.setattr(ctx.store, "write_bytes", boom)
    else:
        monkeypatch.setattr(practice.app_run, "check_upload", boom)
    updates = list(practice.run_upload(ctx, analyzer, "en", str(m4a_file), language="en",
                                       scenario="s1_presentation"))  # fmt: skip
    final = updates[-1]
    assert not final.busy and final.report == ""
    assert final.status.startswith("The analysis could not finish (")
    assert "RuntimeError" in final.status and "zebra" not in final.status


def test_a_second_analysis_while_one_runs_is_told_to_wait(ctx: AppContext, analyzer: Any):
    analyzer._lock.acquire()
    try:
        out = practice.analyse_recording(ctx, analyzer, "en", _recording(), language="en",
                                         scenario="s1_presentation", extra_s=60)  # fmt: skip
    finally:
        analyzer._lock.release()
    assert out.status.startswith("Another analysis is running") and not out.busy


def test_analysis_logs_one_start_and_one_end_line_without_text(
    ctx: AppContext, analyzer: Any, m4a_file: Path, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level("INFO", logger="presenceai"):
        practice.analyse_upload(ctx, analyzer, "en", str(m4a_file), language="en",
                                scenario="s1_presentation")  # fmt: skip
    lines = [r.getMessage() for r in caplog.records if r.name.startswith("presenceai")]
    (start,) = [m for m in lines if m.startswith("analysis started")]
    (end,) = [m for m in lines if m.startswith("analysis finished")]
    (s,) = ctx.db.list_sessions()
    assert f"session={s.id}" in start and "source=upload" in start and "audio_s=2.0" in start
    for step in ("load_models", "speech", "content", "save"):
        assert step in end
    assert "total_s=" in end
    assert not any("zebra" in m or "lanterns" in m for m in lines)


def test_wma_gets_its_own_message(ctx: AppContext, analyzer: Any, tmp_path: Path):
    wma = tmp_path / "Recording.wma"
    wma.write_bytes(b"0&\xb2u")  # an ASF header start
    out = practice.analyse_upload(ctx, analyzer, "en", str(wma), language="en",
                                  scenario="s1_presentation")  # fmt: skip
    assert out.status.startswith("WMA files are not accepted.") and not wma.exists()
    assert ctx.db.list_sessions() == ()


def test_mic_box_not_full_screen_and_timer_skips_the_queue(ctx: AppContext, analyzer: Any):
    from fastrtc import WebRTC

    from presenceai.ui.app import build_blocks

    blocks = build_blocks(ctx, "en", analyzer)
    (mic,) = [b for b in blocks.blocks.values() if isinstance(b, WebRTC)]
    assert mic.full_screen is False  # the full-screen layer swallowed every click (round 1)
    import gradio as gr

    timers = {i for i, b in blocks.blocks.items() if isinstance(b, gr.Timer)}
    ticks = [d for d in blocks.fns.values() if any(t[0] in timers for t in d.targets)]
    assert ticks and all(d.queue is False for d in ticks)
    assert all(not blocks.blocks[i].active for i in timers)  # only while recording


def test_windows_connection_resets_are_dropped_and_other_errors_kept():
    from presenceai.ui.app import ignore_connection_resets

    seen: list[dict[str, Any]] = []

    class Loop:
        def default_exception_handler(self, context: dict[str, Any]) -> None:
            seen.append(context)

    reset = ConnectionResetError(10054, "forcibly closed")
    reset.winerror = 10054  # type: ignore[attr-defined]  (set on Windows by the OS error)
    ignore_connection_resets(Loop(), {"exception": reset})  # type: ignore[arg-type]
    assert seen == []
    other = {"exception": ConnectionResetError(104, "reset by peer")}
    ignore_connection_resets(Loop(), other)  # type: ignore[arg-type]
    ignore_connection_resets(Loop(), {"message": "Task was destroyed"})  # type: ignore[arg-type]
    assert len(seen) == 2


# --- report cards after the owner's real S1 runs (2026-11-25) ---------------------------------


def test_long_quotes_are_shortened_on_the_card_only(ctx: AppContext, fake_assessment: Any):
    long = " ".join(f"w{i}" for i in range(60))
    assert report_page.shorten(long) == " ".join(f"w{i}" for i in range(25)) + " …"
    assert report_page.shorten("a short quote") == "a short quote"
    assert ctx.store is not None
    s = ctx.db.new_session("s1_presentation", "en", "app", now=T0, keep_days=None)
    reports.save(ctx.db, ctx.store, s.id, fake_assessment)
    view = reports.open_report(ctx.db, ctx.store, s.id, lang="en")
    md = report_page.render(view, "en", "s1_presentation")
    quotes = [e.transcript.text for f in fake_assessment.assessment.findings
              for e in f.evidence if e.transcript is not None]  # fmt: skip
    longest = max(quotes, key=lambda q: len(q.split()))
    assert len(longest.split()) > 25  # the fake talk has a quote long enough to shorten
    assert report_page.escape(longest) not in md and " …”" in md
    assert view.report is not None  # the full quote stays in the report data (and the export)
    kept = [e.transcript.text for f in view.report.assessment.findings
            for e in f.evidence if e.transcript is not None]  # fmt: skip
    assert longest in kept


def test_cards_name_unmeasured_measures_and_why(ctx: AppContext, analyzer: Any, m4a_file: Path):
    second = m4a_file.with_name("again.m4a")
    second.write_bytes(m4a_file.read_bytes())  # each upload's temp copy is deleted after reading
    out = practice.analyse_upload(ctx, analyzer, "en", str(m4a_file), language="en",
                                  scenario="s1_presentation")  # fmt: skip
    md = out.report
    assert r"Not measured: seconds over the time limit \(uploaded file: no time limit\)" in md
    assert r"Not counted \(camera off\): share of speaking time looking toward the camera" in md
    ar = practice.analyse_upload(ctx, analyzer, "ar", str(second), language="en",
                                 scenario="s1_presentation")  # fmt: skip
    assert "لم يُقَس" in ar.report and r"لم يُحسب \(الكاميرا متوقفة\)" in ar.report


def _cm_view(*, information_only: bool = True) -> ReportView:
    return ReportView(summary=SessionSummary(
        session_id="0" * 32,
        dimensions=(
            DimensionSummary(dimension="central_message", measured=0, total=4,
                             information_only=information_only),
            DimensionSummary(dimension="delivery", level=3, score=2 / 3, confidence="low",
                             measured=2, total=3),
        ),
        measures=(MeasureValue(measure="text.central_message.possible_main_point", value=0),),
    ))  # fmt: skip


@pytest.mark.parametrize("lang", ["en", "ar"])
def test_the_level_1_hint_is_gone_and_the_card_never_says_no_main_point(
    lang: Literal["en", "ar"],
):
    md = report_page.render(_cm_view(), lang, "s1_presentation")
    for wording in ("No sentence stating", "main point was found", "Try adding one sentence",
                    "لم يُعثر على جملة", "جرّب إضافة جملة"):  # fmt: skip
        assert wording not in md
    badge = "Information only, not scored" if lang == "en" else "للاطلاع فقط، لا يدخل في التقييم"
    assert f"**{badge}**" in md
    line = ("Levels based on 1 of 2 dimensions" if lang == "en"
            else "المستويات مبنية على 1 من 2 أبعاد")  # fmt: skip
    assert report_page.escape(line) in md


def test_no_report_text_tells_the_user_no_main_point_was_found():
    """ADR 0025 item 3: never "no main point found", in any language pack."""
    from presenceai.lang import load_pack_file

    for lang in ("en", "ar"):
        for name in ("report.yaml", "ui.yaml"):
            dumped = str(load_pack_file(lang, name)).lower()
            for wording in ("no main point", "main point was found", "no sentence stating",
                            "لم يُعثر على جملة تذكر فكرتك"):  # fmt: skip
                assert wording not in dumped, (lang, name, wording)


def test_the_practice_page_has_the_main_point_tip():
    from presenceai.lang import ui_text

    assert "state your main point in one sentence" in ui_text("en", "practice.tip_main_point")
    assert "فكرتك الرئيسية" in ui_text("ar", "practice.tip_main_point")
    src = (Path(practice.__file__)).read_text("utf-8")
    assert '"practice.tip_main_point"' in src


def test_the_start_log_line_has_the_completeness_ratio(
    ctx: AppContext, analyzer: Any, caplog: pytest.LogCaptureFixture
):
    rec = _recording(complete=False)
    with caplog.at_level("INFO", logger="presenceai"):
        practice.analyse_recording(
            ctx, analyzer, "en", rec, language="en", scenario="s1_presentation", extra_s=60
        )
    (start,) = [r.getMessage() for r in caplog.records if "analysis started" in r.getMessage()]
    assert "source=app" in start and "completeness=0.670" in start
