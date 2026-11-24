"""Practice page and report rendering (P3-d). Fakes only: no models, no microphone, no browser."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from presenceai.live.recorder import Recording
from presenceai.pipeline import app_run
from presenceai.schemas.storage import Consent, DimensionSummary, SessionSummary
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
    for f in fake_assessment.assessment.findings:
        assert f"based on {f.measured} of {f.total} measures" in md
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
