"""Uploads and running an analysis in the app (P3-d; ADR 0023 item 2, ADR 0024). No models."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet

from presenceai.pipeline import app_run
from presenceai.settings import AppSettings, Settings
from presenceai.speech.audio import MemoryAudio
from presenceai.storage import media, reports
from presenceai.storage.db import Database
from presenceai.storage.layout import Layout

T0 = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


# --- uploads ----------------------------------------------------------------------------------


def test_m4a_upload_is_accepted_read_into_memory_and_the_temp_copy_deleted(m4a_file: Path):
    check = app_run.check_upload(m4a_file, AppSettings())
    assert check.reason is None and check.audio is not None
    assert check.audio.name == "upload.m4a" and check.audio.data[4:8] == b"ftyp"
    assert check.duration_s == pytest.approx(2.0, abs=0.1)
    assert not m4a_file.exists()


def test_upload_limits_are_from_the_config(m4a_file: Path, tmp_path: Path):
    data = m4a_file.read_bytes()
    check = app_run.check_upload(m4a_file, AppSettings(upload_max_audio_s=1.0))
    assert check.audio is None and check.reason == "practice.upload_too_long"
    assert not m4a_file.exists()
    big = tmp_path / "big.m4a"
    big.write_bytes(data)
    check = app_run.check_upload(big, AppSettings(upload_max_mb=0.001))
    assert check.reason == "practice.upload_too_big" and not big.exists()


def test_config_holds_the_owner_upload_and_recording_values():
    s = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert (s.app.upload_max_mb, s.app.upload_max_audio_s) == (50, 300)
    assert (s.app.recording_extra_s, s.app.completeness_min_ratio) == (60, 0.95)
    assert s.app.unload_models_after_analysis is True
    assert s.asr.cpu_threads == 4


def test_unreadable_or_unsupported_uploads_are_refused(tmp_path: Path):
    junk = tmp_path / "talk.m4a"
    junk.write_bytes(b"not audio at all")
    assert app_run.check_upload(junk, AppSettings()).reason == "practice.upload_unreadable"
    doc = tmp_path / "notes.pdf"
    doc.write_bytes(b"%PDF")
    assert app_run.check_upload(doc, AppSettings()).reason == "practice.upload_type"
    assert not junk.exists() and not doc.exists()


# --- the analyser -----------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path):
    layout = Layout(tmp_path / "home")
    db = Database(layout)
    store = media.MediaStore(layout, Fernet(Fernet.generate_key()))
    yield db, store
    db.close()


def _analyzer(fake_assessment: Any, fake_parser: Any) -> tuple[app_run.Analyzer, list[str]]:
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    loads: list[str] = []

    def load_whisper(name: str, options: Any) -> object:
        loads.append(name)
        assert options.cpu_threads == 4  # heat: ADR 0024
        return object()

    analyzer = app_run.Analyzer(
        settings,
        load_whisper=load_whisper,
        load_parser=lambda: fake_parser,
        analyze=lambda audio, **_: fake_assessment.speech,
    )
    return analyzer, loads


def test_analysis_saves_the_report_and_unloads_the_models(
    storage: Any, fake_assessment: Any, fake_parser: Any
):
    db, store = storage
    s = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=30)
    store.write_bytes(s.id, "talk.wav", b"x")
    analyzer, loads = _analyzer(fake_assessment, fake_parser)
    report = analyzer.run(db, store, s, MemoryAudio("talk.wav", b"x"), limit_s=180.0, keep=True)
    assert loads == ["whisper_small"] and not analyzer.models_loaded
    assert db.get_session(s.id).status == "analysed"
    view = reports.open_report(db, store, s.id, lang="en")
    assert view.report is not None and view.report.assessment == report.assessment
    assert store.media_names(s.id) == ("talk.wav",)  # kept: consent (c)
    assert "timing.overrun_s" in {m.measure for m in view.summary.measures}  # limit known


def test_without_keep_consent_the_recording_is_deleted_after_analysis(
    storage: Any, fake_assessment: Any, fake_parser: Any
):
    db, store = storage
    s = db.new_session("s1_presentation", "en", "upload", now=T0, keep_days=None)
    store.write_bytes(s.id, "upload.m4a", b"x")
    analyzer, _ = _analyzer(fake_assessment, fake_parser)
    analyzer.run(db, store, s, MemoryAudio("upload.m4a", b"x"), limit_s=None, keep=False, now=T0)
    assert store.media_names(s.id) == ()
    assert db.get_session(s.id).media_deleted_at == T0
    assert store.has_assessment(s.id)  # the encrypted report stays


def test_a_failed_analysis_marks_the_session_and_logs_no_user_text(
    storage: Any, fake_assessment: Any, fake_parser: Any, caplog: pytest.LogCaptureFixture
):
    db, store = storage
    s = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=None)
    store.write_bytes(s.id, "talk.wav", b"x")

    def broken(audio: Any, **_: Any) -> Any:
        raise RuntimeError("zebra lanterns glow in the transcript")

    analyzer, _ = _analyzer(fake_assessment, fake_parser)
    analyzer.analyze = broken
    with caplog.at_level("WARNING"), pytest.raises(app_run.AnalysisFailed) as e:
        analyzer.run(db, store, s, MemoryAudio("talk.wav", b"x"), limit_s=180.0, keep=False)
    assert db.get_session(s.id).status == "failed"
    assert "zebra" not in str(e.value) and "zebra" not in caplog.text
    assert store.media_names(s.id) == () and not analyzer.models_loaded


def test_models_stay_loaded_when_unloading_is_off(
    storage: Any, fake_assessment: Any, fake_parser: Any
):
    db, store = storage
    analyzer, loads = _analyzer(fake_assessment, fake_parser)
    analyzer.settings = analyzer.settings.model_copy(
        update={"app": AppSettings(unload_models_after_analysis=False)}
    )
    for _ in range(2):
        s = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=None)
        analyzer.run(db, store, s, MemoryAudio("t.wav", b"x"), limit_s=180.0, keep=False)
    assert loads == ["whisper_small"] and analyzer.models_loaded
