"""Running an analysis inside the app (P3-d; ADR 0023, ADR 0024).

- `check_upload`: an uploaded file is read into memory and Gradio's clear temp copy is deleted at
  once; it is refused when it is larger than `app.upload_max_mb` (50 MB) or holds more than
  `app.upload_max_audio_s` (300 s) of audio, or when it cannot be decoded. M4A (Windows Sound
  Recorder's AAC), MP3, FLAC, WAV, OGG, AAC and WebM are accepted; WMA (Sound Recorder's other
  format) gets its own message.
- `Analyzer.steps` / `run`: one analysis at a time (a lock: the laptop is weak), in four named
  steps the page shows as they start. Whisper and the parser are
  loaded for the analysis and, with `app.unload_models_after_analysis`, released after it. The
  audio is decrypted into memory only (`MemoryAudio`). Scoring runs in the app's mode (ADR 0019:
  a failed finding shows "not measurable"); the report is saved encrypted with its text-free
  summary (ADR 0023 item 12); without consent (c) the recording is deleted right after.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from presenceai.pipeline.assess import assess_report, no_parser
from presenceai.schemas.report import AssessmentReport
from presenceai.schemas.storage import SessionRecord
from presenceai.settings import AppSettings, Settings
from presenceai.speech import analyze as speech_analyze
from presenceai.speech import asr
from presenceai.speech.audio import SAMPLE_RATE, MemoryAudio, load_audio
from presenceai.storage import reports
from presenceai.storage.db import Database
from presenceai.storage.media import MediaStore
from presenceai.text import nlp

log = logging.getLogger(__name__)

ACCEPTED_SUFFIXES = (".m4a", ".wav", ".mp3", ".ogg", ".flac", ".aac", ".webm")


@dataclass(frozen=True, repr=False)
class UploadCheck:
    """`reason` is a UI string key when the upload is refused; `audio` holds it otherwise."""

    audio: MemoryAudio | None = None
    duration_s: float | None = None
    reason: str | None = None
    values: tuple[tuple[str, float], ...] = ()  # placeholders for the reason's text


def check_upload(path: Path, cfg: AppSettings) -> UploadCheck:
    """Read the upload into memory, delete the clear temp copy, and check it."""
    try:
        suffix = path.suffix.lower()
        if suffix == ".wma":  # Sound Recorder can save WMA; PyAV builds may not decode it
            return UploadCheck(reason="practice.upload_wma")
        if suffix not in ACCEPTED_SUFFIXES:
            return UploadCheck(reason="practice.upload_type")
        if path.stat().st_size > cfg.upload_max_mb * 1024 * 1024:
            return UploadCheck(
                reason="practice.upload_too_big", values=(("mb", cfg.upload_max_mb),)
            )
        data = path.read_bytes()
    finally:
        path.unlink(missing_ok=True)  # Gradio's clear copy never outlives the check
    audio = MemoryAudio(name=f"upload{suffix}", data=data)
    try:
        duration = len(load_audio(audio)) / SAMPLE_RATE
    except Exception:  # any decoding failure: refuse, never crash
        return UploadCheck(reason="practice.upload_unreadable")
    if duration > cfg.upload_max_audio_s:
        minutes = cfg.upload_max_audio_s / 60
        return UploadCheck(reason="practice.upload_too_long", values=(("minutes", minutes),))
    if duration <= 0:
        return UploadCheck(reason="practice.upload_unreadable")
    return UploadCheck(audio=audio, duration_s=duration)


class AnalysisFailed(RuntimeError):
    """The analysis could not finish; the session is marked failed. No user text inside."""


STEPS = ("load_models", "speech", "content", "save")  # UI keys: practice.step.<name>


@dataclass(frozen=True)
class Step:
    """Progress of a running analysis: step `index` (1-based) of `total`."""

    index: int
    name: str

    @property
    def total(self) -> int:
        return len(STEPS)


@dataclass
class Analyzer:
    settings: Settings
    load_whisper: Callable[[str, Any], Any] = asr.load_model
    load_parser: Callable[[], Any] = nlp.spacy_parser
    analyze: Callable[..., Any] = speech_analyze.analyze_file

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._whisper: Any = None
        self._parser: Any = None

    @property
    def models_loaded(self) -> bool:
        return self._whisper is not None or self._parser is not None

    def busy(self) -> bool:
        return self._lock.locked()

    def unload(self) -> None:
        self._whisper = None
        self._parser = None
        gc.collect()

    def steps(
        self,
        db: Database,
        store: MediaStore,
        session: SessionRecord,
        audio: MemoryAudio,
        *,
        limit_s: float | None,
        keep: bool,
        duration_s: float | None = None,
        now: datetime | None = None,
    ) -> Generator[Step, None, AssessmentReport]:
        """Analyse one session, yielding each step as it starts; returns the report.

        One analysis at a time (a non-blocking lock: a second caller gets `AnalysisBusy`). Logs
        one line when it starts and one when it ends or fails: session id, source, audio
        duration, step timings and the error type; never any text (ADR 0024)."""
        if not self._lock.acquire(blocking=False):
            raise AnalysisBusy
        timings: dict[str, float] = {}
        current = "start"
        t0 = time.perf_counter()
        log.info(
            "analysis started: session=%s source=%s audio_s=%s",
            session.id, session.source, f"{duration_s:.1f}" if duration_s else "unknown",
        )  # fmt: skip
        try:
            cfg = self.settings
            language = session.language

            def step(i: int) -> Step:
                nonlocal current, t0
                now_t = time.perf_counter()
                if current != "start":
                    timings[current] = round(now_t - t0, 2)
                current, t0 = STEPS[i], now_t
                return Step(index=i + 1, name=STEPS[i])

            yield step(0)
            if self._whisper is None:
                self._whisper = self.load_whisper(cfg.asr.model, cfg.asr)
            if language == "en" and self._parser is None:
                self._parser = self.load_parser()
            yield step(1)
            speech = self.analyze(
                audio, language=language, model=self._whisper, model_name=cfg.asr.model,
                options=cfg.asr, speech=cfg.speech,
            )  # fmt: skip
            yield step(2)
            report = assess_report(
                speech, parser=self._parser if language == "en" else no_parser, cfg=cfg.speech,
                scenario=session.scenario, limit_s=limit_s, strict=False, camera_on=False,
            )  # fmt: skip
            yield step(3)
            reports.save(db, store, session.id, report)
            timings[current] = round(time.perf_counter() - t0, 2)
            log.info(
                "analysis finished: session=%s source=%s timings_s=%s total_s=%.1f",
                session.id, session.source, timings, sum(timings.values()),
            )  # fmt: skip
            return report
        except Exception as e:
            timings[current] = round(time.perf_counter() - t0, 2)
            log.warning(
                "analysis failed: session=%s source=%s step=%s error=%s timings_s=%s",
                session.id, session.source, current, type(e).__name__, timings,
            )  # fmt: skip
            db.set_status(session.id, "failed")
            raise AnalysisFailed(type(e).__name__) from None
        finally:
            try:
                if not keep:  # no consent (c): the recording goes right after the analysis
                    store.delete_media(session.id)
                    db.mark_media_deleted(session.id, now=now or datetime.now(UTC))
                if self.settings.app.unload_models_after_analysis:
                    self.unload()
            finally:
                self._lock.release()

    def run(
        self,
        db: Database,
        store: MediaStore,
        session: SessionRecord,
        audio: MemoryAudio,
        *,
        limit_s: float | None,
        keep: bool,
        duration_s: float | None = None,
        now: datetime | None = None,
    ) -> AssessmentReport:
        """`steps` without progress. Raises `AnalysisFailed` or `AnalysisBusy`."""
        gen = self.steps(
            db, store, session, audio, limit_s=limit_s, keep=keep, duration_s=duration_s, now=now
        )
        while True:
            try:
                next(gen)
            except StopIteration as done:
                return done.value


class AnalysisBusy(RuntimeError):
    """Another analysis is running (one at a time, ADR 0024)."""
