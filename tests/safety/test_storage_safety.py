"""Storage safety (FINAL_11 §4, ADR 0023): delete really deletes, raw media is never written in
the clear, the key never shows, and storage opens no network connection."""

from __future__ import annotations

import logging
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from presenceai.schemas.storage import Consent
from presenceai.storage import media, retention, rights
from presenceai.storage.db import Database
from presenceai.storage.layout import Layout

from .conftest import NetworkBlocked

T0 = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


def _samples() -> np.ndarray:
    rng = np.random.default_rng(20261124)
    return (rng.uniform(-0.5, 0.5, 16_000)).astype(np.float32)


def _files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def test_no_plain_audio_anywhere_under_home(tmp_path: Path, loopback_only: list[str]):
    layout = Layout(tmp_path / "home")
    store = media.MediaStore(layout, Fernet(Fernet.generate_key()))
    db = Database(layout)
    s = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=30)
    samples = _samples()
    store.write_wav(s.id, "talk", samples, 16_000)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    upload = tmp_path / "upload.wav"
    upload.write_bytes(b"RIFF" + pcm[:4000] + b"WAVE")
    store.ingest_upload(s.id, upload)
    db.close()
    for f in _files(layout.home):
        data = f.read_bytes()
        assert b"RIFF" not in data and b"WAVE" not in data, f.name
        assert pcm[1000:1064] not in data, f.name  # no run of the raw samples either
    assert not upload.exists()
    assert loopback_only == []


def test_delete_everything_leaves_no_file_or_row(tmp_path: Path):
    layout = Layout(tmp_path / "home")
    store = media.MediaStore(layout, Fernet(Fernet.generate_key()))
    db = Database(layout)
    db.set_consent(Consent(use_app=True, keep_recordings=True), now=T0)
    for scenario in ("s1_presentation", "s2_project_defense"):
        s = db.new_session(scenario, "en", "app", now=T0, keep_days=30)
        store.write_bytes(s.id, "talk.wav", b"x")
        layout.assessment_path(s.id).write_text("{}", "utf-8")
    rights.export_all(db, store)
    rights.delete_everything(db)
    assert _files(layout.home) == []  # database, sessions, exports, temp: all gone
    fresh = Database(layout)  # the app can start again, with an empty profile
    assert fresh.list_sessions() == () and fresh.profile(now=T0).consent == Consent()
    fresh.close()


def test_retention_with_a_clock_override(tmp_path: Path):
    layout = Layout(tmp_path / "home")
    store = media.MediaStore(layout, Fernet(Fernet.generate_key()))
    db = Database(layout)
    s = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=30)
    store.write_bytes(s.id, "talk.wav", b"x")
    expiry = T0 + timedelta(days=30)
    assert retention.run(db, store, now=expiry - timedelta(seconds=1)).media_deleted == 0
    assert retention.run(db, store, now=expiry).media_deleted == 1
    db.close()


def test_the_key_never_appears_in_errors_logs_or_reprs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    key = Fernet.generate_key().decode()
    layout = Layout(tmp_path / "home")
    with caplog.at_level(logging.DEBUG):
        store = media.MediaStore(layout, media.fernet_from(SecretStr(key)))
        store.write_bytes("3" * 32, "talk.wav", b"x")
        other = media.MediaStore(layout, Fernet(Fernet.generate_key()))
        with pytest.raises(media.MediaUnreadable) as e:
            other.read_bytes("3" * 32, "talk.wav")
    for text in (caplog.text, str(e.value), repr(store), repr(other)):
        assert key not in text and key[:20] not in text


def test_the_loopback_guard_blocks_outside_addresses(loopback_only: list[str]):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlocked):
            s.connect(("203.0.113.5", 443))  # TEST-NET-3, never routed
    finally:
        s.close()
    assert loopback_only == ["203.0.113.5"]
