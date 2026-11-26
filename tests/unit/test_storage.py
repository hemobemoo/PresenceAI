"""Storage (P3-b, ADR 0023): layout, encrypted media, the database, retention, export/delete.

No network, no models: the Fernet key is generated for each test, never read from `.env`.
"""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from presenceai.schemas.storage import Consent
from presenceai.storage import media, reports, retention, rights
from presenceai.storage.db import Database
from presenceai.storage.layout import Layout

T0 = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    return Layout(tmp_path / "home")


@pytest.fixture
def store(layout: Layout) -> media.MediaStore:
    return media.MediaStore(layout, Fernet(Fernet.generate_key()))


@pytest.fixture
def db(layout: Layout):
    d = Database(layout)
    yield d
    d.close()


def _tone(seconds: float = 1.0, sr: int = 16_000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# --- layout -----------------------------------------------------------------------------------


def test_layout_paths_stay_inside_home(layout: Layout):
    sid = "0123456789abcdef0123456789abcdef"
    assert layout.session_dir(sid).parent == layout.sessions_dir
    assert layout.db_path.parent == layout.home
    for bad in ("../x", "a/b", "", "ABC", "0123456789abcdef0123456789abcdeg"):
        with pytest.raises(ValueError):
            layout.session_dir(bad)


# --- the key ----------------------------------------------------------------------------------


def test_missing_or_invalid_key_is_refused_without_showing_it():
    with pytest.raises(media.MediaKeyMissing):
        media.fernet_from(None)
    with pytest.raises(media.MediaKeyMissing):
        media.fernet_from(SecretStr(""))
    with pytest.raises(media.MediaKeyInvalid) as e:
        media.fernet_from(SecretStr("not-a-fernet-key-but-secret"))
    assert "not-a-fernet-key-but-secret" not in str(e.value)
    key = Fernet.generate_key().decode()
    assert media.fernet_from(SecretStr(key)) is not None


# --- encrypted media --------------------------------------------------------------------------


def test_media_round_trip_is_encrypted_on_disk(store: media.MediaStore, layout: Layout):
    sid = "0" * 32
    samples = _tone()
    path = store.write_wav(sid, "talk", samples, 16_000)
    assert path.suffix == ".enc" and path.parent == layout.media_dir(sid)
    raw = path.read_bytes()
    assert not raw.startswith(b"RIFF") and b"WAVE" not in raw
    back, sr = store.read_wav(sid, "talk")
    assert sr == 16_000 and back.dtype == np.float32
    assert np.max(np.abs(back - samples)) < 1e-3  # 16-bit PCM
    assert not list(layout.media_dir(sid).glob("*.part"))


def test_media_needs_the_same_key(store: media.MediaStore, layout: Layout):
    sid = "1" * 32
    store.write_bytes(sid, "upload.m4a", b"secret audio bytes")
    other = media.MediaStore(layout, Fernet(Fernet.generate_key()))
    with pytest.raises(media.MediaUnreadable):
        other.read_bytes(sid, "upload.m4a")


def test_ingest_upload_encrypts_and_removes_the_temp_copy(store: media.MediaStore, tmp_path: Path):
    sid = "2" * 32
    src = tmp_path / "gradio_tmp" / "my talk.m4a"
    src.parent.mkdir()
    src.write_bytes(b"fake m4a bytes")
    name = store.ingest_upload(sid, src)
    assert name == "upload.m4a" and not src.exists()
    assert store.read_bytes(sid, name) == b"fake m4a bytes"
    assert store.media_names(sid) == ("upload.m4a",)


# --- the database -----------------------------------------------------------------------------


def test_one_pseudonymous_profile_with_consent(db: Database):
    p = db.profile(now=T0)
    assert len(p.id) == 32 and p.consent == Consent()  # everything off until saved
    assert db.profile(now=T0 + timedelta(days=1)).id == p.id
    p = db.set_consent(Consent(use_app=True, keep_recordings=True), now=T0)
    assert p.consent.use_app and p.consent.keep_recordings and not p.consent.hosted_ai
    assert p.consent_updated_at == T0


def test_sessions_record_expiry_from_the_keep_consent(db: Database):
    kept = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=30)
    assert kept.media_expires_at == T0 + timedelta(days=30) and kept.status == "recorded"
    later = T0 + timedelta(minutes=5)
    unkept = db.new_session("s1_presentation", "ar", "upload", now=later, keep_days=None)
    assert unkept.media_expires_at is None
    db.set_status(kept.id, "analysed")
    assert db.get_session(kept.id).status == "analysed"
    assert [s.id for s in db.list_sessions()] == [kept.id, unkept.id]
    db.delete_session(kept.id)
    assert [s.id for s in db.list_sessions()] == [unkept.id]


def test_database_survives_reopening(layout: Layout):
    d = Database(layout)
    s = d.new_session("s2_project_defense", "en", "app", now=T0, keep_days=30)
    d.close()
    d = Database(layout)
    got = d.get_session(s.id)
    assert got.media_expires_at == T0 + timedelta(days=30)
    assert got.created_at.tzinfo is not None
    d.close()


# --- retention --------------------------------------------------------------------------------


def test_retention_deletes_expired_and_unkept_media(
    db: Database, store: media.MediaStore, layout: Layout
):
    kept = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=30)
    unkept = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=None)
    for s in (kept, unkept):
        store.write_bytes(s.id, "talk.wav", b"x")
        store.write_assessment(s.id, b"{}")
    r = retention.run(db, store, now=T0 + timedelta(days=1))
    assert r.media_deleted == 1  # the unkept one: the user did not consent to keeping it
    assert store.media_names(kept.id) and not store.media_names(unkept.id)
    assert db.get_session(unkept.id).media_deleted_at == T0 + timedelta(days=1)
    r = retention.run(db, store, now=T0 + timedelta(days=30))
    assert r.media_deleted == 1 and not store.media_names(kept.id)
    for s in (kept, unkept):
        assert store.read_assessment(s.id) == b"{}"  # derived data stays until deleted


def test_retention_clears_temp_and_exports(db: Database, store: media.MediaStore, layout: Layout):
    layout.tmp_dir.mkdir(parents=True)
    (layout.tmp_dir / "gradio" / "x").mkdir(parents=True)
    (layout.tmp_dir / "gradio" / "x" / "upload.wav").write_bytes(b"RIFF....WAVE")
    layout.exports_dir.mkdir(parents=True)
    (layout.exports_dir / "old.zip").write_bytes(b"PK")
    r = retention.run(db, store, now=T0)
    assert r.temp_files_deleted == 2
    assert not any(layout.tmp_dir.rglob("*.*")) and not any(layout.exports_dir.iterdir())


# --- export and delete ------------------------------------------------------------------------


def test_export_session_has_json_and_decrypted_media(
    db: Database, store: media.MediaStore, layout: Layout
):
    s = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=30)
    store.write_wav(s.id, "talk", _tone(0.5), 16_000)
    store.write_assessment(s.id, b'{"assessment": 1}')
    out = rights.export_session(db, store, s.id)
    assert out.parent == layout.exports_dir
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert {"session.json", "assessment.json", "media/talk.wav"} <= names
        assert z.read("media/talk.wav").startswith(b"RIFF")
        assert z.read("assessment.json") == b'{"assessment": 1}'  # decrypted
        meta = json.loads(z.read("session.json"))
        assert meta["id"] == s.id and meta["scenario"] == "s1_presentation"


def test_export_all_holds_the_profile_and_every_session(
    db: Database, store: media.MediaStore, layout: Layout
):
    db.set_consent(Consent(use_app=True), now=T0)
    a = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=None)
    b = db.new_session("s2_project_defense", "en", "app", now=T0, keep_days=30)
    store.write_bytes(b.id, "answer_1.wav", b"RIFFxxxxWAVE")
    out = rights.export_all(db, store)
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
    assert "profile.json" in names
    assert f"sessions/{a.id}/session.json" in names
    assert f"sessions/{b.id}/media/answer_1.wav" in names


def test_delete_session_removes_files_and_row(
    db: Database, store: media.MediaStore, layout: Layout
):
    s = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=30)
    store.write_bytes(s.id, "talk.wav", b"x")
    store.write_assessment(s.id, b"{}")
    rights.delete_session(db, s.id)
    assert not layout.session_dir(s.id).exists()
    assert db.list_sessions() == ()
    with pytest.raises(KeyError):
        db.get_session(s.id)


# --- reports: encrypted text, clear text-free summary (ADR 0023 item 12) ----------------------


def test_saved_report_round_trips_and_its_summary_is_text_free(
    db: Database, store: media.MediaStore, fake_assessment: Any
):
    s = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=30)
    reports.save(db, store, s.id, fake_assessment)
    assert db.get_session(s.id).status == "analysed"
    view = reports.open_report(db, store, s.id, lang="en")
    assert view.report == fake_assessment and view.text_note is None
    summary = view.summary
    dims = {d.dimension: d for d in summary.dimensions}
    a = fake_assessment.assessment
    assert set(dims) == {r.dimension for r in a.dimensions}
    for r, f in zip(a.dimensions, a.findings, strict=True):
        d = dims[r.dimension]
        assert (d.level, d.capped_from, d.score, d.confidence) == (
            r.level,
            r.capped_from,
            r.score,
            r.confidence,
        )
        assert (d.measured, d.total) == (f.measured, f.total)
        assert d.information_only == r.information_only
    assert dims["central_message"].information_only  # ADR 0025
    values = {m.measure: m.value for m in summary.measures}
    assert values["timing.overrun_s"] == 0.0
    # method A's output, as numbers only, for the later comparison with the P7 judge
    key = "text.central_message.possible_main_point"
    assert values[key] == 1.0
    cm = next(f for f in a.findings if f.dimension == "central_message")
    span = cm.evidence[0]
    assert span.transcript is not None and span.time is not None
    assert (values[f"{key}.start_char"], values[f"{key}.end_char"]) == (
        span.transcript.start_char, span.transcript.end_char)  # fmt: skip
    assert (values[f"{key}.start_s"], values[f"{key}.end_s"]) == (
        span.time.start_s,
        span.time.end_s,
    )
    assert "text.central_message.statements" not in values  # v1 is no longer computed
    dumped = summary.model_dump_json()
    for word in ("zebra", "lanterns", "practice", "Today"):
        assert word not in dumped  # numbers and codes only


def test_with_a_wrong_key_sessions_and_summaries_load_and_text_is_unavailable(
    layout: Layout, fake_assessment: Any
):
    d = Database(layout)
    good = media.MediaStore(layout, Fernet(Fernet.generate_key()))
    s = d.new_session("s1_presentation", "en", "app", now=T0, keep_days=30)
    reports.save(d, good, s.id, fake_assessment)
    d.close()
    d = Database(layout)
    wrong = media.MediaStore(layout, Fernet(Fernet.generate_key()))
    assert [x.id for x in d.list_sessions()] == [s.id]
    for lang, note in (("en", "text unavailable"), ("ar", None)):
        view = reports.open_report(d, wrong, s.id, lang=lang)  # type: ignore[arg-type]
        assert view.report is None and view.summary.dimensions
        assert view.text_note is not None
        if note:
            assert note in view.text_note
    d.close()


def test_deleting_a_session_removes_its_summary(
    db: Database, store: media.MediaStore, fake_assessment: Any
):
    s = db.new_session("s1_presentation", "en", "app", now=T0, keep_days=30)
    reports.save(db, store, s.id, fake_assessment)
    rights.delete_session(db, s.id)
    assert db.summary(s.id).dimensions == ()


def test_an_older_database_gets_the_information_only_column(layout: Layout):
    """A file made before ADR 0025 has no information_only column; opening it adds one."""
    import sqlite3

    layout.home.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(layout.db_path)
    con.execute("CREATE TABLE dimension_summary (session_id VARCHAR NOT NULL, dimension VARCHAR "
                "NOT NULL, level INTEGER, capped_from INTEGER, score FLOAT, confidence VARCHAR, "
                "measured INTEGER NOT NULL, total INTEGER NOT NULL, failed BOOLEAN NOT NULL, "
                "PRIMARY KEY (session_id, dimension))")  # fmt: skip
    con.execute("INSERT INTO dimension_summary VALUES ('a', 'delivery', 3, NULL, 0.67, 'low', "
                "2, 3, 0)")  # fmt: skip
    con.commit()
    con.close()
    d = Database(layout)
    try:
        with sqlite3.connect(layout.db_path) as c:
            cols = {row[1] for row in c.execute("PRAGMA table_info(dimension_summary)")}
            assert "information_only" in cols
            assert c.execute("SELECT information_only FROM dimension_summary").fetchone() == (0,)
        Database(layout).close()  # a second open does not add it again
    finally:
        d.close()
