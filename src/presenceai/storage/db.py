"""The app database (P3-b, ADR 0023): SQLite at `PRESENCEAI_HOME/app.sqlite`, through SQLModel.

One pseudonymous profile per installation (a random id; no name or personal details,
FINAL_11 §3) with its consent switches, and the practice sessions. Callers get the frozen
records from `schemas/storage.py`; the table classes stay private to this module. Times are
timezone-aware UTC (SQLModel refuses naive ones) and returned in UTC.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine
from sqlmodel import Field, Session, SQLModel, col, create_engine, select

from presenceai.schemas.storage import (
    Consent,
    DimensionSummary,
    MeasureValue,
    MediaSource,
    ProfileRecord,
    SessionRecord,
    SessionStatus,
    SessionSummary,
)
from presenceai.storage.layout import Layout, check_id


class _Profile(SQLModel, table=True):
    __tablename__ = "profile"  # pyright: ignore[reportAssignmentType]
    id: str = Field(primary_key=True)
    created_at: datetime
    use_app: bool = False
    camera: bool = False
    keep_recordings: bool = False
    hosted_ai: bool = False
    consent_updated_at: datetime | None = None


class _Session(SQLModel, table=True):
    __tablename__ = "practice_session"  # pyright: ignore[reportAssignmentType]
    id: str = Field(primary_key=True)
    created_at: datetime = Field(index=True)
    scenario: str
    language: str
    source: str
    status: str
    media_expires_at: datetime | None = None
    media_deleted_at: datetime | None = None


class _DimensionRow(SQLModel, table=True):
    """Text-free summary per session and dimension (ADR 0023 item 12): numbers and codes only."""

    __tablename__ = "dimension_summary"  # pyright: ignore[reportAssignmentType]
    session_id: str = Field(primary_key=True, foreign_key="practice_session.id")
    dimension: str = Field(primary_key=True)
    level: int | None = None
    capped_from: int | None = None
    score: float | None = None
    confidence: str | None = None
    measured: int
    total: int
    failed: bool = False


class _MeasureRow(SQLModel, table=True):
    __tablename__ = "measure_value"  # pyright: ignore[reportAssignmentType]
    session_id: str = Field(primary_key=True, foreign_key="practice_session.id")
    measure: str = Field(primary_key=True)
    value: float
    unit: str | None = None
    confidence: str | None = None


def _store(t: datetime) -> datetime:
    if t.tzinfo is None:
        raise ValueError("times must be timezone-aware")
    return t.astimezone(UTC)


def _load(t: datetime | None) -> datetime | None:
    if t is None:
        return None
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def _profile(row: _Profile) -> ProfileRecord:
    return ProfileRecord(
        id=row.id,
        created_at=_load(row.created_at) or row.created_at,
        consent=Consent(
            use_app=row.use_app,
            camera=row.camera,
            keep_recordings=row.keep_recordings,
            hosted_ai=row.hosted_ai,
        ),
        consent_updated_at=_load(row.consent_updated_at),
    )


def _session(row: _Session) -> SessionRecord:
    return SessionRecord.model_validate(
        {
            "id": row.id,
            "created_at": _load(row.created_at),
            "scenario": row.scenario,
            "language": row.language,
            "source": row.source,
            "status": row.status,
            "media_expires_at": _load(row.media_expires_at),
            "media_deleted_at": _load(row.media_deleted_at),
        }
    )


class Database:
    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        layout.home.mkdir(parents=True, exist_ok=True)
        self._engine: Engine = create_engine(f"sqlite:///{layout.db_path.as_posix()}")
        SQLModel.metadata.create_all(self._engine)

    def close(self) -> None:
        """Release the file (needed before deleting it on Windows)."""
        self._engine.dispose()

    # --- profile and consent ------------------------------------------------------------------

    def profile(self, *, now: datetime) -> ProfileRecord:
        """The installation's profile, created (all consent off) on first use."""
        with Session(self._engine) as s:
            row = s.exec(select(_Profile)).first()
            if row is None:
                row = _Profile(id=uuid.uuid4().hex, created_at=_store(now))
                s.add(row)
                s.commit()
                s.refresh(row)
            return _profile(row)

    def set_consent(self, consent: Consent, *, now: datetime) -> ProfileRecord:
        pid = self.profile(now=now).id
        with Session(self._engine) as s:
            row = s.get_one(_Profile, pid)
            row.use_app = consent.use_app
            row.camera = consent.camera
            row.keep_recordings = consent.keep_recordings
            row.hosted_ai = consent.hosted_ai
            row.consent_updated_at = _store(now)
            s.add(row)
            s.commit()
            s.refresh(row)
            return _profile(row)

    # --- sessions -----------------------------------------------------------------------------

    def new_session(
        self,
        scenario: str,
        language: str,
        source: MediaSource,
        *,
        now: datetime,
        keep_days: int | None,
    ) -> SessionRecord:
        """`keep_days`: the retention period when the user consented to keeping recordings,
        None when they did not (media is then deleted after analysis)."""
        expires = None if keep_days is None else _store(now + timedelta(days=keep_days))
        row = _Session(
            id=uuid.uuid4().hex,
            created_at=_store(now),
            scenario=scenario,
            language=language,
            source=source,
            status="recorded",
            media_expires_at=expires,
        )
        record = _session(row)  # validate before writing
        with Session(self._engine) as s:
            s.add(row)
            s.commit()
        return record

    def get_session(self, session_id: str) -> SessionRecord:
        with Session(self._engine) as s:
            row = s.get(_Session, check_id(session_id))
            if row is None:
                raise KeyError("no such session")
            return _session(row)

    def list_sessions(self) -> tuple[SessionRecord, ...]:
        """Oldest first."""
        with Session(self._engine) as s:
            rows = s.exec(select(_Session).order_by(col(_Session.created_at), col(_Session.id)))
            return tuple(_session(r) for r in rows)

    def _update(self, session_id: str, **values: object) -> None:
        with Session(self._engine) as s:
            row = s.get(_Session, check_id(session_id))
            if row is None:
                raise KeyError("no such session")
            for k, v in values.items():
                setattr(row, k, v)
            s.add(row)
            s.commit()

    def set_status(self, session_id: str, status: SessionStatus) -> None:
        self._update(session_id, status=status)

    def mark_media_deleted(self, session_id: str, *, now: datetime) -> None:
        self._update(session_id, media_deleted_at=_store(now))

    def delete_session(self, session_id: str) -> None:
        sid = check_id(session_id)
        with Session(self._engine) as s:
            self._clear_summary(s, sid)
            row = s.get(_Session, sid)
            if row is not None:
                s.delete(row)
            s.commit()

    # --- text-free summaries (ADR 0023 item 12) -----------------------------------------------

    @staticmethod
    def _clear_summary(s: Session, session_id: str) -> None:
        for d in s.exec(select(_DimensionRow).where(col(_DimensionRow.session_id) == session_id)):
            s.delete(d)
        for m in s.exec(select(_MeasureRow).where(col(_MeasureRow.session_id) == session_id)):
            s.delete(m)

    def save_summary(self, summary: SessionSummary) -> None:
        """Replace the session's summary."""
        sid = check_id(summary.session_id)
        with Session(self._engine) as s:
            self._clear_summary(s, sid)
            s.flush()
            for d in summary.dimensions:
                s.add(_DimensionRow(session_id=sid, **d.model_dump()))
            for m in summary.measures:
                s.add(_MeasureRow(session_id=sid, **m.model_dump()))
            s.commit()

    def summary(self, session_id: str) -> SessionSummary:
        sid = check_id(session_id)
        with Session(self._engine) as s:
            dims = s.exec(
                select(_DimensionRow)
                .where(col(_DimensionRow.session_id) == sid)
                .order_by(col(_DimensionRow.dimension))
            ).all()
            rows = s.exec(
                select(_MeasureRow)
                .where(col(_MeasureRow.session_id) == sid)
                .order_by(col(_MeasureRow.measure))
            ).all()
            dimensions = tuple(
                DimensionSummary.model_validate(d.model_dump(exclude={"session_id"})) for d in dims
            )
            measures = tuple(
                MeasureValue.model_validate(m.model_dump(exclude={"session_id"})) for m in rows
            )
        return SessionSummary(session_id=sid, dimensions=dimensions, measures=measures)
