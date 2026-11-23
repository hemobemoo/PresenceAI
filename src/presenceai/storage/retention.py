"""Retention (FINAL_11 §3, ADR 0023): run at every app start (Stage A).

- Media of a session whose `media_expires_at` has passed is deleted (30 days by default,
  `configs/default.yaml: retention.raw_media_days`).
- Media of a session without consent (c) to keep recordings (`media_expires_at` None) is
  deleted whenever retention runs; the pipeline also deletes it right after analysis (P3-d).
- `tmp/` (Gradio's upload copies) and `exports/` (export zips, the one deliberate clear copy,
  made on the user's request) are emptied.

Derived data (`assessment.json`) is kept until the user deletes the session (FINAL_11 §3).
`now` is passed in, so tests can move the clock.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from presenceai.schemas.storage import RetentionResult, SessionRecord
from presenceai.storage.db import Database
from presenceai.storage.media import MediaStore


def due(s: SessionRecord, now: datetime) -> bool:
    if s.media_deleted_at is not None:
        return False
    return s.media_expires_at is None or now >= s.media_expires_at


def _empty(folder: Path) -> int:
    if not folder.exists():
        return 0
    n = sum(1 for p in folder.rglob("*") if p.is_file())
    for child in folder.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return n


def run(db: Database, store: MediaStore, *, now: datetime) -> RetentionResult:
    deleted = 0
    for s in db.list_sessions():
        if due(s, now):
            if store.delete_media(s.id):
                deleted += 1
            db.mark_media_deleted(s.id, now=now)
    swept = _empty(store.layout.tmp_dir) + _empty(store.layout.exports_dir)
    return RetentionResult(media_deleted=deleted, temp_files_deleted=swept)
