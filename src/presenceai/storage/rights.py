"""Export and delete (FINAL_11 §3-§4, ADR 0023).

- **Export** writes one zip to `exports/`: `session.json` (the session record),
  `assessment.json` (decrypted) when analysed, and the media decrypted, if still retained. It
  is the one deliberate clear copy, made only on the user's request; `exports/` is emptied at
  the next start (retention).
- **Delete** removes a session's folder (media and derived data) and its row at once.
  **Delete everything** removes the database file, every session, `exports/` and `tmp/`; the app
  then starts again with a new, empty profile.
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import UTC, datetime

from presenceai.storage.db import Database
from presenceai.storage.media import MediaStore


def _add_session(z: zipfile.ZipFile, db: Database, store: MediaStore, sid: str, prefix: str):
    z.writestr(f"{prefix}session.json", db.get_session(sid).model_dump_json(indent=2))
    if store.has_assessment(sid):
        z.writestr(f"{prefix}assessment.json", store.read_assessment(sid))
    for name in store.media_names(sid):
        z.writestr(f"{prefix}media/{name}", store.read_bytes(sid, name))


def _zip_path(store: MediaStore, stem: str):
    store.layout.exports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return store.layout.exports_dir / f"presenceai-{stem}-{stamp}.zip"


def export_session(db: Database, store: MediaStore, session_id: str):
    out = _zip_path(store, f"session-{session_id[:8]}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        _add_session(z, db, store, session_id, "")
    return out


def export_all(db: Database, store: MediaStore):
    out = _zip_path(store, "all")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("profile.json", db.profile(now=datetime.now(UTC)).model_dump_json(indent=2))
        for s in db.list_sessions():
            _add_session(z, db, store, s.id, f"sessions/{s.id}/")
    return out


def delete_session(db: Database, session_id: str) -> None:
    folder = db.layout.session_dir(session_id)
    if folder.exists():
        shutil.rmtree(folder)
    db.delete_session(session_id)


def delete_everything(db: Database) -> None:
    """Everything the app stored. `db` is closed; open a new `Database` to continue."""
    layout = db.layout
    db.close()
    for folder in (layout.sessions_dir, layout.exports_dir, layout.tmp_dir):
        if folder.exists():
            shutil.rmtree(folder)
    for suffix in ("", "-journal", "-wal", "-shm"):
        layout.db_path.with_name(layout.db_path.name + suffix).unlink(missing_ok=True)
