"""Where runtime data lives under `PRESENCEAI_HOME` (default `~/.presenceai`, FINAL_08):

    app.sqlite                      profile, consent, sessions
    sessions/<id>/media/*.enc       raw media, Fernet-encrypted at write time
    sessions/<id>/assessment.json   derived data (transcript, measures, findings)
    tmp/                            Gradio's upload folder (P3-d); emptied at every start
    exports/                        export zips on request; emptied at every start

Session ids are 32 lowercase hex characters, checked before any path is built, so no id can
point outside `sessions/`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_ID = re.compile(r"^[0-9a-f]{32}$")


def check_id(session_id: str) -> str:
    if not _ID.fullmatch(session_id):
        raise ValueError("a session id is 32 lowercase hex characters")
    return session_id


@dataclass(frozen=True)
class Layout:
    home: Path

    @property
    def db_path(self) -> Path:
        return self.home / "app.sqlite"

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def tmp_dir(self) -> Path:
        return self.home / "tmp"

    @property
    def exports_dir(self) -> Path:
        return self.home / "exports"

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / check_id(session_id)

    def media_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "media"

    def assessment_path(self, session_id: str) -> Path:
        d = self.session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        return d / "assessment.json"
