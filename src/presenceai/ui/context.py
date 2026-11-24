"""What the app's pages share (P3-c): the storage layout, the database and the media store.

Opening the context runs retention (FINAL_11 §3: at every app start). Without a usable MEDIA_KEY
the app still opens: `store` is None, so nothing can be recorded, exported or decrypted, while
listing and deleting still work (ADR 0023).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import SecretStr

from presenceai.schemas.storage import Consent
from presenceai.storage import media, retention
from presenceai.storage.db import Database
from presenceai.storage.layout import Layout


@dataclass
class AppContext:
    layout: Layout
    db: Database
    store: media.MediaStore | None  # None: MEDIA_KEY missing or invalid
    raw_media_days: int
    _maintenance: media.MediaStore = field(repr=False)  # list and delete only, no key

    @classmethod
    def open(
        cls,
        home: Path,
        media_key: SecretStr | None,
        *,
        raw_media_days: int,
        now: datetime | None = None,
    ) -> AppContext:
        layout = Layout(home)
        db = Database(layout)
        try:
            store: media.MediaStore | None = media.MediaStore(layout, media.fernet_from(media_key))
        except (media.MediaKeyMissing, media.MediaKeyInvalid):
            store = None
        maintenance = media.MediaStore(layout, None)
        retention.run(db, maintenance, now=now or datetime.now(UTC))
        return cls(layout, db, store, raw_media_days, maintenance)

    @property
    def key_ok(self) -> bool:
        return self.store is not None

    @property
    def maintenance(self) -> media.MediaStore:
        return self._maintenance

    def consent(self) -> Consent:
        return self.db.profile(now=datetime.now(UTC)).consent

    def reset(self) -> None:
        """After "delete everything": a new, empty database (the old one was closed)."""
        self.db = Database(self.layout)

    def close(self) -> None:
        self.db.close()
