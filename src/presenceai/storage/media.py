"""Encrypted raw media (AGENTS.md rule 5; FINAL_11 §3; ADR 0023).

Raw media is encrypted **at write time**: bytes are encrypted in memory with Fernet (AES-128-CBC
plus HMAC-SHA256, from `cryptography`) and only the ciphertext reaches the disk, through a
`.part` file renamed into place. Reading decrypts into memory; audio is handed to the analysis
as an array (ADR 0006). The key is `MEDIA_KEY` (`.env`, backed up in a password manager,
FINAL_13); it is never generated, logged or shown here. Without the same key stored media cannot
be read.

Stage B (Supabase Storage, encrypted before upload) will be the only network use in this module
(AGENTS.md rule 2); Stage A opens no connection.
"""

from __future__ import annotations

import io
import os
import shutil
import wave
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr

from presenceai.storage.layout import Layout, check_id

SUFFIX = ".enc"


class MediaKeyMissing(RuntimeError):
    """MEDIA_KEY is not set: the app cannot store recordings (see FINAL_13, MEDIA_KEY)."""


class MediaKeyInvalid(RuntimeError):
    """MEDIA_KEY is set but is not a Fernet key. The value is never shown."""


class MediaUnreadable(RuntimeError):
    """Stored media could not be decrypted: a different or lost MEDIA_KEY, or a damaged file."""


def fernet_from(key: SecretStr | None) -> Fernet:
    if key is None or not key.get_secret_value():
        raise MediaKeyMissing("MEDIA_KEY is not set; see the setup guide (FINAL_13, MEDIA_KEY)")
    try:
        return Fernet(key.get_secret_value().encode())
    except (ValueError, TypeError):
        raise MediaKeyInvalid("MEDIA_KEY is not a valid Fernet key") from None


def _check_name(name: str) -> str:
    if not name or Path(name).name != name or name.startswith(".") or name.endswith(SUFFIX):
        raise ValueError("a media name is a plain file name")
    return name


class MediaStore:
    def __init__(self, layout: Layout, fernet: Fernet) -> None:
        self.layout = layout
        self._fernet = fernet

    def __repr__(self) -> str:  # never the key
        return f"MediaStore(home={self.layout.home.name!r})"

    def _path(self, session_id: str, name: str) -> Path:
        return self.layout.media_dir(session_id) / f"{_check_name(name)}{SUFFIX}"

    def write_bytes(self, session_id: str, name: str, data: bytes) -> Path:
        path = self._path(session_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_bytes(self._fernet.encrypt(data))
        os.replace(part, path)
        return path

    def read_bytes(self, session_id: str, name: str) -> bytes:
        try:
            return self._fernet.decrypt(self._path(session_id, name).read_bytes())
        except InvalidToken:
            raise MediaUnreadable("stored media could not be decrypted with MEDIA_KEY") from None

    def write_wav(self, session_id: str, name: str, samples: np.ndarray, sample_rate: int) -> Path:
        """Mono float samples in [-1, 1] as 16-bit PCM WAV, encrypted (`<name>.wav.enc`)."""
        pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
        return self.write_bytes(session_id, f"{name}.wav", buf.getvalue())

    def read_wav(self, session_id: str, name: str) -> tuple[np.ndarray, int]:
        with wave.open(io.BytesIO(self.read_bytes(session_id, f"{name}.wav")), "rb") as w:
            rate = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return (pcm.astype(np.float32) / 32767).astype(np.float32), rate

    def ingest_upload(self, session_id: str, src: Path) -> str:
        """Encrypt an uploaded file (Gradio's temp copy) and delete the clear copy."""
        name = f"upload{src.suffix.lower()}"
        try:
            self.write_bytes(session_id, name, src.read_bytes())
        finally:
            src.unlink(missing_ok=True)
        return name

    def media_names(self, session_id: str) -> tuple[str, ...]:
        d = self.layout.media_dir(session_id)
        if not d.exists():
            return ()
        return tuple(sorted(p.name[: -len(SUFFIX)] for p in d.glob(f"*{SUFFIX}")))

    def delete_media(self, session_id: str) -> int:
        """Remove the session's media folder; returns the number of media files removed."""
        d = self.layout.media_dir(check_id(session_id))
        if not d.exists():
            return 0
        n = sum(1 for _ in d.glob(f"*{SUFFIX}"))
        shutil.rmtree(d)
        return n
