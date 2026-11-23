"""Stored records (P3-b, ADR 0023): the profile with its consent, and practice sessions.

No name, age, gender, employer or nationality is stored (FINAL_11 §3): the profile id is a
random pseudonym. Times are timezone-aware UTC. This module imports nothing from the project.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SessionStatus = Literal["recorded", "analysed", "failed"]
MediaSource = Literal["app", "upload"]  # app: recorded in the app (time limit known)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Consent(_Frozen):
    """The app's consent switches (a)-(d), FINAL_11 §3. Everything is off until saved."""

    use_app: bool = False  # (a)
    camera: bool = False  # (b)
    keep_recordings: bool = False  # (c) keep encrypted raw recordings temporarily
    hosted_ai: bool = False  # (d) AI wording and content checks (OpenRouter, P7)


class ProfileRecord(_Frozen):
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    created_at: datetime
    consent: Consent
    consent_updated_at: datetime | None = None


class SessionRecord(_Frozen):
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    created_at: datetime
    scenario: str = Field(min_length=1)
    language: Literal["en", "ar"]
    source: MediaSource
    status: SessionStatus
    # None: the user did not consent to keeping recordings, so media goes after analysis
    media_expires_at: datetime | None = None
    media_deleted_at: datetime | None = None


class RetentionResult(_Frozen):
    media_deleted: int = Field(ge=0)  # sessions whose media was deleted
    temp_files_deleted: int = Field(ge=0)  # files removed from tmp/ and exports/
