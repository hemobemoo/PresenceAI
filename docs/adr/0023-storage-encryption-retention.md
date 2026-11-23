# 0023 — Storage, media encryption, retention, export and delete (P3-b)

- Status: **proposed** (2026-11-24). It follows the approved P3 plan; the owner accepts it with the PR.
- Date: 2026-11-24
- Decider: project owner
- Plan reference: FINAL_02 §6–§7 (storage layout), FINAL_08 (runtime data), FINAL_11 §3–§4
  (consent, encryption, retention, export, delete, tests), FINAL_13 (MEDIA_KEY), AGENTS.md
  rules 2, 5, 9; P3 plan (owner, 2026-11-24)
- Config keys affected: `configs/default.yaml: retention.raw_media_days` (30, from FINAL_11 §3;
  its "ADR pending" note now points here)

## Context

P3 records and stores practice sessions. AGENTS.md rule 5 requires raw media to be encrypted at
write time and never logged, printed or uploaded; FINAL_11 asks for 30-day retention, delete
that really deletes, and a one-zip export. In the P3 plan the owner asked this ADR to say where
MEDIA_KEY lives, what the encryption does and does not protect against, and that losing the key
makes stored media unreadable.

## Decision

1. **Layout** under `PRESENCEAI_HOME` (default `~/.presenceai`, outside the repository and
   outside OneDrive's default folders): `app.sqlite`; `sessions/<id>/media/*.enc` (raw media,
   encrypted); `sessions/<id>/assessment.json` (derived data); `tmp/` (Gradio's upload folder,
   P3-d); `exports/` (export zips). Session ids are 32 random hex characters and are checked
   before any path is built (`storage/layout.py`).
2. **Database** (`storage/db.py`): SQLite through SQLModel (MIT; SQLAlchemy MIT). One profile per
   installation with a **random pseudonymous id and no personal details**, its consent switches
   (a)–(d) (all off until saved) and the practice sessions (scenario, language, source app or
   upload, status, media expiry). The table classes are private; other modules get the records
   in `schemas/storage.py`. Times are timezone-aware UTC.
3. **Encryption** (`storage/media.py`): Fernet from `cryptography` (Apache-2.0 OR BSD-3-Clause):
   AES-128 in CBC mode with a random IV per file, authenticated with HMAC-SHA256. Bytes are
   encrypted **in memory**; only ciphertext is written, through a `.part` file renamed into
   place, so no clear copy is written by the app. Audio is decrypted into memory and passed on
   as an array (ADR 0006). Recordings are stored as 16-bit PCM WAV, mono, before encryption.
   An upload is encrypted as it arrives and Gradio's clear temp copy is deleted at once.
4. **Where MEDIA_KEY lives.** In `.env` on the laptop (git-ignored), with a copy in the owner's
   password manager (FINAL_13: generated once with `Fernet.generate_key()`); in Stage B, as a
   Space secret. **The app never generates, writes, logs or shows the key**: when MEDIA_KEY is
   missing or not a Fernet key, the app refuses to store media and says only which variable is
   wrong (`MediaKeyMissing`, `MediaKeyInvalid`). Code refers to it by name only (rule 9).
5. **Losing the key makes stored media unreadable.** There is no recovery and no back door:
   a different key fails Fernet's authentication (`MediaUnreadable`). Derived data
   (`assessment.json`, the database) stays readable, because it is not encrypted (item 7).
   Changing the key (rotation) is not supported in P3; it would need `MultiFernet` and
   re-encryption, recorded in a later ADR.
6. **What the encryption protects against:** the media files being read by someone who gets
   the files but not the key: the folder copied, backed up, synced to a cloud drive, left on a
   disk, or read by another user account. It also shows tampering: a changed file fails to
   decrypt.
7. **What it does not protect against:**
   - **anyone who can use the owner's account** on the laptop, or malware there: the key is in
     `.env` on the same machine;
   - **derived data**: transcripts, quotes, measures and findings in `assessment.json`, and the
     session metadata in `app.sqlite`, are **in the clear** (rule 5 covers raw media; see open
     item 1);
   - **metadata**: file sizes (about the recording length), times and the number of sessions;
   - **audio in memory** while recording or analysing, which the operating system may write to
     its page or hibernation file;
   - **export zips**, which hold decrypted media on purpose (item 9), and the user's own
     original of an uploaded file;
   - **recovery of deleted files** from the disk: the app deletes files but does not wipe them.
     What can be recovered is ciphertext, except export zips and Gradio's brief temp copy of an
     upload.
8. **Retention** (`storage/retention.py`), run at every app start (Stage A) with the clock passed
   in. With consent (c), media expires `retention.raw_media_days` (30) days after the session;
   without (c), it is deleted after analysis (P3-d) and by every retention run. `tmp/` and
   `exports/` are emptied. Derived data stays until the user deletes it (FINAL_11 §3).
9. **Export** (`storage/rights.py`): one zip in `exports/`: `session.json`, `assessment.json`
   and the decrypted media while still retained; "export all" adds `profile.json` and every
   session. This is the only deliberate clear copy, made on the user's request, and `exports/`
   is emptied at the next start.
10. **Delete**: a session's folder and row are removed at once. **Delete everything** removes
    the database file, all sessions, `exports/` and `tmp/`; the app starts again with a new,
    empty profile.
11. **Network:** storage opens no connection in Stage A. Supabase (Stage B) will be the only
    network use in `storage/media.py` (rule 2).

## Consequences

- Safety tests (`tests/safety/test_storage_safety.py`): no clear audio (no WAV header, no run
  of the raw samples) anywhere under `PRESENCEAI_HOME` after a recording and an upload; delete
  everything leaves no file and the app restarts empty; retention with a clock override (one
  second before and at 30 days); the key never appears in errors, logs or reprs; storage makes
  no non-loopback connection. `tests/safety/conftest.py: loopback_only` is the guard P3-d uses
  for recording and analysis.
- Unit tests (`tests/unit/test_storage.py`): paths, key handling, round trips, profile and
  consent, sessions, retention, export and delete.

### Open items

1. **Derived data in the clear (owner to decide).** `assessment.json` holds the transcript, so
   it says what was spoken. Rule 5 and FINAL_11 cover raw media only. Options: keep as is
   (documented here), or also encrypt `assessment.json` with the same key, which makes reports
   unreadable if the key is lost.
2. **Upload size.** Fernet encrypts in memory, so a very large upload (for example a long
   video) needs that much RAM. P3-d will set a size limit for uploads in `configs/`.
