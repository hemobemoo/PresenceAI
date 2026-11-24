# 0023 — Storage, media encryption, retention, export and delete (P3-b)

- Status: **accepted** by the owner on 2026-11-24, with open item 1 decided (item 12).
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
   outside OneDrive's default folders): `app.sqlite` (with the text-free summaries, item 12);
   `sessions/<id>/media/*.enc` (raw media, encrypted); `sessions/<id>/assessment.json.enc`
   (the report, encrypted, item 12); `tmp/` (Gradio's upload folder, P3-d); `exports/`
   (export zips). Session ids are 32 random hex characters and are checked
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
   a different key fails Fernet's authentication (`MediaUnreadable`). **Reports lose their
   text, but progress survives** (item 12): the report page says "Report text unavailable" and
   still shows each dimension's level, N of M and the measure values from the clear summary.
   Changing the key (rotation) is not supported in P3; it would need `MultiFernet` and
   re-encryption, recorded in a later ADR.
6. **What the encryption protects against:** the media files being read by someone who gets
   the files but not the key: the folder copied, backed up, synced to a cloud drive, left on a
   disk, or read by another user account. It also shows tampering: a changed file fails to
   decrypt.
7. **What it does not protect against:**
   - **anyone who can use the owner's account** on the laptop, or malware there: the key is in
     `.env` on the same machine;
   - **the text-free summary** in `app.sqlite` (levels, scores, confidence, N of M, numeric
     measure values) and the session metadata (scenario, language, times) are in the clear by
     design (item 12); they hold no words the user said;
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
   (decrypted) and the decrypted media while still retained; "export all" adds `profile.json` and every
   session. This is the only deliberate clear copy, made on the user's request, and `exports/`
   is emptied at the next start.
10. **Delete**: a session's folder and row are removed at once. **Delete everything** removes
    the database file, all sessions, `exports/` and `tmp/`; the app starts again with a new,
    empty profile.
11. **Network:** storage opens no connection in Stage A. Supabase (Stage B) will be the only
    network use in `storage/media.py` (rule 2).
12. **The report's text is encrypted; a text-free summary stays in the clear** (owner,
    2026-11-24, open item 1).
    - Every text-bearing part of the assessment (the transcript, evidence quotes, finding
      text, not-measurable reasons, follow-up questions and answers: anything that can hold
      the user's words) is stored in `assessment.json.enc`, encrypted with MEDIA_KEY like
      media (`storage/reports.py: save`). It sits outside `media/`, so retention keeps it
      until the session is deleted.
    - A **text-free summary** is stored in SQLite in the clear, per session and dimension:
      level, capped_from, score, confidence, N of M and whether the app could not check it
      (`dimension_summary`), and per measure the numeric value, unit code and confidence
      (`measure_value`). No quotes, no transcript, no finding text, no reasons.
      **The progress page and the engine (P5) read only this summary.**
    - **If the key is lost, reports lose their text but progress survives**: the session list
      and the summaries still load, and the report shows "Report text unavailable" instead of
      failing (`reports.open_report`).
    - Export decrypts everything into the zip, as before.

## Consequences

- Safety tests (`tests/safety/test_storage_safety.py`): no clear audio (no WAV header, no run
  of the raw samples) anywhere under `PRESENCEAI_HOME` after a recording and an upload; no
  transcript text in the clear anywhere under it, the SQLite file included (a known phrase
  from a fake transcript, searched as UTF-8 and UTF-16); delete
  everything leaves no file and the app restarts empty; retention with a clock override (one
  second before and at 30 days); the key never appears in errors, logs or reprs; storage makes
  no non-loopback connection. `tests/safety/conftest.py: loopback_only` is the guard P3-d uses
  for recording and analysis.
- Unit tests (`tests/unit/test_storage.py`): paths, key handling, round trips, profile and
  consent, sessions, retention, export and delete; the summary is text-free and matches the
  assessment; with a wrong key the sessions and summaries load and the report says its text
  is unavailable.

### Open items

1. ~~**Derived data in the clear.**~~ **Decided 2026-11-24 (item 12):** the report's text is
   encrypted; a text-free summary stays in the clear.
2. ~~**Upload size.**~~ **Closed 2026-11-24 (owner, P3-d):** uploads are refused above
   `app.upload_max_mb: 50` MB or `app.upload_max_audio_s: 300` s (5 minutes) of audio, both
   provisional (`pipeline/app_run.py: check_upload`). The file is read into memory and Gradio's
   clear temp copy is deleted before the checks; a refused upload stores nothing. M4A (Windows
   Sound Recorder) is accepted and tested with an AAC fixture built in memory.
