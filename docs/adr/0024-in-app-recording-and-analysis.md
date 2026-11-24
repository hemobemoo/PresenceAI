# 0024 — In-app recording and analysis on the laptop (P3-d)

- Status: **accepted** by the owner on 2026-11-24. The values are the owner's P3 plan answers;
  all thresholds are **provisional**.
- Date: 2026-11-24
- Decider: project owner
- Plan reference: FINAL_02 §3.1 (live loop, recording), FINAL_12 §2 (P3 exit: record → report
  end to end locally), ADR 0002 (S1: FastRTC), ADR 0019 (overrun, app mode), ADR 0023 (storage)
- Config keys affected: `configs/default.yaml: app.recording_extra_s` (60), 
  `app.completeness_min_ratio` (0.95), `app.unload_models_after_analysis` (true),
  `asr.cpu_threads` (4); uploads: `app.upload_max_mb` (50), `app.upload_max_audio_s` (300),
  recorded in ADR 0023 item 2

## Context

P3's exit is record → report on the owner's laptop (i7-1255U, little RAM, gets hot). The owner
chose FastRTC audio-only recording with a completeness check, an elapsed-time timer, a hard stop
at the limit + 60 s, models unloaded after each analysis, and no outside connection while
recording or analysing (P3 plan, 2026-11-24).

## Decision

1. **Recording** (`live/recorder.py`, `ui/pages/practice.py`): FastRTC
   `WebRTC(modality="audio", mode="send")`; frames are resampled to 16 kHz by FastRTC and kept
   **in memory only**; the finished take is encoded as WAV and encrypted before it is written
   (ADR 0023). Video is not recorded in P3 (vision is P6).
2. **No STUN or TURN server in Stage A.** `{"iceServers": []}` is passed on the browser side
   (`rtc_configuration`) and on the server side (`server_rtc_configuration`). Without the server
   setting aiortc falls back to `stun:stun.l.google.com:19302`; a safety test pins both.
3. **Elapsed-time timer:** wall-clock time since the first frame, refreshed every 0.5 s, shown
   against the scenario limit (S1: 180 s, the upper end of `duration_s`).
4. **Hard stop at the limit + `app.recording_extra_s` (60 s, provisional):** later frames are
   ignored and the stream's time limit closes it; the report says the recording was stopped.
   The 60 s leave room above the largest overrun the rubric distinguishes (30 s for level 2),
   so overrun stays measurable (ADR 0019).
5. **Completeness check:** received audio seconds ÷ wall-clock seconds covered by the frames
   (first frame to last frame; idle time after the stream ended is not counted). Below
   `app.completeness_min_ratio` (0.95, provisional) the page warns: "The recording looks
   incomplete: X s of audio arrived over Y s." The analysis still runs; nothing is scored
   differently.
6. **One analysis at a time,** across both language pages (one `Analyzer` with a lock); a second
   request is told to wait.
7. **Models unloaded after each analysis** (`app.unload_models_after_analysis: true`): Whisper and
   the spaCy parser are loaded for the analysis and released after it (e5 joins in P3-e).
8. **`asr.cpu_threads: 4`** for CTranslate2 on the laptop (of 12 logical threads), to limit
   heat. The analysis is slower than with every thread; re-check with the owner's first real
   runs.
9. **The app's analysis mode** (ADR 0019): `strict=False`, camera off (camera measures left out
   of M), overrun measured against the scenario limit for in-app recordings only.
10. **Failures:** a failed analysis marks the session failed and logs only the error class; the
    page says so. Without consent (c) the recording is deleted after the analysis, whether it
    succeeded or not.
11. **The recording path and the upload path give the same audio** (owner, at acceptance):
    the same int16 samples recorded through the FastRTC handler or uploaded as a 16-bit WAV
    decode to the same float32 array, bit for bit, and give the same speech measures
    (`tests/unit/test_record_upload_parity.py`). The test found a scale mismatch: recorded
    audio was written back as WAV with a factor of 32767 while it had been read with 32768,
    so recordings came out 0.003 % quieter than the same audio uploaded (99.8 % of samples
    off by one step). `media.wav_bytes` now uses 32768 with rounding, the scale PyAV uses.

## Consequences

- Tests: `tests/unit/test_recorder.py` (timer, hard stop, completeness with a fake clock,
  int16 frames scaled), `tests/unit/test_app_run.py` (uploads incl. M4A, limits from config,
  analysis saved, models unloaded or kept, failure), `tests/ui/test_practice.py` (report cards,
  badges, text-unavailable fallback, Markdown escaping, M4A upload end to end with fakes,
  recording notices), `tests/safety/test_recording_network.py` (no ICE server on either side;
  no non-loopback connection while recording through the FastRTC handler and analysing).
- Not checked by tests: the browser's microphone path. The owner's real S1 run (P3 exit) checks
  it, together with the completeness ratio on real loopback WebRTC and the analysis time with 4
  threads.

### Open items

1. Re-check `completeness_min_ratio` (0.95) and `cpu_threads` (4) on the owner's first real
   recordings; record the observed ratio and analysis time here.
