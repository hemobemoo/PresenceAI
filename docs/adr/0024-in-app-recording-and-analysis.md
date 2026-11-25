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

## Round 1 of the owner's real S1 test (2026-11-25): failed, fixed

The owner pressed "Stop and analyse" and "Analyse the upload" and nothing visible happened; the
log held only Windows WinError 10054 tracebacks (633 in the recording session). Reproduced in a
scripted browser (Playwright, a fake microphone playing a C1 clip through the real WebRTC path,
and a real 60 s WAV upload). Causes and fixes:

1. **Every click was swallowed.** FastRTC 0.0.34's `WebRTC` defaults to `full_screen=True`, which
   draws a full-screen waveform layer over the page; neither analyse button ever received a
   click. Now `full_screen=False` (a test checks the built component).
2. **No sign of progress.** Each button first runs a step outside Gradio's queue that shows
   "Preparing the audio…" and disables both buttons (about 1 s after the press in the browser),
   then the analysis shows "Analysing… step N of 4: …" (load models, speech, content, save). Any
   failure ends as a message on the page with its error type; a test checks that every press
   ends in a report or a message (analysis error, storage error, upload-check error, a second
   analysis while one runs).
3. **The take and stopping the microphone box.** The audio survived the stop, but FastRTC's
   `shutdown` was not handled (the timer kept counting), and the buffer was reset in `start_up`,
   which FastRTC runs in a thread at the same time as the first frames. Now a take starts on its
   connection's first frame and `shutdown` ends the stream while keeping the take for "Stop and
   analyse" (tested).
4. **The timer flooded the server.** The 0.5 s `gr.Timer` sent a queued request (join plus event
   stream) per tick, about 720 over a 3-minute take, the source of the 10054 resets. Now a 1 s
   timer runs only while the microphone records (`start_recording` / `stop_recording`) and skips
   the queue.
5. **Windows 10054 noise.** The app runs uvicorn on its own event loop with an exception handler
   that drops only `ConnectionResetError` with `winerror == 10054`; everything else goes to
   asyncio's default handler (tested both ways). The browser re-run logged none.
6. **Logs.** One line when an analysis starts (session id, source, audio seconds) and one when
   it ends (step timings, total) or fails (step, error type); never text.
7. **A reload locked Practice again** (found while reproducing): the consent state and the My
   data table were read once at app start. Every page load now reads them again.
8. **WMA** (Sound Recorder's fifth format) gets its own message; the upload box lists the
   accepted formats.

Browser re-run on the laptop (whisper_small, 4 threads, models unloaded after each run):

| Path | Audio | Analysis | Of which speech | Report |
|---|---|---|---|---|
| Recording (fake microphone, real WebRTC) | 15.7 s | 10.0 s | 8.4 s | shown, quotes from the clip |
| WAV upload | 59.8 s | 17.4 s | 16.6 s | shown, 10 N-of-M badges |

### Open items

1. Re-check `completeness_min_ratio` (0.95) and `cpu_threads` (4) on the owner's first real
   recordings; record the observed ratio and analysis time here.
   - **Round 2 of the owner's real S1 test (2026-11-25), passed.** whisper_small, int8, 4 threads,
     models unloaded after each run:

     | Path | Audio | Analysis | load_models | speech | content | save |
     |---|---|---|---|---|---|---|
     | Live recording | 163.1 s | 69.3 s | 5.14 s | 63.58 s | 0.47 s | 0.07 s |
     | Upload (Sound Recorder) | 160.5 s | 69.6 s | 1.93 s | 67.28 s | 0.32 s | 0.07 s |

     Laptop: fan quiet, heat normal. No "incomplete" warning on the recording (the ratio was not
     yet logged; it is from this PR on). Report, export and delete as expected; no terminal
     errors.
   - **Decision (owner): keep `cpu_threads: 4`.** About 69 s for 2:40 of audio is 15 % over
     FINAL_02 §3.2's "under 60 s for a 3-minute talk", at no thermal cost; the speech step is
     about 95 % of it.
   - Still open: the completeness ratio on a real recording (now in the start log line).

## Correction (owner, 2026-11-25): the browser made an outside request until this fix

**The "no outside connection" claims for P3-d and P3-e were wrong until this fix.** The tests
named above (`test_recording_network.py`, ADR 0023's `loopback_only`) watch the **server**
process only; the **browser** was never watched. FastRTC 0.0.34's microphone component runs
`new Audio("https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/pop-sounds.mp3")`
on every load of the page (`/` and `/ar/`, each reload included). Gradio 5.23's page sets no
referrer policy, so the browser's default applied.

**What reached Hugging Face:** one `GET` for that public sound file per page load, with an
empty body. It carried what any such request carries: **the IP address and the time of the page
load**, plus the browser's standard headers. These were checked by intercepting the same request
in Chromium and aborting it locally:
- `User-Agent` and client hints (browser name and version, Windows);
- `Referer: http://127.0.0.1:7860/` (the origin only; it shows that a local app was open on
  that port);
- `Accept: */*` and `Range: bytes=0-`.

**Nothing from the app was in it**: no audio, transcript, report, consent setting, file name or
session id. Cookies: the test browser had none. If the browser was signed in to huggingface.co,
it may also have sent Hugging Face's own cookies, depending on how Hugging Face sets them; that
was not checked (it would mean reading the owner's browser profile).

**Fix** (`ui/offline.py`):
1. The URL is rewritten in the served script to `/local/silent.wav`, 0.1 s of silence served
   by the app.
2. Every page now carries a Content-Security-Policy that allows scripts, styles, fonts, images,
   media and connections from this server only. A browser that still holds the old script in
   its cache is blocked by it ("violates media-src", seen in the owner's browser before the
   cache was cleared). Any outside fetch a later Gradio or FastRTC adds is blocked the same way.
   WebRTC is not governed by CSP; it has no ICE server (item 2).

**Tests:**
- `tests/safety/test_app_network.py`: the served FastRTC scripts contain no Hugging Face URL;
  the sound is fetched from this computer; both pages carry the CSP.
- `tests/ui/test_browser_journey.py` (slow, needs_browser): a full S2 session in Chromium with
  a fake microphone. It covers page load, consent, recording, explanation, a silent first
  answer, the retry, both answers, stop, and analyse. It logs every request and fails on any
  host other than this computer or any CSP violation. Run on 2026-11-25: **148 requests, hosts
  `127.0.0.1` only, no CSP violation.**

**Found by the same test:** the retry check counted audio still in flight when the question
appeared (about 0.1–0.3 s, ADR 0022 "P3-e" item 2) as an answer. A silent answer therefore
never got the retry offer. It now uses the latency rule, so running speech is not an onset
(ADR 0022 "P3-e" item 5).
