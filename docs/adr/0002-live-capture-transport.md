# 0002 — Live capture transport (spike S1)

- Status: **proposed**. It stays proposed until the landmark run below passes; then it goes to the owner for approval.
- Date: 2026-11-12
- Decider: project owner
- Plan reference: docs/plan/FINAL_12_ROADMAP_AND_THESIS.md §1 (S1); FINAL_02 §3.1; FINAL_10 E8
- Config keys affected: none yet; `configs/live.yaml` stays as is (`video.min_fps` 8, `latency_budget_ms` 300)
- Evidence: `experiments/results/s1/20261112-181421_real_*` (60 s) and `20261112-181657_real_*` (180 s), produced by `experiments/s1_live_capture/measure.py`

## Context

S1 asks: *can FastRTC inside `gr.Blocks` stream mic + webcam to Python handlers locally, with
meters updating?* Pass condition from FINAL_12: **≥ 8 fps landmarks and < 300 ms meter delay on
the laptop**. If it fails, the plan falls back to two Gradio streaming components (`gr.Audio` +
`gr.Image` webcam).

For this spike, "landmarks" is replaced by a stand-in: each received frame is downscaled to
640×360 and its grey mean computed. Real MediaPipe landmarks are a follow-up measurement.

## Options measured

1. **FastRTC** `WebRTC(modality="audio-video", mode="send-receive")` with an
   `AudioVideoStreamHandler`. FastRTC supports audio-video **only in send-receive mode**, not
   "send" as FINAL_02 assumed. The handler therefore returns a video frame: it echoes a
   **160×90** downscaled frame (`echo_size` in `experiments/configs/s1_live_capture.yaml`) to keep
   the return path light, and sends no audio. The meter is published every 100 ms through
   `AdditionalOutputs`. Track constraints: 640×360, ideal 30 fps.
2. **Fallback**: `gr.Audio(sources="microphone", streaming=True)` and
   `gr.Image(sources="webcam", streaming=True)`, each with `.stream(stream_every=0.1)`. The
   browser sends a snapshot every 0.1 s, so the frame rate is **capped at 10 fps by design**.
   That is only 2 fps of headroom over the 8 fps threshold.

## How it is measured

- **Per second, server side** (`*_server.csv`): video frames received and processed, audio
  chunks, audio seconds received, meter values published, processing ms (p50 and max), and
  process CPU% (psutil, sampled once per second; 100 % = one logical core). A run starts at the
  first media event. The last second, cut short by Stop, is not reported.
- **Per meter value, browser side** (`*_browser.csv`): the page is polled once per animation
  frame, and `meter_ms = render time − server publish timestamp`. Both clocks are the same
  machine's wall clock.
- **meter_ms covers server → render only. It does NOT include camera/microphone → server**
  (capture, encoding, WebRTC jitter buffer or Gradio upload). FastRTC hands the handler a plain
  frame array without its capture timestamp, so that leg cannot be measured from Python.
- **Lag (accepted by the owner, 2026-11-12):** because frame capture timestamps are not
  available, a backlog is detected indirectly from two signals:
  - the **meter_ms slope**: the least-squares slope of meter_ms over the post-warm-up run, in
    ms per minute. A clearly positive slope means outputs are queueing.
  - the **audio real-time ratio**: audio seconds received per wall-clock second, ≈ 1.0 when
    audio keeps up. Below 1.0 means audio arrives late or is lost.
- **Warm-up**: the first 5 s of each transport run (`warmup_s`) stay in the per-second CSV,
  flagged `warmup=True`, but are excluded from all fps and latency statistics.

## PASS rule

For each transport, over post-warm-up seconds only:

> **PASS = processed fps ≥ `min_fps` in ≥ 95 % of seconds AND meter_ms p95 < `latency_budget_ms`.**

`min_fps` (8) and `latency_budget_ms` (300) are read from `configs/live.yaml`. The 95 % share
(`min_share_fps_ok`) and the 5 s warm-up are set in `experiments/configs/s1_live_capture.yaml`.
**meter_ms max is reported but not used in PASS.**

## Results

Machine: Intel Core i7-1255U (10 cores / 12 threads), 15.7 GB RAM, Windows 11 Pro 10.0.26300,
on charger (machine details read from the PC the runs were made on; owner to confirm) ·
Browser: Chromium 158.0.8127.18 via Playwright 1.65.1, headed · gradio 5.23.1, fastrtc 0.0.34 ·
Profile (`configs/default.yaml`): standard · Camera and microphone: built-in.

| Run | Transport | Seconds used / total | fps p50 | fps min | share fps ≥ 8 | meter_ms p50 | p95 | max | meter_ms slope (ms/min) | audio real-time ratio | audio chunks/s p50 | CPU% p50 / max | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 s | FastRTC | 53 / 58 | 30 | 29 | 100 % | 52.4 | 69.5 | 85.8 | +0.62 | 1.00 | 50 | 31.3 / 43.8 | ✓ |
| 60 s | Fallback | 55 / 60 | 10 | 10 | 100 % | 67.4 | 90.7 | 110.1 | −8.40 | 0.955 | 10 | 23.4 / 34.4 | ✓ |
| 180 s | FastRTC | 173 / 178 | 30 | 29 | 100 % | 52.9 | 69.3 | 73.3 | +0.40 | 1.00 | 50 | 28.1 / 40.6 | ✓ |
| 180 s | Fallback | 175 / 180 | 10 | 10 | 100 % | 66.5 | 91.7 | 111.3 | +0.82 | 0.947 | 9 | 23.4 / 34.4 | ✓ |

meter_n (meter values matched in the browser): 529 / 521 (60 s) and 1736 / 1652 (180 s), FastRTC / fallback.

Reading the numbers:

- **Both transports pass S1 in both runs.** FastRTC keeps 30 fps, almost 4× the threshold, with
  meter_ms p95 ≈ 70 ms, under a quarter of the 300 ms budget. The fallback sits at its 10 fps cap
  with p95 ≈ 91 ms.
- **No backlog in FastRTC:** the meter_ms slope is under 1 ms/min over 3 minutes and the audio
  real-time ratio is 1.00.
- **The fallback loses about 5 % of audio time** (ratio 0.95), so its meter would miss some
  sound. FastRTC loses none.
- **FastRTC costs more CPU** (p50 28–31 % against 23 %), as expected at 3× the frame rate. The
  stand-in processing is cheap, so real landmarks will change this.

Observations (owner): the run went well, but camera quality dropped and the image looked
pixelated once the camera started. Not investigated yet. A likely cause, not verified, is
WebRTC's encoder starting at a low bitrate and degrading resolution until its bandwidth
estimate ramps up. FastRTC documents `rtp_params={"degradationPreference": "maintain-resolution"}`
for this. The stand-in downscales every frame to 640×360, so this did not affect the fps
numbers, but it would affect landmark quality. Check it in the MediaPipe follow-up.

Limits of this evidence: one run of each length on one machine; meter_ms excludes capture →
server; the landmark step is a stand-in.

## Landmark run (follow-up, pending)

The first runs used a stand-in for landmarks. This run replaces it with MediaPipe in VIDEO mode,
one person, models fetched and SHA-256-pinned by `scripts/download_models.py` (digests in
`models/MANIFEST.json`):

- `--landmarks none | pose-lite | pose-full | pose-lite+face`, one app start per mode, to see
  the load step by step. Face Landmarker runs with `output_face_blendshapes=False` (geometry only).
- Landmarks run on a worker thread that keeps only the latest frame. FastRTC's `video_receive`
  is never blocked, and frames that arrive while the worker is busy are counted as
  `frames_dropped`.
- Frames sent to the worker: every 3rd FastRTC frame (≈ 10 fps from 30 fps; FINAL_02 §3.1
  "every 2nd–3rd frame") and every fallback frame (`landmark_process_every` in the spike config).
  `--process-every N` overrides this.
- **Same PASS rule as above, now on landmark-processed fps.** Also reported, but not part of PASS:
  pose and face success rate (share of processed frames where a pose or face was found), share
  of frames dropped, and the smallest received frame size per second.
- **Pixelation fix:** FastRTC now sends with `rtp_params={"degradationPreference":
  "maintain-resolution"}` (spike config `rtp_params`), so under load the browser lowers frame
  rate instead of resolution. Before/after: run once with `--degradation browser-default` and
  once with the default `config`, then compare `frame_w_min` / `frame_h_min` and what you see.
  The first runs did not record frame size, so "before" here means the new browser-default run.

| Run | Landmarks | Degradation | fps p50 / min | share fps ≥ 8 | meter_ms p50 / p95 / max | pose found | face found | frames dropped | smallest frame | CPU% p50 / max | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | none | maintain-resolution | | | | n/a | n/a | | | | |
| | pose-lite | maintain-resolution | | | | | n/a | | | | |
| | pose-full | maintain-resolution | | | | | n/a | | | | |
| | pose-lite+face | maintain-resolution | | | | | | | | | |
| | pose-lite+face | browser-default | | | | | | | | | |

Pixelation before/after (owner's observation): ____

## Other findings from the spike

- **External request from the Gradio frontend: not accepted for the product.** Every page load
  fetches `iframeResizer.contentWindow.min.js` (iframe-resizer 4.3.1) from
  `cdnjs.cloudflare.com`. Gradio 5.23.1 hard-codes it in `templates/frontend/index.html` for
  resizing embedded iframes. That host is outside the allowed list (AGENTS.md rule 2 /
  FINAL_11 §4), and both real runs logged it in `*_network.csv`. Tracked in
  [issue #3](https://github.com/hemobemoo/PresenceAI/issues/3) ("must work offline before P3");
  not fixed in this spike.
- **Licence:** `soxr` (LGPL-2.1-or-later, via fastrtc → librosa) is accepted by the owner: LGPL
  used unmodified is compatible with Apache-2.0. It is allowed by name in
  `scripts/check_licenses.py` with that reason and listed in NOTICE. GPL/AGPL stay blocked.
- **Fallback 10 fps cap:** left as is unless FastRTC fails later (owner, 2026-11-12).
- Gradio analytics are off (`GRADIO_ANALYTICS_ENABLED=False`, `analytics_enabled=False`). There
  is no share link and no TURN/STUN (`rtc_configuration=None`); the server binds to 127.0.0.1.

## Decision (proposed)

**Use FastRTC** (`WebRTC`, `modality="audio-video"`, `mode="send-receive"`, small echo frame) as
the live capture transport. It passes S1 with a wide margin on both criteria (30 fps,
meter_ms p95 ≈ 70 ms), shows no backlog over 3 minutes and receives all audio. The fallback also
passes and stays the documented plan B. Runs: `20261112-181421_real`, `20261112-181657_real`.

## Consequences

- P6 builds `live/stream.py` on `AudioVideoStreamHandler` in send-receive mode with a small echo
  frame. FINAL_02 §3.1 should say "send-receive" instead of "send" (plan edit, owner's call).
- Before P6 relies on these numbers: re-measure with real MediaPipe landmarks in place of the
  stand-in, which changes CPU load and the effective fps, and check the start-up pixelation.
- Issue #3 must be closed before P3 so the app works offline.
