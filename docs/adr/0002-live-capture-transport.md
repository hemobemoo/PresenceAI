# 0002 — Live capture transport (spike S1)

- Status: **accepted** (owner, 2026-11-13), after the dim-light confirmation run passed with time-based sampling.
- Date: 2026-11-12 (proposed); 2026-11-13 (accepted)
- Decider: project owner
- Plan reference: docs/plan/FINAL_12_ROADMAP_AND_THESIS.md §1 (S1); FINAL_02 §3.1; FINAL_10 E8
- Config keys affected: none yet; `configs/live.yaml` stays as is (`video.min_fps` 8, `latency_budget_ms` 300)
- Evidence: `experiments/results/s1/20261112-181421_real_*` (60 s) and `20261112-181657_real_*` (180 s), stand-in processing; `20261112-215239_real_*`, `20261112-220536_real_*`, `20261112-221317_real_*` (60 s) and `20261112-221704_real_*` (180 s), MediaPipe landmarks; `20261113-011905_real_*` (180 s), `20261113-013737_real_*` and `20261113-021333_real_*` (60 s), confirmation runs. All produced by `experiments/s1_live_capture/measure.py`

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
server; the landmark step in these first runs is a stand-in (real landmarks: next section).

## Landmark run (follow-up)

The first runs used a stand-in for landmarks. This run replaces it with MediaPipe in VIDEO mode,
one person, models fetched and SHA-256-pinned by `scripts/download_models.py` (digests in
`models/MANIFEST.json`):

- `--landmarks none | pose-lite | pose-full | pose-lite+face`, one app start per mode, to see
  the load step by step. Face Landmarker runs with `output_face_blendshapes=False` (geometry only).
- Landmarks run on a worker thread that keeps only the latest frame. FastRTC's `video_receive`
  is never blocked, and frames that arrive while the worker is busy are counted as
  `frames_dropped`.
- Frames sent to the worker: the runs of 2026-11-12 and the first two confirmation runs used
  every Nth FastRTC frame (FINAL_02 §3.1 "every 2nd–3rd frame"), every 3rd unless marked, and
  every fallback frame. The final setting is time-based sampling at 15 fps (see "Confirmation
  runs" below). `--process-every N` forces every Nth frame.
- **Same PASS rule as above, now on landmark-processed fps.** Also reported, but not part of PASS:
  pose and face success rate (share of processed frames where a pose or face was found), share
  of frames dropped, and the smallest received frame size per second.
- **Pixelation fix:** FastRTC now sends with `rtp_params={"degradationPreference":
  "maintain-resolution"}` (spike config `rtp_params`), so under load the browser lowers frame
  rate instead of resolution. Before/after: run once with `--degradation browser-default` and
  once with the default `config`, then compare `frame_w_min` / `frame_h_min` and what you see.
  The first runs did not record frame size, so "before" here means the new browser-default run.

Same machine and browser as above, on charger, FastRTC only. Every run received 30 fps from the
camera, and the audio real-time ratio was 1.00 in every run.

**Primary result: the plan's setting, every 3rd frame (≈ 10 fps landmarks).**

| Run | Landmarks | Degradation | Seconds used / total | fps p50 / min | share fps ≥ 8 | meter_ms p50 / p95 / max | slope (ms/min) | landmark ms p50 / max | pose found | face found | frames dropped | smallest frame | CPU% p50 / max | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `215239` | none (stand-in, every frame) | maintain-resolution | 53 / 58 | 30 / 29 | 100 % | 50.9 / 68.0 / 84.0 | −2.84 | 4.3 / 15.2 | n/a | n/a | 0 % | 640×360 | 29.7 / 67.7 | ✓ |
| `215239` | pose-lite | maintain-resolution | 54 / 59 | 10 / 9 | 100 % | 54.1 / 71.0 / 72.2 | +1.99 | 16.1 / 50.7 | 100 % | n/a | 0 % | 640×360 | 35.7 / 64.1 | ✓ |
| `215239` | pose-full | maintain-resolution | 53 / 58 | 10 / 9 | 100 % | 61.3 / 78.5 / 79.7 | −0.45 | 24.2 / 50.7 | 100 % | n/a | 0 % | 640×360 | 42.2 / 59.4 | ✓ |
| `215239` | pose-lite+face | maintain-resolution | 53 / 58 | 10 / 9 | 100 % | 55.6 / 72.8 / 74.5 | +0.40 | 29.0 / 51.8 | 100 % | 80.5 % | 0 % | 640×360 | 48.4 / 65.6 | ✓ |
| `220536` | pose-lite+face | browser-default | 53 / 58 | 10 / 9 | 100 % | 50.8 / 84.2 / 85.4 | +0.46 | 27.1 / 55.7 | 100 % | 100 % | 0 % | 640×360 | 42.2 / 59.4 | ✓ |
| `221704` (180 s) | pose-lite+face | maintain-resolution | 173 / 178 | 10 / 9 | 100 % | 60.7 / 77.9 / 79.9 | −0.07 | 27.1 / 55.6 | 100 % | 97.7 % | 0 % | 640×360 | 43.8 / 70.3 | ✓ |

**Margin option: every 2nd frame (≈ 15 fps landmarks), `--process-every 2`. Chosen as the default (see below).**

| Run | Landmarks | Degradation | Seconds used / total | fps p50 / min | share fps ≥ 8 | meter_ms p50 / p95 / max | slope (ms/min) | landmark ms p50 / max | pose found | face found | frames dropped | smallest frame | CPU% p50 / max | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `221317` | pose-lite+face | maintain-resolution | 53 / 58 | 15 / 14 | 100 % | 60.4 / 77.6 / 80.4 | +0.89 | 27.5 / 52.3 | 100 % | 100 % | 0 % | 640×360 | 54.7 / 79.7 | ✓ |

Run ids are `20261112-<id>_real`; the mode is in the file name. "landmark ms" is the median of the
per-second p50 and the largest per-second max of `process_ms`, post-warm-up, from `*_server.csv`.

Reading the numbers:

- **S1 passes with real MediaPipe landmarks.** Every run passes, including pose-lite + face for
  180 s: landmark fps ≥ 8 in 100 % of seconds, meter_ms p95 ≤ 85 ms (budget 300), no backlog
  (slopes within ±2 ms/min) and no audio lost.
- **The every-3rd-frame setting passes with little margin on fps.** The camera's 30 fps gives
  10 landmark fps, and the minimum second was 9, one frame above `min_fps`. The camera would only
  need to drop below 24 fps (for example in low light, where webcams often slow down) for seconds
  to fail. That was not tested here.
- **Every 2nd frame gives 15 / 14 fps**, 6 fps above the threshold, and would tolerate a camera
  down to 16 fps. It costs about +6 CPU points at p50 (+14 at max; 100 % = one logical core of
  12) and ≈ +5 ms on meter_ms p95. It is one 60 s run.
- **Landmarks are not the bottleneck.** Pose-lite + face takes ≈ 27–29 ms per frame (p50), so the
  worker could keep up with ≈ 35 fps; no frame was dropped in any run.
- **Detection:** pose was found in 100 % of processed frames in every run. Face was found in
  80.5 % of frames in the first pose-lite + face run and 97.7–100 % in the others. The 80.5 % is
  not explained by these logs; head position or turning away are possible causes (not verified).
  These are found/not-found rates only, not landmark accuracy.

**Pixelation before/after.** Smallest received frame was **640×360 in every second of every
run**, with `maintain-resolution` and with the browser default. Owner's observation, the same for
both the browser-default run and the 180 s maintain-resolution run: *"good in terms of not having
too much pixelation, and it is good clarity"*. So the start-up pixelation from the first runs did
not reproduce, and `maintain-resolution` made no measurable difference here. The cause of the
first observation is not confirmed. Note that `frame_w_min`/`frame_h_min` only catch resolution
drops; blockiness from a low encoder bitrate at full resolution would not show in them.
`maintain-resolution` stays in the config: it costs nothing measurable and protects landmark
input under load.

**Chosen by the owner (2026-11-13): about 15 landmark fps**, i.e. every 2nd frame of a 30 fps
camera. It is inside the plan's own range (FINAL_02 §3.1, "every 2nd–3rd frame"), so it is not
a plan change. It is the only setting measured with a real fps margin, and the CPU and latency
it costs are small next to the budget. Every 3rd frame also passed, but with the low-light risk
noted above. The confirmation runs below then showed that risk.

### Confirmation runs (2026-11-13)

Owner's runs, pose-lite + face, `maintain-resolution`, FastRTC. "camera fps" is frames received
per second (p50 / min), from `*_server.csv`.

| Run | Light | Power | Sampling | Seconds used / total | camera fps | landmark fps p50 / min | share fps ≥ 8 | meter_ms p50 / p95 / max | slope (ms/min) | landmark ms p50 / max | face found | frames dropped | CPU% p50 / max | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `011905` (180 s) | normal room light | not recorded | every 2nd frame | 173 / 178 | 30 / 29 | 15 / 14 | 100 % | 56.5 / 84.3 / 92.4 | +0.32 | 27.0 / 69.1 | 99.8 % | 0 % | 55.4 / 78.1 | ✓ |
| `013737` | dim (hallway, no lights) | battery | every 2nd frame | 53 / 58 | 15 / 14 | 7 / 7 | 49 % | 58.7 / 84.6 / 89.8 | +0.55 | 42.3 / 145.0 | 99.5 % | 0 % | 60.9 / 152.3 | ✗ |
| `021333` | dim (same as `013737`) | charger | **time-based, 15 fps** | 53 / 58 | 15 / 14 | 14 / 12 | 100 % | 49.1 / 74.1 / 79.0 | −4.01 | 26.2 / 53.9 | 98.4 % | 0 % | 51.6 / 67.2 | ✓ |

Pose was found in 100 % of processed frames in all three runs. Face was found in at least 98.4 %,
above the 95 % the owner set as the flag level. The smallest received frame was 640×360
throughout. Run ids are `20261113-<id>_real`.

- **Cause of the dim-light failure:** in `013737` the camera itself delivered only 15 fps (30 in
  room light), so every 2nd frame gave 7–8 landmark fps. The landmark worker was not the
  bottleneck: no frame was dropped.
- **Fix: time-based sampling.** Frames go to the landmark worker when 1/15 s has passed since the
  last one.
  - Settings: `landmark_target_fps: 15` and `landmark_sampling_tolerance: 0.25` in
    `experiments/configs/s1_live_capture.yaml`.
  - Code: `experiments/s1_live_capture/sampling.py`, unit tests in `tests/unit/test_s1_sampling.py`.
  - A 30 fps camera still gives about every 2nd frame, and a 15 fps camera gives every frame.
  - In `021333`, at the same light and camera rate as `013737`, 96 % of received frames were
    processed and the run passed.
- **Cost of processing every frame in the dark:** CPU p50 51.6 % and 26.2 ms per landmark frame,
  against 55.4 % and 27.0 ms for every 2nd frame in room light (`011905`). Both process about 15
  frames per second, so the cost is about the same.
- **Power state is a measured factor.** In short fake-camera checks, pose-lite + face took
  ≈ 22 ms per frame on charger (2026-11-12) and ≈ 100 ms on battery (2026-11-13, CPU at 1.7 GHz).
  These used Chromium's synthetic camera for 15–20 s, written to a temporary folder and not kept.
  On battery the time was the same with every-Nth sampling, so the slowdown came from the
  machine, not the sampler.
- **The 27 → 42 ms rise in `013737`**: `013737` ran on battery (owner's note). `021333`, in the
  same light on charger, took 26.2 ms. This points to power state, not darkness, as the cause.
  The size of the battery effect differs between the real run (42 ms) and the fake-camera check
  (≈ 100 ms), so this is likely but not proven.

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

## Decision (accepted, 2026-11-13)

**Use FastRTC** (`WebRTC`, `modality="audio-video"`, `mode="send-receive"`, small echo frame) as
the live capture transport. It passes S1 with a wide margin on both criteria (30 fps,
meter_ms p95 ≈ 70 ms), shows no backlog over 3 minutes and receives all audio. With real
MediaPipe pose-lite + face landmarks it still passes, with meter_ms p95 ≤ 85 ms in every run:
10 landmark fps at every 3rd frame (60 s and 180 s) and 15 at every 2nd (60 s). **Landmarks are
sampled by time at about 15 fps** (owner's choice, 2026-11-13). That is every 2nd frame of a
30 fps camera and every frame of a camera at 15 fps or slower. This passed for 180 s in room
light, and for 60 s in dim light where the camera fell to 15 fps. The fallback also passes
(stand-in only) and stays the documented plan B. Runs: `20261112-181421_real`,
`20261112-181657_real`, `20261112-215239_real`, `20261112-220536_real`, `20261112-221317_real`,
`20261112-221704_real`, `20261113-011905_real`, `20261113-013737_real` (fail, fixed),
`20261113-021333_real`.

## Consequences

- P6 builds `live/stream.py` on `AudioVideoStreamHandler` in send-receive mode with a small echo
  frame. FINAL_02 §3.1 should say "send-receive" instead of "send" (plan edit, owner's call).
- P6 moves time-based sampling (`landmark_target_fps` 15, tolerance 0.25) and
  `degradationPreference: maintain-resolution` from the spike config into `configs/live.yaml`,
  recorded in an ADR.
- P6 follow-up, low light: the camera can be too slow for 8 landmark fps even with every frame
  processed. When that happens, show a low-light / low-camera-fps notice in the live UI and lower
  the vision confidence rating for those segments.
- P6 follow-up, power: measure one run on battery with time-based sampling. If landmarks cannot
  keep up (landmark fps below 8, or frames dropped), lower the vision confidence rating for
  those segments and show the same low-performance notice as for low light.
- Open observations, not explained by these logs:
  - the 80.5 % face-found rate in run `20261112-215239` (pose-lite + face);
  - the 27 → 42 ms rise per landmark frame in `013737`, most likely power state (see above).
- The fallback was not re-measured with real landmarks; do that only if FastRTC is dropped.
- Issue #3 must be closed before P3 so the app works offline.
