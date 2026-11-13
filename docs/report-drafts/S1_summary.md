# Spike S1: live capture transport

## 1. Aim

Spike S1 examined whether FastRTC, embedded in a Gradio `Blocks` page, could stream webcam and
microphone input to Python handlers in real time on the project laptop while on-screen meters
continued to update. The pass condition was taken from the project roadmap
(`docs/plan/FINAL_12_ROADMAP_AND_THESIS.md`, §1). If FastRTC failed, the plan provided a fallback
built from Gradio's own streaming audio and webcam components.

## 2. Method

**Equipment.** All runs were made on one laptop (Intel Core i7-1255U, 10 cores / 12 threads,
15.7 GB RAM, Windows 11 Pro 10.0.26300) with its built-in camera and microphone. The page was
driven by Playwright 1.65.1 in a headed Chromium 158 window. Software versions were gradio 5.23.1
and fastrtc 0.0.34.

**Transports compared.**
- *FastRTC*: WebRTC in audio-video, send-receive mode, returning a 160×90 echo frame.
- *Fallback*: `gr.Audio` and `gr.Image` streaming components, each sending a snapshot every
  0.1 s, which caps the frame rate at 10 fps by design.

**Processing.** The first runs used a stand-in for landmark detection (a downscale to 640×360 and
a grey-level mean). Later runs used MediaPipe Pose (lite or full) and Face Landmarker in video
mode on a separate worker thread that kept only the latest frame.

**Frame sampling.** Three settings were tested in turn: every 3rd frame, every 2nd frame, and
time-based sampling at a target of 15 frames per second.

**Measurement.** The script `experiments/s1_live_capture/measure.py` wrote, for each run, a
per-second server log, a per-meter-value browser log, a network log and a summary file under
`experiments/results/s1/`. The meter delay was measured from the server's publish time to the
moment the value was rendered in the browser. The first 5 s of each run were treated as warm-up
and excluded from statistics. Only counters and timestamps were recorded; no audio or video was
stored.

## 3. Pass conditions

A run passed when, over the seconds after warm-up:

1. landmark processing reached at least 8 frames per second (`video.min_fps` in
   `configs/live.yaml`) in at least 95 % of seconds; and
2. the 95th percentile of the meter delay was below 300 ms (`latency_budget_ms` in
   `configs/live.yaml`).

The 95 % share and the 5 s warm-up were set in `experiments/configs/s1_live_capture.yaml`.

## 4. Results

Table 1 lists every run. Each row is taken from the summary file
`experiments/results/s1/<run id>_summary.csv`.

*Table 1. Spike S1 runs. Landmark fps is the median and minimum over post-warm-up seconds; meter
delay p95 is in milliseconds.*

| Run id | Transport | Processing | Light | Sampling | Seconds used / total | Landmark fps (median / min) | Meter delay p95 (ms) | Face found | Result |
|---|---|---|---|---|---|---|---|---|---|
| `20261112-181421_real` | FastRTC | stand-in | room | every frame | 53 / 58 | 30 / 29 | 69.5 | n/a | Pass |
| `20261112-181421_real` | Fallback | stand-in | room | every frame | 55 / 60 | 10 / 10 | 90.7 | n/a | Pass |
| `20261112-181657_real` | FastRTC | stand-in | room | every frame | 173 / 178 | 30 / 29 | 69.3 | n/a | Pass |
| `20261112-181657_real` | Fallback | stand-in | room | every frame | 175 / 180 | 10 / 10 | 91.7 | n/a | Pass |
| `20261112-215239_real` | FastRTC | pose-lite + face | room | every 3rd frame | 53 / 58 | 10 / 9 | 72.8 | 80.5 % | Pass |
| `20261112-220536_real` | FastRTC | pose-lite + face, browser-default degradation | room | every 3rd frame | 53 / 58 | 10 / 9 | 84.2 | 100 % | Pass |
| `20261112-221317_real` | FastRTC | pose-lite + face | room | every 2nd frame | 53 / 58 | 15 / 14 | 77.6 | 100 % | Pass |
| `20261112-221704_real` | FastRTC | pose-lite + face | room | every 3rd frame | 173 / 178 | 10 / 9 | 77.9 | 97.7 % | Pass |
| `20261113-011905_real` | FastRTC | pose-lite + face | room | every 2nd frame | 173 / 178 | 15 / 14 | 84.3 | 99.8 % | Pass |
| `20261113-013737_real` | FastRTC | pose-lite + face | dim, on battery | every 2nd frame | 53 / 58 | 7 / 7 | 84.6 | 99.5 % | Fail |
| `20261113-021333_real` | FastRTC | pose-lite + face | dim, on charger | time-based, 15 fps | 53 / 58 | 14 / 12 | 74.1 | 98.4 % | Pass |

Run `20261112-215239_real` also measured three intermediate settings (no landmarks, pose-lite
only and pose-full only), each of which passed; they are listed in the same summary file and in
ADR 0002.

**Transport comparison.** Both transports passed with the stand-in processing. FastRTC delivered
30 fps with an audio real-time ratio of 1.00, meaning no audio time was lost. The fallback was
held at its 10 fps cap and had audio real-time ratios of 0.955 and 0.947, a loss of about 5 % of
audio time (runs `20261112-181421_real` and `20261112-181657_real`, column
`audio_realtime_ratio`).

**Landmark processing.** With MediaPipe pose-lite and face landmarks, every room-light run passed
on both criteria. Sampling every 3rd frame gave a minimum of 9 landmark fps, one frame above the
threshold. Sampling every 2nd frame gave a minimum of 14 fps.

**Low light.** In run `20261113-013737_real` the camera delivered only 15 frames per second in
dim light. Every-2nd-frame sampling therefore produced 7 landmark fps, and the run failed: the
threshold was met in 49 % of seconds (`share_fps_ok` = 0.491). No frames were dropped by the
landmark worker, which showed that the camera rate, not processing time, was the limiting factor.

**Time-based sampling.** Sampling was then changed so that a frame was processed whenever 1/15 s
had elapsed since the previous one (commit `9f3b530`). Under the same light, run
`20261113-021333_real` passed, with a minimum of 12 landmark fps and a meter delay p95 of 74.1 ms.
Its median CPU use (51.6 %) was similar to that of the room-light, every-2nd-frame run
`20261113-011905_real` (55.4 %), since both processed about 15 frames per second.

## 5. Decision

The project owner accepted FastRTC as the live capture transport on 2026-11-13, with landmarks
sampled by time at a target of 15 frames per second (ADR 0002, `docs/adr/0002-live-capture-transport.md`;
commit `973fae0`). The Gradio-component fallback was retained as the documented alternative.
The decision was merged into the main branch in commit `f12ec58`.

## 6. Limitations

- All runs were made on a single machine with a single camera.
- The meter delay covered only the path from server to screen. The time from camera or
  microphone to the server could not be measured, because FastRTC does not pass capture
  timestamps to the handler.
- In run `20261112-215239_real` a face was found in only 80.5 % of processed frames. The cause was
  not identified by the recorded data.
- Only received frame resolution was recorded (640×360 in every second of every landmark run).
  Visual blockiness caused by a low encoder bitrate at full resolution was not measured.
- Power state affected processing time. Run `20261113-013737_real`, on battery, had a median
  landmark time of 42.3 ms against 26.2 ms in run `20261113-021333_real`, on charger, under the
  same light. Battery operation was therefore identified as a condition that the live
  application must detect and report.
- The fallback transport was measured only with stand-in processing.
- Each configuration was measured in one or two runs, so run-to-run variation was not estimated.

## Sources

- Results: `experiments/results/s1/<run id>_summary.csv`, `_server.csv`, `_browser.csv` and
  `_network.csv` for the run ids in Table 1.
- Measurement code: `experiments/s1_live_capture/` (unit tests `tests/unit/test_s1_*.py`).
- Settings: `experiments/configs/s1_live_capture.yaml`, `configs/live.yaml`.
- Decision record: `docs/adr/0002-live-capture-transport.md`.
- Commits: `9f3b530` (time-based sampling), `973fae0` (ADR accepted), `f12ec58` (merge to main).
