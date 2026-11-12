# 0002 — Live capture transport (spike S1)

- Status: proposed (draft, waiting for measurements on the target laptop)
- Date: 2026-11-12
- Decider: project owner
- Plan reference: docs/plan/FINAL_12_ROADMAP_AND_THESIS.md §1 (S1); FINAL_02 §3.1; FINAL_10 E8
- Config keys affected: `configs/live.yaml` (`video.*`, `latency_budget_ms`) once the transport is chosen
- Evidence: `experiments/results/s1/<run-id>_*` produced by `experiments/s1_live_capture/measure.py`

## Context

S1 asks: *can FastRTC inside `gr.Blocks` stream mic + webcam to Python handlers locally, with
meters updating?* Pass condition from FINAL_12: **≥ 8 fps landmarks and < 300 ms meter delay on
the laptop**. If it fails, the plan falls back to two Gradio streaming components (`gr.Audio` +
`gr.Image` webcam).

For this spike, "landmarks" is replaced by a stand-in: each received frame is downscaled to
640×360 and its grey mean computed. MediaPipe is a follow-up step.

## Options measured

1. **FastRTC** `WebRTC(modality="audio-video", mode="send-receive")` with an
   `AudioVideoStreamHandler`. FastRTC supports audio-video **only in send-receive mode**, not
   "send" as FINAL_02 assumed. The handler therefore returns a video frame: it echoes a
   **160×90** downscaled frame (`echo_size` in `experiments/configs/s1_live_capture.yaml`) to keep
   the return path light, and sends no audio. The meter is published every 100 ms through
   `AdditionalOutputs`.
2. **Fallback**: `gr.Audio(sources="microphone", streaming=True)` and
   `gr.Image(sources="webcam", streaming=True)`, each with `.stream(stream_every=0.1)`. The
   browser sends a snapshot every 0.1 s, so the frame rate is **capped at 10 fps by design**.
   That is only 2 fps of headroom over the 8 fps threshold.

## How it is measured

- **Per second, server side** (`*_server.csv`): video frames received and processed, audio
  chunks, audio seconds received, meter values published, processing ms (p50 and max), and
  process CPU% (psutil, sampled once per second; it can exceed 100 % on multi-core machines). A
  run starts at the first media event. The last second, cut short by Stop, is not reported.
- **Per meter value, browser side** (`*_browser.csv`): the page is polled once per animation
  frame, and `meter_ms = render time − server publish timestamp`. Both clocks are the same
  machine's wall clock.
- **meter_ms covers server → render only. It does NOT include camera/microphone → server**
  (capture, encoding, WebRTC jitter buffer or Gradio upload). FastRTC hands the handler a plain
  frame array without its capture timestamp, so that leg cannot be measured from Python. The lag
  trend below is the indirect check that frames are not queueing.
- **Warm-up**: the first 5 s of each transport run (`warmup_s`) stay in the per-second CSV,
  flagged `warmup=True`, but are excluded from all fps and latency statistics.
- **Lag trend**: the least-squares slope of meter_ms over the post-warm-up run (ms per minute),
  plus the audio real-time ratio (audio seconds received per wall-clock second, ≈ 1.0 when audio
  keeps up). A clearly positive slope over a 3-minute run means a growing backlog.
- Browser: Playwright's bundled Chromium. Record the version and whether it ran headed.

## PASS rule

For each transport, over post-warm-up seconds only:

> **PASS = processed fps ≥ `min_fps` in ≥ 95 % of seconds AND meter_ms p95 < `latency_budget_ms`.**

`min_fps` (8) and `latency_budget_ms` (300) are read from `configs/live.yaml`. The 95 % share
(`min_share_fps_ok`) and the 5 s warm-up are set in `experiments/configs/s1_live_capture.yaml`.
**meter_ms max is reported but not used in PASS.**

## Results (fill in from `<run-id>_summary.csv`)

Machine: ____ (CPU, RAM, OS) · Chromium: ____ · Profile (`configs/default.yaml`): ____ · Camera/mic: ____

| Run | Transport | Seconds used | fps p50 | fps min | share fps ≥ 8 | meter_ms p50 | p95 | max | lag trend (ms/min) | audio real-time ratio | CPU% p50 / max | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 s | FastRTC | | | | | | | | | | | |
| 60 s | Fallback | | | | | | | | | | | |
| 3 min | FastRTC | | | | | | | | | | | |
| 3 min | Fallback | | | | | | | | | | | |

Observations (browser warnings, stalls, audio gaps, anything felt wrong while speaking): ____

## Other findings from the spike

- **External request from the Gradio frontend:** every page load fetches
  `iframeResizer.contentWindow.min.js` from `cdnjs.cloudflare.com`. That is a host outside the
  allowed list (AGENTS.md rule 2 / FINAL_11 §4). It comes from Gradio itself, not our code;
  `*_network.csv` logs it for each run. It needs a product decision before P3 (for example,
  self-hosting the asset or accepting and documenting it).
- **Licence check:** after adding gradio, fastrtc and psutil it reports 0 FAIL and 1 WARN:
  `soxr` (LGPL-2.1-or-later, via fastrtc → librosa). LGPL is not banned by rule 6; it is
  recorded here for the licence section.
- Gradio analytics are off (`GRADIO_ANALYTICS_ENABLED=False`, `analytics_enabled=False`). There
  is no share link and no TURN/STUN (`rtc_configuration=None`); the server binds to 127.0.0.1.

## Decision

____ (FastRTC / fallback / neither; one sentence why, citing the run ids above)

## Consequences

- If FastRTC passes: P6 builds `live/stream.py` on `AudioVideoStreamHandler` in send-receive
  mode with a small echo frame. Update FINAL_02 §3.1 ("send" → "send-receive").
- If only the fallback passes: the live loop is capped at 10 fps by `stream_every`; vision runs
  on every frame, not every 2nd–3rd.
- If neither passes: revisit the budget, the processing size or the transport before P6, and
  re-run with the MediaPipe stand-in replaced by real landmarks.
- Re-measure with MediaPipe landmarks on (follow-up), since that changes CPU load.
