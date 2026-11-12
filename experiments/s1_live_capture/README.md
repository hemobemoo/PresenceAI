# Spike S1 — live capture

Question, pass rule and results: `docs/adr/0002-live-capture-transport.md`.

| File | Role |
|---|---|
| `app.py` | Gradio page with two tabs: FastRTC (audio-video WebRTC) and the fallback (`gr.Audio` + `gr.Image` streaming). It shows a level meter and a frame counter, and writes one counter row per second |
| `measure.py` | Starts `app.py`, drives it in Playwright's Chromium (fake or real devices), records when each meter value appears, and writes the browser CSV, summary and network log |
| `metrics.py` | Per-second recorder, CSV I/O and the PASS rule (unit-tested in `tests/unit/test_s1_metrics.py`) |
| `landmarks.py` | MediaPipe Pose/Face in VIDEO mode on a latest-frame worker thread (unit-tested in `tests/unit/test_s1_landmarks.py`); models come from `scripts/download_models.py` |
| `../configs/s1_live_capture.yaml` | Warm-up, 95 % share, echo size, meter interval. Thresholds come from `configs/live.yaml` |

Only counters and timestamps are written. No audio, video or frames are stored, and nothing is
uploaded. The page makes one request outside 127.0.0.1, which comes from Gradio itself (see the ADR).

## Commands

```bash
uv run python experiments/s1_live_capture/measure.py --devices real --headed --seconds 60
uv run python experiments/s1_live_capture/measure.py --devices fake --seconds 20     # no hardware
uv run python experiments/s1_live_capture/app.py                                     # just look at it, http://127.0.0.1:7861
uv run pytest -m needs_browser                                                       # end-to-end check with fake devices
```

Real-device runs write to `experiments/results/s1/`; fake runs write to a temp dir unless you pass
`--out`. Keep the headed window in front while it measures: background tabs throttle animation
frames, which would inflate meter_ms.

What meter_ms does not cover: the time from camera or microphone to the Python handler. See the ADR.
