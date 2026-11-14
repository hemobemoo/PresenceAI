# Spike S5 — gaze zones

Question, protocol, pass rule and results: `docs/adr/0004-gaze-zone-spike.md`.
Report draft: `docs/report-drafts/S5_summary.md`.

| File | Role |
|---|---|
| `protocol.py` | Session schedule (calibration + 16 blocks), balanced block order, timing file |
| `features.py` | Gaze features from the Face Landmarker (iris ratios, eyelid aperture, head yaw/pitch/roll, face size) and the `unknown` rule |
| `record.py` | Gradio + FastRTC page that runs one session, beeps, and saves features only |
| `evaluate.py` | Manifest check, excluded share per session, macro-F1 for every fold/model/zone set, decision |
| `../configs/s5_gaze.yaml` | Zones, notes position, sessions, timing, balanced rows, thresholds, pass rule |

Unit tests: `tests/unit/test_s5_gaze.py` (fake landmarks and synthetic feature files).

## What is stored

- `data/s5/<run>_<session>_features.csv`: one row per processed frame (time, segment, label,
  face found, eyes closed, ten features). **Git-ignored** (`data/*`); never committed or uploaded.
- `experiments/results/s5/<run>_<session>_blocks.csv`: the schedule, the balanced row used for each
  repetition, and when each step actually began.
- `experiments/results/s5/features_manifest.csv`: file name, row count and SHA-256 of every features
  file. `evaluate.py` refuses a features file whose SHA-256 does not match.
- `experiments/results/s5/<stamp>_eval.csv`, `_excluded.csv`, `_decision.csv`.

No image, video or audio is written. The camera stays on 127.0.0.1.

## Commands

```bash
uv run python experiments/s5_gaze/record.py --session A   # http://127.0.0.1:7861
uv run python experiments/s5_gaze/evaluate.py             # after A and B (C, D optional)
```

On the page: **Click to Access Webcam**, then **Record**. Check that the line under the heading
shows camera frames being received, then press **Start**. Beeps: one = start holding, two = block
over, three = session over.
