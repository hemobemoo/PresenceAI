# 0004 — Gaze zones from webcam landmarks (spike S5)

- Status: **proposed**. The protocol, data handling and pass rule were set by the owner on
  2026-11-14. Results are pending: no session has been recorded yet.
- Date: 2026-11-14
- Decider: project owner
- Plan reference: docs/plan/FINAL_12_ROADMAP_AND_THESIS.md §1 (S5); FINAL_04 §4.2; FINAL_06 §4 (T3);
  FINAL_10 E7
- Config keys affected: `experiments/configs/s5_gaze.yaml` (spike); if the fallback applies,
  `zones` for T3 (FINAL_04 §4.2) loses `at_screen`
- Evidence: `experiments/results/s5/` (block timing files, features manifest, evaluation CSVs),
  produced by `experiments/s5_gaze/` (record.py, evaluate.py). Per-frame features stay in the
  git-ignored `data/s5/`.

## Context

S5 asks: *can iris and head-pose features separate camera / notes / away for the owner?* The pass
condition in FINAL_12 is **≥ 0.8 macro-F1 on the owner's own protocol**. If it fails, the plan
merges `at_screen` into `at_camera`.

## Protocol (owner, 2026-11-14)

- **Zones:** `at_camera` (the lens), `at_screen` (the ● on the recording page), `down_notes`,
  `away`.
- **Notes position (fixed for every session):** an A4 sheet flat on the desk, immediately to the
  right of the laptop, on the same taped spot each time.
- **Away targets:** two fixed wall points about 60–90° to the left and right. Away blocks
  alternate left, right, left, right.
- **Sessions:**

  | Session | Condition | Use |
  |---|---|---|
  | A | normal light, no glasses | main |
  | B | normal light, no glasses, different time or day | main |
  | C | dim (room light off, screen only), no glasses | robustness |
  | D | normal light, glasses on (the owner wears glasses) | robustness |

- **Schedule per session (about 2 min 50 s):**
  - Calibration: camera 3 s, notes 3 s, neutral 4 s.
  - Then 16 blocks, 4 per zone. Each block has three steps:
    - cue: 2 s, the target name on screen, not used;
    - settle: 1 s after the first beep, discarded;
    - hold: 7 s, ending with two beeps.
- **Block order:** a balanced 4×4 Latin square (Williams design): every zone once per position
  and every ordered pair of zones once as neighbours.
  - Session A uses rows 0, 1, 2, 3 for its four repetitions, B uses 1, 2, 3, 0, C uses 2, 3, 0,
    1, and D uses 3, 0, 1, 2. No two sessions share the same sequence.
  - The row used for each repetition is written to the session's timing file.
- **Labels** come from the schedule (no manual annotation).

## Data handling (owner, 2026-11-14)

- The webcam image is never written. Each sampled frame (target 15 fps, 640×360) is turned into
  features at once:
  - iris position relative to the eye corners and lids, for both eyes;
  - eyelid aperture;
  - head yaw/pitch/roll from the facial transformation matrix;
  - face size.
- Per-frame features go to `data/s5/` and are **not committed**, because the repository will
  become public.
- Committed:
  - the block timing files;
  - the evaluation CSVs;
  - `features_manifest.csv`, with the SHA-256 of every features file. `evaluate.py` refuses a
    file that does not match.

## Unknown frames (owner, 2026-11-14)

- **Rule:** frames with no face or with closed eyes are `unknown`. They are excluded from F1, and
  the excluded share is reported per session (`<stamp>_excluded.csv`).
- **Eyes closed:** the mean of the Face Landmarker's `eyeBlinkLeft` and `eyeBlinkRight`
  blendshapes is ≥ `eyes_closed_blink`. The value **0.5 is proposed**, not tuned; the owner
  decides.
  - The blendshapes are used only for this exclusion.
  - They are not an expression event, and nothing derived from them feeds a score
    (AGENTS.md rule 3).

## Evaluation (owner-approved, 2026-11-14)

- **Frames:** the hold frames of each block.
- **Calibration:** features minus the session's mean over known `calibration_camera` frames.
- **Models:** logistic regression (standardised) and HistGradientBoosting (`random_state` 0).
- **Variants:** 4 zones and 3 zones (`at_screen` merged into `at_camera`), each with and without
  calibration.
- **Folds:**
  - main: train A → test B, and train B → test A;
  - robustness: train A+B → test C, and train A+B → test D.

## Pass rule (owner, 2026-11-14)

1. 4-zone macro-F1 ≥ 0.8 → **pass** as planned (four zones).
2. Otherwise, 3-zone macro-F1 (screen merged into camera) ≥ 0.8 → **pass with the planned
   fallback**: `at_screen` is merged into `at_camera`.
3. Neither → report the numbers and **change nothing**.

**Proposed, for the owner to confirm:** the rule is judged on HistGradientBoosting with
calibration, as the mean macro-F1 of the two main folds (A → B, B → A). This follows FINAL_06 §4,
where HistGradientBoosting is the main model and the calibration offset is part of T3. All other
combinations are reported.

## Results

Not yet recorded.

A smoke test on 2026-11-14 used Playwright's Chromium with a synthetic camera, and its output was
not kept. It confirmed:
- the page, cues, beeps, timing file and manifest;
- about 11 processed frames per second.

All frames were `unknown`, as expected, because the synthetic video contains no face.

## Consequences

- **pass:** T3 keeps four zones; E7 measures it on more people (FINAL_06 §4).
- **pass with fallback:** FINAL_04 §4.2 and T3 use three zones (`at_camera` includes the screen
  area). This needs a plan update and a config change recorded here.
- **fail:** no change; the numbers are reported.
- The decision is wrong if E7 (several people, leave-one-person-out) gives a much lower
  macro-F1 than S5.
