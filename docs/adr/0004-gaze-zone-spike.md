# 0004 — Gaze zones from webcam landmarks (spike S5)

- Status: **proposed**, waiting for the owner's acceptance.
  - The protocol, data handling and pass rule were set by the owner on 2026-11-14, before
    recording.
  - Sessions A–D were recorded on 2026-11-14.
  - Outcome: **pass with the planned fallback**.
- Date: 2026-11-14
- Decider: project owner
- Plan reference: docs/plan/FINAL_12_ROADMAP_AND_THESIS.md §1 (S5); FINAL_04 §4.2; FINAL_06 §4 (T3);
  FINAL_10 E7
- Config keys affected:
  - `experiments/configs/s5_gaze.yaml: mvp_zones`
  - `src/presenceai/schemas/features.py: GazeZone` (MVP labels)
- Evidence, all produced by `experiments/s5_gaze/` (record.py, evaluate.py):
  - Recordings (timing files): `experiments/results/s5/20261114-020534_A_blocks.csv`,
    `20261114-135928_B_blocks.csv`, `20261114-021437_C_blocks.csv` and
    `20261114-021934_D_blocks.csv`.
  - Feature files: `features_manifest.csv` holds their SHA-256. The files themselves stay in
    the git-ignored `data/s5/`.
  - Evaluation `20261114-140550`, run by the owner: `_eval.csv`, `_excluded.csv`,
    `_decision.csv`.
  - Re-run `20261114-142816`, which adds `_confusion.csv`, `_excluded_by_zone.csv` and
    `_calibration.csv`. Its `_eval.csv`, `_excluded.csv` and decision are identical to the
    first run.

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

### Checks before use

- **Checksums:** the SHA-256 of each of the four feature files matches `features_manifest.csv`,
  and the row counts match.
- **Unused files:** `data/s5/` also holds four files from interrupted attempts that are not in
  the manifest and are not used: three header-only A files and a 365-row B file.
- **No per-frame data committed:** `experiments/results/s5/` contains no per-frame features.
- **Printed numbers:** the decision printed by `evaluate.py` matches the eval CSV.
  - 4 zones: mean of 0.6444 (A→B) and 0.7645 (B→A) = 0.7045.
  - 3 zones: mean of 0.9190 and 0.9338 = 0.9264.

### Main result (primary: HistGradientBoosting with calibration)

| Folds | 4 zones | 3 zones |
|---|---|---|
| A → B | 0.6444 | 0.9190 |
| B → A | 0.7645 | 0.9338 |
| **Mean** | **0.7045** | **0.9264** |

The outcome does not depend on the combination chosen as primary:
- No model and calibration combination reaches a 4-zone mean of 0.8. The highest is logistic
  regression with calibration, at 0.7372.
- Every combination has a 3-zone mean of at least 0.8, ranging from 0.8982 to 0.9807.

### Confusion matrices (primary model, `20261114-142816_confusion.csv`; rows = true label)

4 zones, A → B:

| true \ predicted | at_camera | at_screen | away | down_notes |
|---|---|---|---|---|
| at_camera | 75 | 296 | 0 | 40 |
| at_screen | 86 | 323 | 0 | 4 |
| away | 0 | 0 | 414 | 0 |
| down_notes | 141 | 0 | 0 | 265 |

4 zones, B → A:

| true \ predicted | at_camera | at_screen | away | down_notes |
|---|---|---|---|---|
| at_camera | 354 | 26 | 11 | 18 |
| at_screen | 254 | 138 | 0 | 12 |
| away | 0 | 0 | 332 | 3 |
| down_notes | 19 | 1 | 4 | 310 |

3 zones:

| Fold | at_camera → down_notes | down_notes → at_camera | other errors | correct |
|---|---|---|---|---|
| A → B | 12 | 111 | 0 | 1521 of 1644 |
| B → A | 1 | 81 | 4 | 1396 of 1482 |

- Over both 4-zone folds, **662 of 915 errors (72 %) are camera ↔ screen**.
- Merging the two removes them, which is the planned fallback.
- The largest remaining error is `down_notes` predicted as `at_camera`: 141 and 19 frames in
  4 zones, and 111 and 81 in 3 zones.

### Excluded (`unknown`) frames per zone (`20261114-142816_excluded_by_zone.csv`)

| Session | at_camera | at_screen | down_notes | away | Whole session |
|---|---|---|---|---|---|
| A (normal) | 0.2 % | 1.0 % | **18.5 %** (eyes closed) | **19.1 %** (no face) | 9.7 % |
| B (normal) | 0.2 % | 0.7 % | 1.2 % | 0.2 % | 0.6 % |
| C (dim) | 0 % | 0 % | 1.5 % | **42.0 %** (no face) | 11.0 % |
| D (glasses) | 0 % | 0.7 % | **8.9 %** (eyes closed) | **68.0 %** (no face) | 19.4 % |

- **Session A.** The 9.7 % is not all `down_notes`. Half of it comes from `down_notes`, where
  looking down raised the eye-blink score above 0.5 (76 frames). The other half comes from
  `away`, where no face was found (78 frames).
- **Session D (glasses).** 258 of the 318 excluded frames are `away` frames with no face found.
- **The owner's hypothesis**, recorded as a hypothesis: reflections on the glasses caused the
  higher exclusion in D.
  - The data does not show reflections, because only features are stored.
  - It does show that most of D's exclusion happened while the head was turned away. The same
    no-face pattern appears without glasses in C (42 %) and A (19 %).
  - So glasses reflections may contribute. The larger driver appears to be the face detector
    losing the face when the head turns far to the side, and how far the head was turned varied
    between sessions.

### Calibration (owner's observation, checked)

- **Effect:** calibration did not help consistently. Over the 16 model × zone × fold
  comparisons, it raised macro-F1 in 9 and lowered it in 7.
- **Largest changes:**
  - logistic regression A→B, 3 zones: 0.9892 without calibration, 0.8276 with;
  - logistic regression A→B, 4 zones: 0.5532 without calibration, 0.8017 with.
- **Implementation check:**
  - The offset is computed and applied as specified: the session's own mean over known
    `calibration_camera` frames, subtracted from that session's hold frames, the same way for
    training and test sessions.
  - Unit tests cover it.
- **The estimate is noisy** (`20261114-142816_calibration.csv`):
  - It rests on only 25–32 frames from a single 3-second look.
  - Its mean lies on average 0.4 (C) to 1.0 (D) standard deviations from the same session's
    `at_camera` hold frames, and up to 2.3 SD on one feature (D, `r_iris_y`).
  - In B and D, the first second of the calibration is further off (1.6 and 1.9 SD against 0.7
    and 1.0 after it). This fits the eyes still moving to the lens, because the calibration has
    no settle period.
- **Not changed here:** the decision.
- **Proposal for T3:** use a longer calibration with a settle period, or several calibration
  looks.

### Robustness (train A+B)

| Test session | Primary, 3 zones | HistGB without calibration, 3 zones | Primary, 4 zones |
|---|---|---|---|
| C (dim light) | 0.9742 | 0.9701 | 0.8123 |
| D (glasses) | 0.8258 | 0.8865 | 0.6929 |

- Dim light did not reduce 3-zone accuracy.
- Glasses reduced it to 0.83 with calibration and 0.89 without, and raised the excluded share to
  19.4 %.

### Recording rate

All four sessions were recorded at about 14–15 processed frames per second: 2399–2493 frames over
about 170 s, per `features_manifest.csv`.

## Decision

Applying the pass rule fixed before recording, unchanged:
- The 4-zone mean of 0.7045 is below 0.8.
- The 3-zone mean of 0.9264 is at least 0.8.
- **Pass with the planned fallback.**

For the MVP, the gaze labels are:

| Label | Meaning |
|---|---|
| `at_camera` | Toward the audience: the lens or the screen area near it |
| `down_notes` | Looking down at notes or the keyboard |
| `away` | Looking elsewhere |
| `unknown` | No face found, or eyes closed |

- A separate `at_screen` label is **future work**.
- Applied in:
  - FINAL_04 §4.2;
  - FINAL_06 §4 (T3 labels and evaluation);
  - FINAL_10 E7;
  - the FINAL_12 §1 status line;
  - `GazeZone` in `src/presenceai/schemas/features.py`;
  - `mvp_zones` in `experiments/configs/s5_gaze.yaml`.
- The spike's `zones` stays at four, because it defines the recorded protocol.

## Consequences

- **T3 and E7** train and evaluate three labels, and report the excluded share per zone and
  condition.
- **For T3 to address:**
  - `down_notes` frames lost to the eyes-closed rule;
  - `away` frames with no face found;
  - the noisy calibration offset.

  In the product, a lost face while the head is turned may itself be evidence for `away`. That
  is a design question for P6, not decided here.
- **Limitation:** one participant (the owner). Cross-person accuracy is tested in E7 with
  volunteers, leave-one-person-out.
- **The decision is wrong if** E7 gives a 3-zone macro-F1 well below 0.8 across people.
