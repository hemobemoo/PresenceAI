# Spike S5: gaze zones from webcam landmarks

## 1. Aim

Spike S5 examined whether features computed from a laptop webcam could tell where a presenter
was looking. The features were the position of the iris relative to the eye corners and lids,
together with head orientation. Four gaze zones were distinguished: the camera lens, the screen,
notes on the desk, and away from the laptop.

The pass condition came from the project roadmap (`docs/plan/FINAL_12_ROADMAP_AND_THESIS.md`,
§1): a macro-averaged F1 score of at least 0.8 on the project owner's own recording protocol. If
the screen and the camera could not be separated, the plan provided a fallback in which the two
zones were merged.

## 2. Method

**Participant and setup.** The project owner was the only participant. The owner sat at the
project laptop as during a presentation, with the webcam at the top of the screen. Each zone had
a fixed target:

- **camera:** the webcam lens;
- **screen:** a dot on the recording page;
- **notes:** an A4 sheet lying flat on the desk, immediately to the right of the laptop, on the
  same marked spot in every session;
- **away:** one of two fixed points on the walls, about 60–90° to the left and to the right.

**Sessions.** Four sessions were recorded on 14 November 2026:

- two main sessions in normal room light, without glasses, at different times of day (A and B);
- one session in dim light, with the room light off and only the screen lit (C);
- one session in normal light, with the owner's glasses on (D).

Sessions C and D served as robustness checks.

**Schedule.** Each session lasted about 2 minutes 50 seconds. It began with a 10-second
calibration: 3 s looking at the camera, 3 s at the notes and 4 s neutral. Sixteen blocks
followed, four for each zone. Each block had three steps:

1. **Cue:** the target was named on screen for 2 s.
2. **Settle:** a beep marked the start of the hold. Its first second was discarded to allow for
   the eye movement.
3. **Hold:** the owner kept looking at the target for 7 s, until two beeps ended the block.

**Block order.** The blocks followed a balanced 4×4 Latin square (a Williams design), in which
each zone appears once in every position and each ordered pair of zones appears once as
neighbours. Each session started from a different row of the square, so no two sessions used the
same sequence. The rows used were recorded in each session's timing file. Labels came from the
schedule, so no manual annotation was needed.

**Features.** Camera frames (640×360) were processed by the MediaPipe Face Landmarker at about
14–15 frames per second. Ten features were computed per frame:

- the horizontal and vertical position of each iris;
- the opening of each eye;
- head yaw, pitch and roll;
- face size.

No image was stored. Because the repository was to become public, the per-frame features were
kept outside it. Only the timing files, the evaluation results and a SHA-256 manifest of the
feature files were committed.

**Unknown frames.** A frame was labelled unknown if no face was found, or if the eyes were
closed: the mean of the Face Landmarker's two eye-blink scores was 0.5 or more. Unknown frames
were excluded from the F1 calculation, and their share was reported for each session and zone.

**Models.** Two classifiers were compared: logistic regression on standardised features and a
histogram-based gradient-boosting classifier. Each was evaluated in four variants:

- four zones, without calibration;
- four zones, with calibration;
- three zones (screen merged into camera), without calibration;
- three zones, with calibration.

With calibration, the session's mean feature values during the camera part of the calibration
were subtracted from that session's features.

**Folds.** The main evaluation trained on session A and tested on session B, and the reverse. The
robustness evaluation trained on A and B together, and tested on C and, separately, on D.

## 3. Pass conditions

The pass rule was set by the project owner on 14 November 2026, before any session was recorded:

1. If the four-zone macro-F1 was at least 0.8, the spike passed as planned.
2. Otherwise, if the three-zone macro-F1 was at least 0.8, the spike passed with the planned
   fallback: the screen zone was merged into the camera zone.
3. If neither held, the numbers were reported and nothing was changed.

The rule was judged on the gradient-boosting classifier with calibration, using the mean
macro-F1 of the two main folds. All other combinations were reported for comparison.

## 4. Results

The figures below come from evaluation `20261114-140550`. A second run, `20261114-142816`, added
the confusion matrices, the per-zone exclusion and the calibration check, and reproduced the
first run's scores exactly. Before evaluation, the SHA-256 of every feature file was checked
against the manifest, and all four matched.

**Table 1.** Macro-F1 of the primary model (gradient boosting with calibration).

| Fold | Four zones | Three zones |
|---|---|---|
| A → B | 0.644 | 0.919 |
| B → A | 0.765 | 0.934 |
| Mean | 0.705 | 0.926 |

No model and calibration combination reached a four-zone mean of 0.8; the highest was 0.737.
Every combination reached a three-zone mean of at least 0.8, with values from 0.898 to 0.981.

**Errors.** The confusion matrices showed that the four-zone errors came mainly from the camera
and the screen being mistaken for each other: 662 of 915 misclassified frames (72 %). Merging
the two zones removed these errors. The largest remaining error was notes frames classified as
camera, with 111 frames from A → B and 81 from B → A in the three-zone model.

**Table 2.** Share of hold frames excluded as unknown, by session and zone.

| Session | Camera | Screen | Notes | Away | All |
|---|---|---|---|---|---|
| A, normal light | 0.2 % | 1.0 % | 18.5 % | 19.1 % | 9.7 % |
| B, normal light | 0.2 % | 0.7 % | 1.2 % | 0.2 % | 0.6 % |
| C, dim light | 0 % | 0 % | 1.5 % | 42.0 % | 11.0 % |
| D, glasses | 0 % | 0.7 % | 8.9 % | 68.0 % | 19.4 % |

Two patterns accounted for almost all of the exclusions:

- In the notes zone, looking down raised the eye-blink score above the threshold, so open eyes
  were counted as closed.
- In the away zone, the face detector lost the face when the head turned far to the side.

The owner suggested that reflections on the glasses caused the higher exclusion in session D.
This was not observable in the stored features. Most of D's exclusions occurred in the away
zone, and the same pattern appeared without glasses in C and A. Reflections may therefore have
contributed, but head rotation appears to have been the larger factor.

**Calibration.** Calibration did not improve the results consistently. It raised macro-F1 in
9 of 16 comparisons and lowered it in 7. For example, logistic regression from A to B with three
zones scored 0.989 without calibration and 0.828 with it.

The offset was confirmed to be computed and applied as specified, but its estimate was noisy:

- It was based on 25–32 frames from a single 3-second look.
- Its mean lay on average 0.4–1.0 standard deviations from the session's own camera frames.
- The first second of the calibration lay further away. This is consistent with the eyes still
  moving to the lens, since the calibration had no settle period.

**Robustness.** Trained on A and B, the primary model scored 0.974 (three zones) in dim light
(C). With glasses (D), it scored 0.826 with calibration and 0.887 without.

## 5. Decision

Under the pass rule fixed before recording, the spike passed with the planned fallback: the
four-zone mean of 0.705 was below 0.8, and the three-zone mean of 0.926 was above it. The rule
and the threshold were not changed after the results were known.

For the minimum viable product, the gaze labels became:

- **camera:** toward the audience, either the lens or the screen area near it;
- **notes:** looking down at notes or the keyboard;
- **away:** looking elsewhere;
- **unknown:** no face found, or eyes closed.

A separate screen label was left as future work. The change was recorded in ADR 0004 and applied
to:

- the measurement plan (FINAL_04 §4.2);
- the gaze-model plan (FINAL_06 §4);
- the evaluation plan (FINAL_10, E7);
- the shared label type in the code.

## 6. Limitations

- There was one participant, so the results show feasibility for one person only. Accuracy
  across people is tested later in experiment E7 with volunteers, by leaving one person out at a
  time.
- Gaze was prompted and held, which may differ from the brief glances made while presenting.
- Excluded frames were concentrated in the notes and away zones, so the F1 scores for those zones
  were computed on the easier frames that remained.
- The eyes-closed threshold of 0.5 was a proposed value and was not tuned. It also excluded some
  genuine notes frames.
- The calibration step was short and had no settle period, which limits what the
  with-calibration results can show.

## Sources

- Results: `experiments/results/s5/`:
  - timing files: `20261114-020534_A_blocks.csv`, `20261114-135928_B_blocks.csv`,
    `20261114-021437_C_blocks.csv` and `20261114-021934_D_blocks.csv`;
  - `features_manifest.csv`;
  - first evaluation: `20261114-140550_eval.csv`, `_excluded.csv` and `_decision.csv`;
  - second evaluation: `20261114-142816_eval.csv`, `_confusion.csv`, `_excluded_by_zone.csv`,
    `_calibration.csv` and `_decision.csv`.
- Settings: `experiments/configs/s5_gaze.yaml`.
- Code: `experiments/s5_gaze/`, with unit tests in `tests/unit/test_s5_gaze.py` and
  `tests/unit/test_gaze_zones.py`.
- Decision record: `docs/adr/0004-gaze-zone-spike.md`.
