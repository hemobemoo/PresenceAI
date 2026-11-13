# Spike S5: gaze zones from webcam landmarks

> Status: method and pass conditions are complete. Sections 4 and 5 are to be completed from the
> evaluation files once sessions A–D have been recorded. No results are reported yet.

## 1. Aim

Spike S5 examined whether features computed from a laptop webcam could tell where the presenter
was looking. The features were the position of the iris relative to the eye corners and lids,
together with head orientation. Four gaze zones were distinguished: the camera lens, the screen,
notes on the desk, and away from the laptop. The pass condition was taken from the project
roadmap (`docs/plan/FINAL_12_ROADMAP_AND_THESIS.md`, §1): a macro-averaged F1 score of at least
0.8 on the project owner's own recording protocol. If the screen and the camera could not be
separated, the plan provided a fallback that merges these two zones.

## 2. Method

**Participant and setup.** The project owner was the only participant. The owner was seated at
the project laptop, as during a presentation, with the webcam at the top of the screen. Each zone
had a fixed target:

- **camera:** the webcam lens;
- **screen:** a dot on the recording page;
- **notes:** an A4 sheet lying flat on the desk, immediately to the right of the laptop, on the
  same marked spot in every session;
- **away:** one of two fixed points on the walls, about 60–90° to the left and right.

**Sessions.** Four sessions were planned:

- two main sessions in normal room light, without glasses, at different times (A and B);
- one session in dim light, with the room light off and only the screen lit (C);
- one session in normal light with the owner's glasses on (D).

C and D served as robustness checks.

**Schedule.** Each session lasted about 2 minutes 50 seconds. It began with a 10-second
calibration: 3 s looking at the camera, 3 s at the notes, and 4 s neutral. Sixteen blocks
followed, four per zone. Each block had three steps:

1. **Cue:** the name of the target was shown on screen for 2 s.
2. **Settle:** a beep marked the start of the hold, and its first second was discarded to allow
   for the eye movement.
3. **Hold:** the owner kept looking at the target for 7 s, until two beeps ended the block.

**Block order.** The order of the blocks followed a balanced 4×4 Latin square (a Williams
design). In this design, each zone appears once in every position, and each ordered pair of
zones appears once as neighbours. Each session started from a different row of the square, so no
two sessions used the same sequence. The order used was recorded in each session's timing file.
Labels came from the schedule, so no manual annotation was needed.

**Features.** Camera frames (640×360) were sampled at up to 15 per second and processed at once
by the MediaPipe Face Landmarker. Ten features were computed per frame:

- the horizontal and vertical position of each iris;
- the opening of each eye;
- head yaw, pitch and roll;
- face size.

No image was stored. Because the repository was to become public, the per-frame features were
kept outside it. Only the timing files, the evaluation results and a SHA-256 manifest of the
feature files were committed.

**Unknown frames.** A frame was labelled unknown if no face was found, or if the eyes were closed:
the mean of the Face Landmarker's two eye-blink scores was 0.5 or more. Unknown frames were
excluded from the F1 calculation, and their share was reported for each session.

**Models.** Two classifiers were compared: logistic regression on standardised features and a
histogram-based gradient-boosting classifier. Each was evaluated in four variants:

- four zones, without calibration;
- four zones, with calibration;
- three zones (screen merged into camera), without calibration;
- three zones, with calibration.

With calibration, each session's mean feature values during the camera part of the calibration
were subtracted from that session's features.

**Folds.** The main evaluation trained on session A and tested on session B, and the reverse. The
robustness evaluation trained on A and B together and tested on C, and separately on D.

## 3. Pass conditions

The pass rule was set by the project owner on 2026-11-14:

1. If the four-zone macro-F1 was at least 0.8, the spike passed as planned.
2. Otherwise, if the three-zone macro-F1 (screen merged into camera) was at least 0.8, the spike
   passed with the planned fallback.
3. If neither held, the numbers were reported and no change was made.

The rule was judged on the gradient-boosting classifier with calibration, using the mean
macro-F1 of the two main folds. This choice was proposed in ADR 0004 for the owner's
confirmation. All other combinations were reported for comparison.

## 4. Results

To be completed from `experiments/results/s5/<stamp>_eval.csv`, `_excluded.csv` and
`_decision.csv`.

## 5. Decision

To be completed after the evaluation, according to the pass rule in Section 3 (ADR 0004).

## 6. Limitations

- There was one participant, so the result shows feasibility for that person only. Accuracy
  across people is measured later in experiment E7.
- Gaze was prompted and held, which may differ from the brief glances made while presenting.
- The eyes-closed threshold of 0.5 was a proposed value and was not tuned.
- The pipeline was first tested with a synthetic camera containing no face. Every frame from
  that test was unknown, so it checked only the recording and timing path.

## Sources

- Settings: `experiments/configs/s5_gaze.yaml`.
- Code: `experiments/s5_gaze/`, with unit tests in `tests/unit/test_s5_gaze.py`.
- Results (after recording): `experiments/results/s5/`.
- Decision record: `docs/adr/0004-gaze-zone-spike.md`.
