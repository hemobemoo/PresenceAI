# FINAL 10 — Evaluation

Main evidence: system experiments on public data (E1–E8, E11, E12) and simulation (E9). Supporting evidence: an expert-agreement check on public talks (E13) and a single-case self-study (E14, n = 1). There is no multi-participant user study (ADR 0010, 2026-11-17); data sources and their terms are in `docs/data/DATA_SOURCES.md`.

## 1. System experiments

| ID | Question | Data | Metrics |
|---|---|---|---|
| E1 | ASR accuracy, EN and AR | LibriSpeech test-clean; Common Voice EN accent slices; Common Voice AR + MASC test; TED-LIUM 3 test; AMI (IHM); 20 hand-corrected clips of the owner (B1) | WER per slice; gap best–worst slice |
| E2 | Rate and pause accuracy | C1 synthetic speech (EN; targets ADR 0009); TED-LIUM 3 as an approximate real-speech check; Arabic: annotated owner clips B1 only (no public Arabic source has word or pause timing, `DATA_SOURCES.md` §3), reported descriptively | wpm error %, pause F1 at 0.5/1.0 s, duration error; share of pauses labelled `mid_sentence` vs C1 ground truth and position error rate (ADR 0011); live syllable-rate vs post wpm correlation; C1 split 30 % dev / 70 % test by script, stratified by rate band and voice, scored on test only |
| E3 | **Filler detection: base vs T1** | DisfluencySpeech test, PodcastFillers test, AMI disfluency layer (40 meetings, CC BY 4.0) | Precision/recall/F1 per detector (lexicon on base, lexicon on T1, acoustic, fusion); CrisperWhisper as upper bound; LibriSpeech WER regression |
| E4 | Content detectors | C2 synthetic transcripts; public talk transcripts (TED, evaluation only) labelled by the two raters | Per-label F1 (central message, organization, hook, conclusion, supporting-material type, undefined jargon) |
| E5 | Rubric detector ablation | Criterion labels on C2 + public talk transcripts | Cohen's κ vs human for rules / Laya / LLM judge / fusion; share of feedback statements with verified evidence |
| E6 | Confidence calibration | E5 data | Reliability diagram, expected calibration error, before/after re-estimating detector reliabilities |
| E7 | **Vision robustness and T3 gaze accuracy** | Owner's B2 protocol (lighting, glasses, camera height, distance, seated/standing); Columbia Gaze for the cross-person check | Gaze-zone macro-F1 on held-out B2 conditions and, cross-person, on Columbia Gaze (zones from known gaze angles by a rule fixed before evaluation) over `at_camera` (lens + screen area), `down_notes`, `away` (ADR 0004), with the excluded (`unknown`) share per zone and condition; per-condition change of framing/head-pose/posture measures; expression-label agreement with 2 human annotators on 300 sampled segments. **T3 is trained on one person (the owner) and tested across people only on Columbia Gaze, which has posed still images, not webcam video, so its cross-person accuracy is an approximation** (`docs/cards/LIMITATIONS.md`); AMI focus-of-attention video is a sanity check only |
| E8 | Latency and live performance | 1 / 3 / 5 min sessions on `lite`, `standard`, `space` | Landmark fps, prompt delay, post-analysis seconds per stage, LLM wait and rate-limit counts, ZeroGPU seconds per analysis |
| E9 | **Adaptive vs fixed in simulation** | C3 synthetic learners; noise per skill from E1–E7 on public data | Adaptive, fixed, random, oracle: sessions to target, final skill, regret vs oracle; sweeps over measurement noise and wrong noise assumptions. **Main evaluation of C1** |
| E11 | Coach-SLM vs hosted LLM vs templates | 150 held-out assessments | Quote found %, numbers correct %, guardrail violations, JSON validity, blind actionability rating (1–5, 2 raters), explains the benefit to the user (rater criterion, `feedback_point_rule`), Arabic items: code-switching rule followed (rater criterion, FINAL_04 §5); rater guide and page in P7, laptop latency |
| E12 | Reference ranges from published talks | ~30 curated TED talks (derived features only) | Distributions of rate, pauses, gaze-at-camera share, gesture rate; reported descriptively, never as targets |
| E13 | **Engine vs expert raters** | ~40 public talks (TED, evaluation only) | Weakest-skill and exercise choice: Cohen's κ rater–rater and each rater–engine, Krippendorff's α across all three, bootstrap 95 % CIs (§3) |
| E14 | Single-case self-study | The owner, ~10 sessions, ABAB (adaptive / fixed blocks) | Per-session measures; level and trend per block; visual analysis; n = 1, no generalisation (§3) |

Every experiment script takes `--config` and `--seed`, writes to `experiments/results/` with the git commit hash, and can be re-run from a clean clone.

## 2. Fairness and robustness slices

| Factor | Where tested | If it fails |
|---|---|---|
| Accent (EN) | E1, E3 | Confidence drops automatically when ASR word probability is low; report it |
| Arabic | E1 (Common Voice Arabic, MASC), E2 (owner only) | Arabic shows basic measures only (by design); state WER; Arabic rate and pause accuracy is single-speaker and descriptive |
| Speaking rate | E2 (110–200 wpm) | Targets are always relative to the user's own band |
| Lighting, camera position, glasses, background, posture | E7 | Disable the affected measure under that condition |
| Camera-off | UI test + camera-off sessions in E14 | Must complete the full journey without penalty |
| Expression labels across faces | E7 annotator agreement | Raise thresholds or drop the label |

Groups are never ranked by performance; the purpose is to find measurement failures.

## 3. Human checks without a user study (E13, E14)

There is no multi-participant user study and no participant recruitment (ADR 0010, 2026-11-17). E10 and its power analysis are removed. C1's main evidence is the simulator (E9, FINAL_05 §8); two human checks complement it.

### E13 — Engine vs expert raters on public talks

- **Material:** ~40 public talks from approved sources (`docs/data/DATA_SOURCES.md`; TED, evaluation only), mixed speakers and topics, including Arabic-language talks where available.
- **Engine:** the full measurement pipeline runs on each talk; the engine outputs its weakest reliable skill and the exercise it would assign.
- **Raters:** two (the owner and one classmate) independently choose the weakest skill and the most useful next exercise for each talk from the same skill list and exercise library, blind to the engine's output and to each other. Calibration round on 5 talks not in the set; both rounds reported.
- **Analysis:** Cohen's κ rater–rater and each rater–engine; Krippendorff's α across all three; bootstrap 95 % CIs; confusion of skill choices. If rater–rater κ < 0.4, the result is reported as rater disagreement, not as engine error, and the engine is not tuned to the raters.
- Raters rate published talks; nobody is recorded, so this is not a human-participant study.

### E14 — Single-case self-study (n = 1)

- **Design:** the owner uses the app for ~10 sessions in an ABAB sequence: A = adaptive policy, B = fixed sequence, blocks of 2–3 sessions; the block order is fixed before the first session.
- **Measures:** the system measures of the targeted skills per session (with confidence), plus timing and live-prompt counts.
- **Analysis:** per-block level and trend, visual analysis of the time series, non-overlap of data points between phases as a descriptive index. Reported as n = 1 with no generalisation claims.
- The owner's recordings follow the app's normal rules: encrypted at write time, 30-day retention, never uploaded to training platforms.

## 4. What the thesis can claim

| If | Then you can say |
|---|---|
| E3 shows T1 raises filler F1 without WER loss | Fine-tuning made a coached behaviour measurable |
| E7 shows gaze F1 is stable across conditions except X | Gaze feedback is reliable under stated conditions for the person it was trained on; across people, only approximately (Columbia Gaze still images) |
| E9 shows adaptive wins below noise level N | Adaptivity helps when measurement error is below N, which E1–E7 show is met for skills S |
| E13 shows rater–engine agreement close to rater–rater agreement | The engine's choices are as consistent with experts as experts are with each other, on public talks |
| E14 shows higher level or steeper trend in adaptive blocks | One learner's data are consistent with the simulation; no generalisation; validation with real learners is future work |
| E11 shows Coach-SLM near parity | Feedback can run fully offline at small cost in quality |
