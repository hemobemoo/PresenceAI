# FINAL 10 — Evaluation

Main evidence: system experiments (E1–E8, E11, E12) and simulation (E9). Supporting evidence: an exploratory user study (E10, n ≈ 20). This weighting is agreed (round 2, Q5).

## 1. System experiments

| ID | Question | Data | Metrics |
|---|---|---|---|
| E1 | ASR accuracy, EN and AR | LibriSpeech test-clean; Common Voice EN accent slices; Common Voice AR + MASC test; 20 hand-corrected pilot clips | WER per slice; gap best–worst slice |
| E2 | Rate and pause accuracy | C1 synthetic speech (EN); annotated pilot clips (AR) | wpm error %, pause F1 at 0.5/1.0 s, duration error; live syllable-rate vs post wpm correlation |
| E3 | **Filler detection: base vs T1** | DisfluencySpeech test, PodcastFillers test, annotated pilot | Precision/recall/F1 per detector (lexicon on base, lexicon on T1, acoustic, fusion); CrisperWhisper as upper bound; LibriSpeech WER regression |
| E4 | Content detectors | C2 synthetic transcripts; annotated pilot/study transcripts | Per-label F1 (central message, organization, hook, conclusion, supporting-material type, undefined jargon) |
| E5 | Rubric detector ablation | Criterion labels on C2 + study transcripts | Cohen's κ vs human for rules / Laya / LLM judge / fusion; share of feedback statements with verified evidence |
| E6 | Confidence calibration | E5 data | Reliability diagram, expected calibration error, before/after re-estimating detector reliabilities |
| E7 | **Vision robustness and T3 gaze accuracy** | B2 protocol (lighting, glasses, camera height, distance, seated/standing) | Gaze-zone macro-F1 leave-one-person-out; per-condition change of framing/head-pose/posture measures; expression-label agreement with 2 human annotators on 300 sampled segments |
| E8 | Latency and live performance | 1 / 3 / 5 min sessions on `lite`, `standard`, `space` | Landmark fps, prompt delay, post-analysis seconds per stage, LLM wait and rate-limit counts, ZeroGPU seconds per analysis |
| E9 | **Adaptive vs fixed in simulation** | C3 synthetic learners | Sessions to target, final skill, regret vs oracle; sweeps over measurement noise and wrong noise assumptions |
| E11 | Coach-SLM vs hosted LLM vs templates | 150 held-out assessments | Quote found %, numbers correct %, guardrail violations, JSON validity, blind actionability rating (1–5, 2 raters), laptop latency |
| E12 | Reference ranges from published talks | ~30 curated TED talks (derived features only) | Distributions of rate, pauses, gaze-at-camera share, gesture rate; reported descriptively, never as targets |

Every experiment script takes `--config` and `--seed`, writes to `experiments/results/` with the git commit hash, and can be re-run from a clean clone.

## 2. Fairness and robustness slices

| Factor | Where tested | If it fails |
|---|---|---|
| Accent (EN) | E1, E3 | Confidence drops automatically when ASR word probability is low; report it |
| Arabic | E1, E2 | Arabic shows basic measures only (by design); state WER |
| Speaking rate | E2 (110–200 wpm) | Targets are always relative to the user's own band |
| Lighting, camera position, glasses, background, posture | E7 | Disable the affected measure under that condition |
| Camera-off | UI test + E10 subgroup | Must complete the full journey without penalty |
| Expression labels across faces | E7 annotator agreement | Raise thresholds or drop the label |

Groups are never ranked by performance; the purpose is to find measurement failures.

## 3. Exploratory user study (E10)

### Design

- n ≈ 20 mixed participants (students, job seekers, general learners), randomised 1:1 to **A (Adaptive)** or **F (Fixed sequence)**, stratified by self-reported first language (Arabic/other) and camera on/off.
- One session, ~45 min: consent → calibration → **T0** baseline 2-min presentation → practice block (3 exercises, ~15 min, live prompts per policy) → **T1** post 2-min presentation on a parallel topic → **T2** one follow-up question → questionnaire.
- T0/T1 topic order counterbalanced; both arms see the same report format; the only difference is exercise and live-prompt choice. This isolates the engine.
- Live prompts are off during T0 and T1 (measurement tasks) in both arms.

### Power reality (80% power, α = 0.05, computed)

| Design | n | Detectable d |
|---|---|---|
| Between arms | 10 per arm | ≈ 1.32 |
| Within person pre/post | 20 | ≈ 0.66 |

So the arm comparison is exploratory: report the adjusted difference with a 95% CI and Hedges' g; no claim of superiority from a p-value. If recruitment falls below ~14, switch to the crossover variant: each person trains one skill adaptively and another with the fixed sequence, order counterbalanced, compare within person.

### Outcomes

| Type | Measure |
|---|---|
| Primary | Change T0 → T1 in blind human-rated VALUE-adapted rubric total (mean of 2–3 raters) |
| Secondary (human) | Per-dimension ratings; T2 follow-up rating |
| Secondary (system) | Fillers/100 words (if chosen goal), supporting-material count, timing overrun, gaze-at-camera share (camera on) |
| Experience | Usefulness, "feedback pointed to something specific", live-prompt distraction (1–5), disputes filed |
| Self-report | Comfort before/after, reported as self-report only |

### Rating

Raters trained on 5 pilot clips; rating page hides arm, time point and identity; audio-only by default so camera-off participants are rated the same way. ICC(2,k) for totals, Krippendorff's α per dimension; one refinement round if α < 0.6, both rounds reported.

### Analysis (pre-registered in `docs/study/analysis_plan.md` before participant 1)

`T1 ~ arm + T0 + topic_order` (ANCOVA) → adjusted difference, 95% CI, Hedges' g; bootstrap CI as robustness; paired Wilcoxon within arm; Spearman system vs human per dimension; Bland–Altman for totals; secondary outcomes labelled descriptive. Optional: TabFM/ridge "system features → human rating" as an analysis-only check of measurement validity.

## 4. What the thesis can claim

| If | Then you can say |
|---|---|
| E3 shows T1 raises filler F1 without WER loss | Fine-tuning made a coached behaviour measurable |
| E7 shows gaze F1 is stable across conditions except X | Gaze feedback is reliable under stated conditions |
| E9 shows adaptive wins below noise level N | Adaptivity helps when measurement error is below N, which E1–E7 show is met for skills S |
| E10 shows positive within-person gains and an adaptive effect with a wide CI | Encouraging, exploratory evidence; a larger study is needed |
| E11 shows Coach-SLM near parity | Feedback can run fully offline at small cost in quality |
