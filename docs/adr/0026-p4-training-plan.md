# 0026 — P4 training plan: scope, data, pre-registered rules, T2 schedule

- Status: **accepted** (owner, 2026-11-27; answers D1–D11 and the open items written in)
- Date: 2026-11-27
- Decider: project owner
- Plan reference: FINAL_12 §2–3 (P4–P7 rows, cut order), FINAL_06 §2–6, FINAL_10 E3/E7/E11,
  FINAL_07 D, `docs/data/DATA_SOURCES.md` §1–4; ADRs 0004, 0005, 0010, 0012, 0016, 0021.
  Working plan: `docs/plan/P4_TRAINING_PLAN.md`.
- Config keys affected: none in this PR. The numbers below move into `configs/evaluation.yaml`
  (T1 pass rule, T3 split, T2 pilot keep-rate) in the PR that first uses them, citing this ADR.

## Context

P3 is complete (main `37225b8`). FINAL_12 puts T1, T3 and T2 in P4. Reviewing the plan against
the repository showed: (1) T2 had no time slot before the P7 feature freeze; (2) T3's S5-derived
rules would be tuned and tested on the same sessions; (3) the AMI disfluency layer covers only
the ten sessions already used by E3; (4) the plan texts disagreed with ADR 0010, ADR 0016 and
`DATA_SOURCES.md` in several places. The owner answered D1–D11 on 2026-11-27.

## Decision

### 1. Ground rules for P4

1. **No training on the laptop.** Training runs on Google Colab (free T4); Kaggle only when Colab
   has no GPU. **One exception (D9): T3**, a scikit-learn fit on feature CSVs that takes seconds,
   loads no real model and keeps face-derived numbers off Google. The owner is warned before it
   runs, as before any local real-model run.
2. **Downloads on Colab only**, one approved source at a time, hash-checked, re-downloaded per
   session (D2). Google Drive holds only checkpoints and results.
3. **Nothing from B1 or any owner recording leaves the laptop** (ADR 0016).
4. Every run has a fixed seed, a data-manifest hash and a `run_info.json` with the SHA-256 of
   every output; results come back as an owner-downloaded zip checked by an `ingest` step
   (ADR 0005).
5. **Every pass rule is fixed before its run** and never tuned to pass (AGENTS.md rule 7).
6. **Result zips** (owner, 2026-11-27, answer to open item 1): the owner downloads every Colab
   result zip into **`data/colab_results/`** inside the repository (git-ignored by `data/*`, not
   a rule-9 path; registered in `data/README.md`). It is the **default input path of every
   `ingest` step**. Ingest checks every file against `run_info.json` SHA-256, copies the checked
   outputs to `models/` (model files) or `experiments/results/<run>/` (metrics and small files),
   then deletes the zip and its extracted copy, as in S3 (ADR 0005). `PRESENCEAI_HOME` is not
   used for this, and AGENTS.md rule 9 gets no exception.

### 2. Scope per track

| Track | P4 | Later |
|---|---|---|
| T1 | 2–3 runs, E3 first numbers, T1 loads in faster-whisper | App wiring after E3 shows a gain |
| T3 | v0 on S5 sessions, development F1 | v1 retrained after B2 (P6), held-out F1 |
| T2 | Teacher decision, pair builder/filter, prompt builder, 30-item teacher and student pilots, E11 held-out draw | Full pairs P5, QLoRA P5–P6 (§6) |

### 3. T1 data (D1, D3, review change 3)

- **Approved downloads (D1):** AMI from the official Edinburgh site only (IHM audio for the
  selected meetings + manual annotations v1.6.2; the `LICENCE` file in the zip is read and its
  wording recorded in `data/README.md` before any other file is used) and LibriSpeech test-clean,
  now. PodcastFillers test with PR-3 (the base E3 run needs it). CrisperWhisper approved,
  optional, E3 upper bound only.
- **Training:** DisfluencySpeech train (rev `b7da294`) and, in T1-r2, AMI meetings **outside** the
  ten disfluency-layer sessions (E3 test ES2002, ES2006, ES2008, IS1003, IS1009, TS3006, TS3007;
  E3 dev ES2005, IS1001, TS3005), whose speakers appear in no E3 session (checked from the corpus
  metadata, recorded in ADR 0012).
- **AMI gives only the word-level orthographic transcripts** for those meetings, so:
  - AMI filler spellings ("um", "uh", "mm-hmm", …) and partial-word notation are mapped onto the
    E3 filler lexicon and DisfluencySpeech conventions, with fixture tests (PR-2);
  - the AMI training hours are a fixed number written into the manifest (a free T4 cannot take
    all ~100 h), chosen in PR-2 before training;
  - **AMI gives only a weak repetition signal** (no disfluency layer for training meetings).
    This is a T1 limitation.
- **Dropped (D3):** the owner's pilot recordings are not T1 training data. B1 stays laptop-only
  and an evaluation set. Recorded as a T1 limitation.

### 4. T1 runs and pass rule (D4, D5, D6, review change 4)

Runs: **T1-r1** whisper-small on DisfluencySpeech; **T1-r2** whisper-small on DisfluencySpeech +
AMI; **T1-r3** (optional) turbo, only if time and GPU allow (D5).

Pass rule (**proposed**, fixed before T1-r1):

- **Deciding sets:** AMI test sessions and PodcastFillers test. DisfluencySpeech test is reported
  but does not decide (one speaker).
- **Pass** only if all hold:
  1. filler F1 ≥ base + **0.05** absolute on **both** deciding sets;
  2. filler F1 above the **base + `initial_prompt`** condition (FINAL_10 E3) on both deciding sets;
  3. filler precision no more than **0.02** below base on both deciding sets;
  4. repetition recall on AMI test ≥ base;
  5. LibriSpeech test-clean WER ≤ base + **1.0** point absolute.
- **Model pick:** filler F1 on DisfluencySpeech validation + AMI dev sessions (ES2005, IS1001,
  TS3005); ties broken by lower WER. **Never pick on test.**
- **Uncertainty:** bootstrap 95 % confidence intervals, resampled by meeting (by episode for
  PodcastFillers).
- **Load check (D6):** the pass rule's "loads in faster-whisper" runs on Colab CPU. One laptop
  latency check at the end, with a warning first.

### 5. T3 v0 (D9, review change 2)

- Data: S5 sessions A–D (ADR 0004). Labels `at_camera`, `down_notes`, `away` (+ `unknown`).
- **Split rule, fixed before the run:** the ADR 0004 item 1 (lost face after a head turn) and
  item 2 (head pitch vs eyes closed) thresholds are set on **sessions A and B only**. C (dim) and
  D (glasses) are reported as robustness results. The v0 leave-one-session-out macro-F1 is
  labelled **"development"**. The held-out T3 number comes from B2 in P6.
- Item 3 (longer calibration with a settle period) needs new recordings and goes with B2 (P6).
- Evaluation is leave-one-session-out, not leave-one-person-out (one person, ADR 0010).
- Runs on the laptop under the rule-1 exception, after a warning.

### 6. T2 (D7, D8, D11)

- **Teacher (D7): option A**, Colab T4, `Qwen/Qwen3.5-9B` @`c202236235762e1c871ad0ccb60c8ee5ba337b9a`,
  4-bit, batch, no public endpoint (licence and hashes verified in ADR 0021). **Move to option B**
  (Kaggle TPU v5e-8, `Qwen/Qwen3.8-27B`) if the 30-item teacher pilot **keeps fewer than 70 %** of
  items after **at most one** prompt fix. Threshold fixed now, before the pilot. Option B needs its
  own LICENSE-file and hash check before use.
  - Name check (2026-11-27, Hugging Face Hub): `Qwen/Qwen3.8-27B` exists (27.8 B parameters,
    card tag `apache-2.0`); so does `Qwen/Qwen3.5-27B`, which ADR 0021 compared. Only the card
    tag was read; the LICENSE file and hashes are checked only if option B is used.
- **Pilots:** pair filter = valid JSON, every quote found in the transcript, numbers match the
  input, no banned term (`coach/guardrails.py`), Arabic `latin_outside_transcript` empty. Student
  pilot: Qwen3.5 0.8B / 2B (licences to confirm), 30 items, on Colab.
- **E11 held-out set (D8), fixed now:** **150 assessments from the C2 test part**, 30 per planted
  group (279 test items: 56/56/55/56/56), drawn with `random.Random(20261127)` over ids sorted
  within each group. The id list is written by a script in PR-7 and committed **before any teacher
  call**. These 150 are **never given to the teacher and never used in training**. They may
  overlap the later E4 60-item draw. T2 training inputs are C2 dev plus new synthetic inputs
  (C4); the rest of the C2 test part is not used.
- **Prompt builder (D8):** built in PR-7 from the existing `feedback_point_rule` and
  `language_rule` in `lang/en/coach_prompt.yaml` and `lang/ar/coach_prompt.yaml`, so T2 no longer
  waits for P7.
- **Schedule (D11, option a):** the prompt builder and a **minimal exercise-wording set** are
  pulled forward; **full T2 pairs in P5 and QLoRA training in P5–P6**, before the P7 freeze. If
  that slips past the end of P6, T2 falls back to cut-order item 2: the pilots are the thesis
  result, and the app keeps OpenRouter + templates.

### 7. Plan-text fixes in this PR (D10)

- FINAL_07 D: AMI from the official Edinburgh files only (not the HF mirror); B1 no longer listed
  as T1 data.
- FINAL_06: §2 rule (owner recordings stay on the laptop); T1 training data and held-out rows;
  T3 evaluation (leave-one-session-out) and "Runs on" (D9 exception); T2 teacher, data and
  schedule; §6 prompt-builder timing.
- FINAL_12: P4–P7 rows (T2 scope and schedule, T3 v0/v1), cut-order item 2.
- ADR 0004: limitation line (no volunteers since ADR 0010).
- ADR 0021 §6: teacher decided here.

### 8. PR order

1. PR-1 docs: this ADR, the P4 plan, plan-text fixes.
2. PR-2 data: `scripts/fetch_public_data.py` (Colab-only, hash checks), `data/README.md` rows,
   AMI LICENCE record and speaker check (ADR 0012), AMI filler/partial-word mapping with fixture
   tests, T1 manifest builder with the fixed AMI hours.
3. PR-3 E3 harness + base numbers (base and base + `initial_prompt`), with PodcastFillers.
4. PR-4 T1-r1.
5. PR-5 T1-r2 (+ r3), model pick on dev, private HF repo, `configs/models.yaml: t1_verbatim`,
   `download_models.py`, model card draft.
6. PR-6 T3 v0 (independent; any time after PR-1).
7. PR-7 T2-a: prompt builder, pair schemas in `schemas/`, `build_pairs` / `filter_pairs`, E11 draw,
   30-item teacher and student pilots.
8. P5–P6: minimal exercise-wording set, full pairs, ~200 gold pairs by the owner, QLoRA, GGUF.

## Consequences

- T1 has a decision rule that a plain-base win cannot pass: it must also beat the base +
  `initial_prompt` condition on multi-speaker data.
- T3's P4 number is development only; the thesis reports the B2 number as held-out.
- T2 either trains before the freeze or is reported as pilots only; no T2 work happens in P8.
- **Former open items, answered by the owner on 2026-11-27:**
  1. **Result zips:** `data/colab_results/` inside the repository, the ingest default path; no
     `PRESENCEAI_HOME` use and no rule-9 exception (§1 item 6).
  2. **E11 / E4 overlap allowed:** the 150 E11 items and the later 60-item E4 central-message
     draw (ADR 0021 §4d) both come from the C2 test part and **may overlap**, because neither
     trains on them. Both reports state the overlap count.
  3. **T2 training inputs:** C2 dev plus new synthetic inputs (C4) only. The rest of the C2 test
     part (outside the 150 E11 items) stays untouched, so E4 and the P7 judge keep a clean test
     part.
- **This decision is wrong if** E3 shows T1 beating base only on DisfluencySpeech (one speaker),
  or if T3 v1 on B2 falls well below the v0 development F1.
