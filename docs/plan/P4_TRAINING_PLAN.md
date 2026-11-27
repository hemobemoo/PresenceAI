# P4 — Training plan

- Status: **approved with changes** (owner, 2026-11-27). Decisions D1–D11 are recorded in
  **ADR 0026 (proposed until the owner accepts it)**; numbers there are the binding version.
- Main at `37225b8` when written (P3 exit met; ADR 0024 item 1 closed).
- Sources: FINAL_12 §2–3, FINAL_06 §2–6, FINAL_10 E3/E7/E11, FINAL_07, `docs/data/DATA_SOURCES.md`,
  ADRs 0004, 0005, 0010, 0012, 0016, 0021, 0026.

## 0. Ground rules

1. **No training on the laptop.** Colab (free T4) by default, Kaggle as fallback. **Only exception:
   T3** (seconds of scikit-learn on feature CSVs, no real model; keeps face-derived numbers off
   Google). Warning first, as for every local real-model run (Whisper, e5, T1 latency check).
2. **Downloads on Colab only**, each approved source, hash-checked, **re-downloaded per session**.
   Drive holds only checkpoints and results.
3. Nothing from B1 or any owner recording leaves the laptop.
4. Fixed seed, manifest hash, `run_info.json` with SHA-256 of every output, `ingest` check of the
   downloaded zip (ADR 0005).
5. Every pass rule is fixed **before** its run.
6. Colab result zips go into **`data/colab_results/`** (git-ignored, the ingest default path),
   never the repo root. Ingest checks SHA-256 against `run_info.json`, copies outputs to
   `models/` or `experiments/results/<run>/`, then deletes the zip and its extracted copy.

## 1. Tracks

| Track | P4 | P5–P6 |
|---|---|---|
| **T1 Verbatim-Whisper** | 2–3 runs, E3 first numbers, loads in faster-whisper | App wiring after E3 shows a gain |
| **T3 Gaze-zone** | v0 on S5 A–D, **development** F1 | v1 after B2 (P6), **held-out** F1 |
| **T2 Coach-SLM** | Teacher decision, prompt builder, pair builder/filter, E11 held-out draw, 30-item pilots | P5: minimal exercise wording, full pairs, gold pairs; P5–P6: QLoRA + GGUF. Not done by end of P6 → pilots only (cut item 2) |

## 2. T1

**Data**

| Data | Role | State |
|---|---|---|
| DisfluencySpeech train/val (rev `b7da294`) | Training; validation = model pick | Ready |
| DisfluencySpeech test | E3, reported, **not deciding** (one speaker) | Ready |
| AMI IHM + manual annotations v1.6.2 (official Edinburgh only) | E3 test/dev sessions; T1-r2 training on **other** meetings | **Approved** (D1). LICENCE file read and recorded first; speaker check (ADR 0012) |
| LibriSpeech test-clean | WER regression | **Approved** (D1) |
| PodcastFillers test | E3 deciding set, never training | **Approved with PR-3** |
| CrisperWhisper 2.0 | E3 upper bound only | Approved, optional |
| Owner's pilot recordings | — | **Dropped** (D3); T1 limitation |

**AMI training gap.** The disfluency layer covers only the ten E3 sessions (test ES2002, ES2006,
ES2008, IS1003, IS1009, TS3006, TS3007; dev ES2005, IS1001, TS3005). T1-r2 therefore trains on
the word-level orthographic transcripts of other meetings with no shared speakers. PR-2: map AMI
filler spellings ("um", "uh", "mm-hmm", …) and partial-word notation to the E3 lexicon /
DisfluencySpeech conventions (fixture tests); fix the AMI training hours in the manifest; the T1
ADR states AMI gives only a weak repetition signal.

**Runs:** r1 small / DisfluencySpeech; r2 small / + AMI; r3 optional turbo.

**Pass rule (ADR 0026 §4, proposed):** deciding sets AMI test + PodcastFillers test; filler F1 ≥
base + 0.05 on both **and** above base + `initial_prompt`; precision ≥ base − 0.02; AMI repetition
recall ≥ base; LibriSpeech WER ≤ base + 1.0 point. Pick on DisfluencySpeech val + AMI dev (ties →
lower WER), never on test. Bootstrap 95 % CIs resampled by meeting. Load check on Colab CPU; one
warned laptop latency check at the end.

## 3. T3 v0

- S5 A–D feature CSVs (`data/s5/`; A, B normal light, C dim, D glasses).
- **Items 1–2 thresholds set on A and B only**; C and D reported as robustness; leave-one-session-out
  F1 labelled **development**. Held-out number from B2 in P6. Item 3 (calibration) waits for B2.
- Logistic regression baseline, HistGradientBoosting main, `skops`. Laptop, after a warning.

## 4. T2

- **Teacher:** option A, Colab T4 `Qwen/Qwen3.5-9B` @`c202236`. Switch to B (Kaggle TPU,
  `Qwen/Qwen3.8-27B`, which exists on HF with an `apache-2.0` card tag; LICENSE file + hashes
  checked before use) if the 30-item teacher pilot keeps **< 70 %** after **at most one** prompt fix.
- **Filter:** valid JSON, quotes found, numbers match, no banned term, Arabic
  `latin_outside_transcript` empty.
- **E11 held-out:** 150 C2 test assessments, 30 per planted group, `random.Random(20261127)`; id
  list committed in PR-7 before any teacher call; never to the teacher, never trained on. May
  overlap the later E4 60-item draw (overlap count reported).
- **T2 training inputs:** C2 dev plus new synthetic inputs (C4); the rest of the C2 test part stays
  untouched.
- **Prompt builder** from the existing `feedback_point_rule` / `language_rule` in
  `lang/{en,ar}/coach_prompt.yaml`, in PR-7.

## 5. Your steps by hand, and when

| When | Step |
|---|---|
| Now | Review PR-1; accept ADR 0026 (or ask for changes) |
| Yourself, any time | Delete the P3 leftovers (export zip, `app.sqlite`, `sessions/`, `exports/`, `tmp/`) |
| PR-2 run (AMI, LibriSpeech) | Open Colab, pick a **T4**, approve the colab-mcp connection (or run the built notebook by hand). No token needed: public data only. Download the result zip into `data/colab_results/` (every later run too) |
| Before PR-3 run | Nothing new (PodcastFillers already approved for PR-3). Same Colab steps |
| **Before PR-4 (T1-r1)** | On Hugging Face: create a **private model repo** for T1 (e.g. `<your-user>/presenceai-whisper-small-verbatim`); create a **fine-grained write token** limited to that repo; add it in Colab → Secrets as **`HF_TOKEN`** and turn on notebook access. I only ever refer to it by that name. Mount Drive for checkpoints (`MyDrive/presenceai_p4/`) |
| PR-4 / PR-5 runs | Start the long training cells; download result zips; read the E3 table before I propose the pick |
| PR-5 | Decide whether to run r3 (turbo); approve the laptop latency check (warned) |
| PR-6 | Approve the T3 laptop run (warned) |
| PR-7 | Colab T4 for the teacher and student pilots (teacher is public and ungated, no token). Read the pilot keep-rate |
| P5 | Write/edit ~200 T2 gold pairs. Before T2 publishing: a second private HF repo for the T2 adapter (same token rules, new token scope) |

## 6. Decisions (answered 2026-11-27)

D1 downloads as in §2 · D2 re-download per session, Drive for checkpoints/results · D3 drop pilot
recordings · D4 pass rule as in §2 (proposed) · D5 small first, turbo only as r3 · D6 Colab CPU load
check + one warned laptop latency check · D7 option A, 70 % / one prompt fix → B · D8 P4 = pilots +
builder/filter + prompt builder + E11 draw · D9 T3 v0 now, laptop exception, split rule §3 · D10
plan-text fixes in PR-1 · D11 option (a): full T2 pairs and training in P5–P6, else cut item 2.

**Former open items (answered 2026-11-27):** result zips in `data/colab_results/` (ingest default,
no rule-9 exception) · E11/E4 draw overlap allowed · T2 training inputs = C2 dev + new synthetic
inputs, rest of the C2 test part untouched.

## 7. PR order

1. **PR-1** docs: this plan, ADR 0026 (proposed), plan-text fixes (FINAL_06, FINAL_07, FINAL_12,
   ADR 0004, ADR 0021).
2. **PR-2** data: `scripts/fetch_public_data.py`, `data/README.md` rows, AMI LICENCE + speaker
   check, AMI filler/partial-word mapping with fixture tests, T1 manifest with fixed AMI hours.
3. **PR-3** E3 harness + base and base + `initial_prompt` numbers (with PodcastFillers).
4. **PR-4** T1-r1.
5. **PR-5** T1-r2 (+ r3), pick on dev, private HF repo, `configs/models.yaml: t1_verbatim`,
   `download_models.py`, model card draft.
6. **PR-6** T3 v0 (independent; any time after PR-1).
7. **PR-7** T2-a: prompt builder, pair schemas, builder/filter, E11 draw, pilots.
