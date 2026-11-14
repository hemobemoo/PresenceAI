# 0005 — Unsloth Whisper fine-tuning on Kaggle (spike S3)

- Status: **proposed**. The pass rule was fixed by the owner on 2026-11-14, before the run. The run
  moved from Kaggle to Google Colab on 2026-11-14 (see "Platform change").
  Results: pass (run `20261114-c3`); see "Results" and "Decision".
- Date: 2026-11-14
- Decider: project owner
- Plan reference: docs/plan/FINAL_12_ROADMAP_AND_THESIS.md §1 (S3); FINAL_06 §3 (T1);
  FINAL_07 (DisfluencySpeech)
- Config keys affected: `experiments/configs/s3_unsloth_whisper.yaml` (spike only)
- Evidence: `experiments/results/s3/` (written by `experiments/s3_unsloth_whisper/colab.py` and
  `check_local.py`). Model files stay in the git-ignored `models/s3/`, and test clips in the
  git-ignored `data/s3/`.

## Context

S3 asks: *does the Unsloth Whisper notebook train on 200 DisfluencySpeech clips on a free GPU, and
does the result convert to CTranslate2 and load in faster-whisper?* This is the pipeline T1
depends on. If it fails, the plan uses plain Hugging Face `Seq2SeqTrainer` with PEFT. S3 proves
only that the pipeline works without damaging the model. Filler improvement is measured in E3.

## Setup (owner-approved, 2026-11-14)

- **Source notebook:** `unslothai/notebooks` `nb/Whisper.ipynb` at commit
  `5f42bb99c0ce07b1b45d41fc9a9dc3452b889b9f` (Kaggle runs 1-2 used
  `nb/Kaggle-Whisper.ipynb`). Its training cells were adapted into a script
  (`experiments/s3_unsloth_whisper/colab/train_s3.py`, which lists every change).
- **Base model:** `unsloth/whisper-small` at revision `023f354567bbe9a875605cb8e7ef40ff9629653a`, a
  mirror of `openai/whisper-small` (MIT).
- **Data:** `amaai-lab/DisfluencySpeech` (Apache-2.0, single speaker) at revision
  `b7da294fe3a70dd96df6640893f2a5dfc2c87638`.
  - Target text: `transcript_a`, which keeps verbal disfluencies and drops non-verbal tags.
  - Clips: 200 train, 50 validation and 10 test, picked with shuffle seed 3407.
- **Training:**
  - LoRA r 64 on `q_proj` and `v_proj`, as in the notebook.
  - 100 steps at batch 1 × 4 accumulation, learning rate 1e-4.
  - Validation loss at step 0 and every 10 steps.
  - LoRA and trainer seeds are both 3407.
- **Run platform:** Google Colab, free T4. It was planned for Kaggle; see "Platform change" below.
  - No secrets are used and Google Drive is not mounted.
  - The training script has a hard time limit of 90 minutes (`timeout 5400`).
  - The runtime is deleted after the results are downloaded.
- **Export:**
  - LoRA merged to 16-bit.
  - Base and fine-tuned models both converted with `ct2-transformers-converter
    --quantization int8`.
- **Local check (owner's laptop CPU):**
  - faster-whisper, int8, beam 5, English, temperature 0, no VAD, no conditioning on previous
    text;
  - the same 10 test clips for both models.
- **Reproducibility:** the run-info file records the notebook commit, dataset revision, model
  revision, seeds, package versions, GPU and run time.

## Pass rule (owner, 2026-11-14, fixed before the run)

All three must hold:

1. **Loss falls.** Both must hold:
   - the validation loss on the 50 held-out clips at the end is lower than at step 0;
   - the mean of the last 10 logged training losses is lower than the mean of the first 10.
2. **Converts and loads.** The fine-tuned CTranslate2 model loads in faster-whisper on the owner's
   laptop CPU and returns a non-empty transcript for all 10 test clips.
3. **No damage.** The fine-tuned model's WER on the 10 test clips is **no more than 10 points**
   worse than the base model's.
   - WER is pooled over the 10 clips (total edits / total reference words × 100) against
     `transcript_a`.
   - Text is normalised: lower case; punctuation and hyphens removed; apostrophes and filler
     words kept.

**Reported for information, not part of the rule:**
- base vs fine-tuned WER and filler counts on the 10 clips;
- CPU transcription time per clip on the owner's laptop.

**If the rule fails:** report the numbers; T1 uses the fallback (Hugging Face
`Seq2SeqTrainer` + PEFT).

## Platform change: Kaggle to Colab (owner, 2026-11-14)

The setup, seeds, data and pass rule above are unchanged. Only where the script runs changed.

- **Kaggle run 1** (`20261114-v1`) failed at import within about 3 minutes.
  - Kaggle's image (Python 3.13, torch 2.11 + cu130) ships `mslk`, whose library does not load.
  - The import chain `import unsloth` → torchao → mslk then fails.
  - Record: `experiments/results/s3/20261114-v1_run_info.json`.
  - Fix: uninstall `mslk`. On Colab this runs only when `mslk` is installed and fails to import;
    any fix applied is recorded in `env_fixes`.
- **Kaggle run 2** waited about two hours for a T4 (15:22–17:25) and never started. It was
  cancelled while still queued, so it used no GPU time and produced no output.
- **Decision:** move to Google Colab (free T4), the documented backup in FINAL_06 §2.
  - Kaggle's background jobs stayed queued.
  - Nothing more is sent to Kaggle, and `push.py` was replaced by `colab.py`, which has no Kaggle
    code.
- **Notebook:** Unsloth's Colab notebook `nb/Whisper.ipynb` at the same commit (`5f42bb9`).
  - Its code cells are identical to `nb/Kaggle-Whisper.ipynb` except the install cell.
  - The Colab install cell keeps Colab's preinstalled torch.
- **Script changes:**
  - outputs in `/content/s3`;
  - `CUDA_VISIBLE_DEVICES=0`;
  - the list of changes from the notebook is now a string variable instead of `__doc__`;
  - the SHA-256 of every output file goes into `run_info.json`;
  - the last line is `S3 DONE: status=ok|error`.
- **Access:** Google's official Colab MCP server, `googlecolab/colab-mcp`.
  - Pinned to commit `b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1` (committed 2026-04-01, reviewed
    and pinned 2026-11-14).
  - Set up at user scope, not in the repository, because it uses the owner's Google sign-in.
  - Fallback: the owner runs the notebook built by `colab.py build` by hand.
- **Results transfer:** `/content/s3` is zipped and downloaded through the browser.
  - `colab.py ingest` checks every file's SHA-256 against `run_info.json` before anything is
    copied.
  - The zip is deleted afterwards.

## Runs on Colab (2026-11-14)

All three runs used the same T4 runtime, one after another, with the same seeds, data and settings.
Each run's script was checked against its build by SHA-256 before it ran. Runs c1 and c2 failed in
the export step after training. Each failure was fixed and rerun with the owner's approval.

| Run | Repo commit | Training | Export | Record |
|---|---|---|---|---|
| `20261114-c1` | `bafb786` | eval loss 2.0898 → 0.5560, 116 s | **failed**: the Unsloth merge wrote no `config.json`, so CTranslate2 could not recognise the model | `20261114-c1_run_info.json`, `_train_log.csv` |
| `20261114-c2` | `411998d` | eval loss 2.0898 → 0.5560, 97 s | fine-tuned model converted; **base failed**: the pinned base snapshot has no `tokenizer.json` | `20261114-c2_run_info.json`, `_train_log.csv` |
| `20261114-c3` | `c1cced7` | eval loss 2.0898 → 0.5559, 99 s | **both converted** | all files below |

**Export fixes.** Both are recorded in `env_fixes` in `20261114-c3_run_info.json`.
- `config.json` and `generation_config.json` are copied from the pinned base model into the merged
  model when the merge writes none. A LoRA merge does not change the architecture.
- `ct2_base` gets `tokenizer.json` from the fine-tuned export. In c2's runtime, the base tokenizer
  and that file had the same vocabulary (51,865 tokens) and gave identical token ids for test text.
  The only difference was the default `<|en|><|transcribe|>` prefix stored in the file, which
  faster-whisper sets itself.
- So `save_pretrained_merged(save_method=None)` does not give a folder CTranslate2 can convert on its
  own. T1 needs these two steps, or a merge that saves the full config.

**Laptop check.**
- faster-whisper 1.2.1 cannot decode files with the project's PyAV 19.0.2: it passes the
  `metadata_errors` argument, which PyAV 19 no longer accepts (`TypeError`).
- `check_local.py` therefore reads the 16 kHz mono WAV clips with the standard `wave` module and
  passes the audio to faster-whisper as an array, which is its documented input.
- This affects any later use of faster-whisper with file paths, and is noted for the T1/P-stage
  owner.

**Reproducibility.**
- Training repeated closely across c1–c3 with the same seeds: the first logged loss was 2.4542 in
  all three runs, and the final validation loss was 0.5560, 0.5560 and 0.5559.
- c2 and c3 reused c1's installed packages in the same runtime. Their install step re-ran the same
  commands over that environment.

## Results (run `20261114-c3`)

Sources: `experiments/results/s3/20261114-c3_run_info.json`, `_train_log.csv`,
`_models_manifest.csv`, `_local.csv`, `_local_info.json` and `_decision.csv`.

**Environment.**
- **Colab:** Tesla T4, 15,360 MiB, driver 580.82.07, CUDA 13.0; image tag
  `release-colab-external-images_20261112-060057_RC00`.
- **Packages:** Python 3.13.16; torch 2.14.1+cu130; transformers 4.57.6; Unsloth and unsloth_zoo
  2026.11.4; PEFT 0.21.2; TRL 0.22.2; datasets 3.6.0; CTranslate2 4.9.0.
- **Training:** 98.6 s, peak GPU memory reserved 0.934 GB; whole script 228 s. Merge:
  `save_pretrained_merged(save_method=None)`.
- **Transfer:** `s3_20261114-c3.zip`, 424,694,737 bytes. All 22 files matched their SHA-256 in
  `run_info.json`; the zip was then deleted.
- **Laptop:** Intel64 Family 6 Model 154, 12 logical CPUs, Windows 11; faster-whisper 1.2.1,
  CTranslate2 4.9.0, int8.

**Pass rule.**

| Condition | Value | Holds |
|---|---|---|
| 1a. Validation loss, step 0 → end | 2.0898 → 0.5559 | yes |
| 1b. Mean training loss, first 10 → last 10 steps | 1.8103 → 0.5655 | yes |
| 2. Fine-tuned model loads on the laptop CPU; 10 of 10 transcripts non-empty | loaded; 10 of 10 | yes |
| 3. Fine-tuned WER − base WER ≤ 10 points | 4.04 − 9.42 = −5.38 | yes |

**For information** (10 clips, 64.56 s of audio):
- **Fillers** (uh, um, er, erm, ah, hm, hmm, mm): reference 8, base 1, fine-tuned 8.
- **CPU time per clip, mean, including decoding:** base 3.868 s, fine-tuned 3.587 s.

## Decision

**Pass** under the rule fixed before the run (owner, 2026-11-14). The Unsloth Whisper pipeline
trains on Colab, exports to CTranslate2 and runs in faster-whisper on the laptop CPU, and the
fine-tuned model is not worse than the base model on the 10 test clips. T1 keeps the Unsloth
route; the Hugging Face `Seq2SeqTrainer` + PEFT fallback is not needed.

Limits that matter for T1 and E3:
- **One speaker.** DisfluencySpeech has a single speaker, and the test clips are that speaker. The
  lower WER and the kept fillers may reflect adaptation to that speaker and to the transcript
  style, for example "fifteen dollars" where the base model wrote "$15".
- **Small sample.** Ten clips and eight reference fillers are too few to estimate filler recall.
  That is E3's job.
- **One new error.** The fine-tuned model wrote "caught" for "called" in `test_06`.

Status stays **proposed** until the owner accepts it.
