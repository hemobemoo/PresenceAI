# 0005 — Unsloth Whisper fine-tuning on Kaggle (spike S3)

- Status: **proposed**. The pass rule was fixed by the owner on 2026-11-14, before the run. The run
  moved from Kaggle to Google Colab on 2026-11-14 (see "Platform change").
  Results are pending.
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

## Results

Pending.
