# Spike S3 — Unsloth Whisper (Google Colab)

Question, setup, pass rule and results: `docs/adr/0005-unsloth-whisper-kaggle.md` (the file name is
kept from the original Kaggle plan; runs 1–2 on Kaggle failed or never started, so the run moved to
Colab, the plan's backup in FINAL_06 §2).
Report draft: `docs/report-drafts/S3_summary.md`.

| File | Role |
|---|---|
| `colab/train_s3.py` | Colab script: LoRA fine-tune of whisper-small with Unsloth (adapted from the pinned notebook), merge, CTranslate2 conversion of base and fine-tuned models, 10 test clips, SHA-256 of every output in `run_info.json`, last line `S3 DONE: status=ok/error` |
| `colab.py` | `build`: the script with config and build info injected, and a ready notebook (`data/s3/colab/S3_<id>.ipynb`). `ingest`: unpacks the downloaded zip, checks every checksum, then moves models, clips and results into place |
| `check_local.py` | Laptop CPU check with faster-whisper: both models on the 10 clips, WER, fillers, CPU time, decision |
| `common.py` | Config, text normalisation, WER, filler counts, pass rule (unit-tested in `tests/unit/test_s3_spike.py`) |
| `../configs/s3_unsloth_whisper.yaml` | Pinned notebook commit, model and dataset revisions, seeds, training settings, pass rule |

## What is stored

- `models/s3/<run>/{base,ft}` (git-ignored): CTranslate2 int8 models. Their SHA-256 values are in
  `experiments/results/s3/<run>_models_manifest.csv`.
- `data/s3/<run>/clips` (git-ignored): the 10 public DisfluencySpeech test clips and their
  references.
- `experiments/results/s3/`: `<run>_train_log.csv`, `<run>_run_info.json`, `<run>_local.csv`,
  `<run>_local_info.json`, `<run>_decision.csv`. `20261114-v1_run_info.json` is the failed Kaggle
  run 1.

No token or secret is used: the model and the dataset are public, Google Drive is not mounted, and
the Colab session is the owner's own browser sign-in.

## Commands

```bash
uv run python experiments/s3_unsloth_whisper/colab.py build  --run-id <id>
# run data/s3/colab/S3_<id>.ipynb on a Colab T4 (through the Colab MCP server or by hand)
uv run python experiments/s3_unsloth_whisper/colab.py ingest --run-id <id> --zip <path>/s3_<id>.zip
uv run python experiments/s3_unsloth_whisper/check_local.py --run-id <id>
```

Manual route: upload the notebook at colab.research.google.com, Runtime > Change runtime type >
T4 GPU, Runtime > Run all, save `s3_<id>.zip` when the browser asks, then run the last cell to
delete the runtime.
