# Spike S3 — Unsloth Whisper on Kaggle

Question, setup, pass rule and results: `docs/adr/0005-unsloth-whisper-kaggle.md`.
Report draft: `docs/report-drafts/S3_summary.md`.

| File | Role |
|---|---|
| `kaggle/train_kernel.py` | Kaggle script: LoRA fine-tune of whisper-small with Unsloth (adapted from the pinned notebook), merge, CTranslate2 conversion of base and fine-tuned models, 10 test clips |
| `push.py` | Builds the private job (config injected), pushes it with a T4 GPU and a time limit, shows status, fetches outputs |
| `check_local.py` | Laptop CPU check with faster-whisper: both models on the 10 clips, WER, fillers, CPU time, decision |
| `common.py` | Config, text normalisation, WER, filler counts, pass rule (unit-tested in `tests/unit/test_s3_spike.py`) |
| `../configs/s3_unsloth_whisper.yaml` | Pinned notebook commit, model and dataset revisions, seeds, training settings, pass rule |

## What is stored

- `models/s3/<run>/{base,ft}` (git-ignored): CTranslate2 int8 models. Their SHA-256 values are in
  `experiments/results/s3/<run>_models_manifest.csv`.
- `data/s3/<run>/clips` (git-ignored): the 10 public DisfluencySpeech test clips and their
  references.
- `experiments/results/s3/`: `<run>_train_log.csv`, `<run>_run_info.json`, `<run>_local.csv`,
  `<run>_local_info.json`, `<run>_decision.csv`.

No token or secret goes to Kaggle: the model and the dataset are public. The Kaggle CLI uses its
own token from `~/.kaggle`; these scripts never read it. The Kaggle user name is passed with
`--owner` and is not stored.

## Commands

```bash
uv run python experiments/s3_unsloth_whisper/push.py push   --owner <kaggle-user>
uv run python experiments/s3_unsloth_whisper/push.py status --owner <kaggle-user>
uv run python experiments/s3_unsloth_whisper/push.py fetch  --owner <kaggle-user> --run-id <id>
uv run python experiments/s3_unsloth_whisper/check_local.py --run-id <id>
```
