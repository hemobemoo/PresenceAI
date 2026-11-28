# T1: Verbatim-Whisper

Defined in `docs/plan/FINAL_06_MODELS_AND_TRAINING.md` §3, ADR 0026–0029. Everything heavy runs
on Colab (ADR 0026 §1); the laptop builds notebooks, ingests result zips and compares numbers.

| File | Role |
|---|---|
| `prepare_data.py`, `colab_data.py`, `colab/data_run.py` | PR-2 data run: AMI manifest, DisfluencySpeech text check (ADR 0027) |
| `colab_e3.py`, `colab/e3_run.py` | PR-3 E3 harness and base numbers (ADR 0028) |
| `colab_t1.py`, `colab/t1_run.py`, `colab/t1train.py` | PR-4 T1-r1: train, pick on dev, publish, E3, per-window (ADR 0029) |
| `t1text.py`, `ami_words.py`, `e3score.py`, `splitzip.py` | Text rules, AMI parsing, E3 scoring, split-zip reading (pure, tested) |
| `chunks.py`, `t1pick.py`, `t1merge.py` | Per-window chunks, dev pick and pass rule, LoRA merge (pure, tested) |

```
uv run python training/t1_whisper_verbatim/colab_t1.py build   --run-id <id>
uv run python training/t1_whisper_verbatim/colab_t1.py ingest  --run-id <id>
uv run python training/t1_whisper_verbatim/colab_t1.py compare --run-id <id>
```
