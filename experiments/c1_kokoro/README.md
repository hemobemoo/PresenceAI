# C1 synthetic speech with Kokoro (Google Colab)

Design, ground truth and E2 rules: `docs/adr/0014-c1-synthetic-speech-and-e2-protocol.md`.

| File | Role |
|---|---|
| `colab/generate_c1.py` | Colab script: installs Kokoro and misaki **in the Colab runtime only**, downloads the pinned model files and checks their SHA-256, replaces misaki's espeak fallback with a guard, checks every script word against misaki's word lists, then builds clips with `presenceai.synth.c1`. Writes clips (FLAC, 24 kHz), truth JSON, `lexicon_check.json` and `run_info.json` (timings, trimming statistics, SHA-256 of every file); last line `C1 DONE: status=ok/error` |
| `colab.py` | `build`: the script with config, build info and the PresenceAI C1 code injected, plus a ready notebook (`data/c1/colab/C1_<id>.ipynb`). `ingest`: unpacks the downloaded zip, checks every checksum, then moves clips and truth files into place |

Modes: `timing` builds 5 clips (one per voice for five voices, five different target rates)
and projects the time for all 300; `full` builds all 300.

## What is stored

- `data/c1/<run>/clips` (git-ignored): the clips.
- `experiments/results/c1/<run>/truth/`: one truth file per clip.
- `experiments/results/c1/<run>_run_info.json`, `<run>_clips_manifest.csv`.
- `resources/c1/lexicon_check.json`: every script word's rating in misaki's word lists.

No token or secret is used: the model is public, Google Drive is not mounted, and the Colab
session is the owner's own browser sign-in.

## Commands

```bash
uv run python experiments/c1_kokoro/colab.py build  --run-id <id> --mode timing
# run data/c1/colab/C1_<id>.ipynb on a Colab T4 (Runtime > Run all), save c1_<id>.zip
uv run python experiments/c1_kokoro/colab.py ingest --run-id <id> --zip <path>/c1_<id>.zip
```
