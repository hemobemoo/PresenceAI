# 0008 — Whisper models for the app: source, pinning and conversion

- Status: **accepted**. The owner approved the source, the two models and the conversion on
  2026-11-17 (P1 decision 2), and the hf.co redirect the same day.
- Date: 2026-11-17
- Decider: project owner
- Plan reference: docs/plan/FINAL_06 §1 (Whisper `small`, `turbo`); FINAL_04 §2, §5 (ASR);
  FINAL_12 P1; ADR 0005 (S3 conversion), ADR 0006 (array input), ADR 0007 (S2 conversion)
- Config keys affected: `configs/models.yaml: asr.whisper_small`, `asr.whisper_turbo`,
  `converter`; `configs/default.yaml: asr`

## Context

FINAL_06 §1 says the app gets Whisper by "auto-download on first `WhisperModel("small")`".
AGENTS.md rule 2 allows model downloads only in `scripts/download_models.py`, so the app must never
download a model when it runs. Before this ADR, `configs/models.yaml` listed both models with
nothing pinned, and the download script allowed only Google's storage host.

## Decision

1. **Source:** `openai/whisper-small` at `973afd24965f72e36ca33b3055d56a652f456b4d` and
   `openai/whisper-large-v3-turbo` at `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, both MIT,
   fetched from `huggingface.co`. The turbo commit is the one S2 converted.
2. **Hosts:** requests go only to `https://huggingface.co`. Hugging Face serves large files
   (`model.safetensors`) by redirecting to its own CDN (`us.aws.cdn.hf.co` on 2026-11-17). Such a
   redirect is followed only to an HTTPS host under `hf.co`; any other redirect fails the download.
3. **Integrity:** before downloading, the script reads the file list of the pinned commit from
   `huggingface.co/api/models/<repo>/tree/<commit>`. Every downloaded file must match it: SHA-256
   for large (LFS) files, the git blob SHA-1 for small ones. A mismatch deletes the file and stops.
   `models/MANIFEST.json` records the SHA-256 and size of every source file and every converted
   file.
4. **Conversion:** `ct2-transformers-converter` with the same versions as S2 (ctranslate2 4.9.0,
   transformers 4.57.6, plus torch 2.15.0 CPU, which S2 left unpinned; see below). It runs in a
   throw-away `uv run --no-project` environment, so torch and transformers never become app
   dependencies. It runs offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) on the downloaded
   folder. whisper-small is stored as `int8`, like the S3 base export. turbo is stored as `float16`,
   like S2. The app loads `models/<dir>` with `local_files_only=True`, and a missing model gives
   the download command, never a download.
5. **Default model:** `asr.model: whisper_small` on CPU with `int8`. `whisper_turbo` is chosen
   with `presenceai analyze --model whisper_turbo`.

## Comparison with the S2 conversion

S2 converted turbo from the Hub id. This script converts the same commit from a downloaded folder.
Whether the converted files are byte-identical to `experiments/results/s2/model_manifest.csv` is
recorded below once the download has run.

| File | S2 SHA-256 | This conversion | Match |
|---|---|---|---|
| (filled in after the first run) | | | |

## Consequences

- `uv run python scripts/download_models.py --only whisper_small whisper_turbo` downloads about
  2.6 GB once. Source files are deleted after conversion unless `--keep-sources` is given.
- **torch version:** S2 did not record it. 2.15.0 is the CPU wheel in the local uv cache from that
  run. If a later conversion differs from the manifest, check the torch version first.
- Revisit this ADR if Hugging Face changes its CDN hosts away from `hf.co`. The download would then
  fail loudly and never fall back to another host.
