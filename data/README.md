# Dataset register

Git-ignored except this file. Every dataset used by the project gets one row here before it is downloaded.
Sources, priority order and download commands: `docs/plan/FINAL_07_DATA_SOURCES.md`.

| Id | Source | Licence | Role | Local path | Retrieved | Notes |
|---|---|---|---|---|---|---|
| colab_results | Owner-downloaded Colab result zips (P4 onward, ADR 0026 §1) | Per run (public or synthetic data only) | Inbox for training and experiment results: the default input path of every `ingest` step | `data/colab_results/` | per run | Not a dataset. Ingest checks every file against the run's `run_info.json` SHA-256, copies checked outputs to `models/` (model files) or `experiments/results/<run>/` (metrics and small files), then deletes the zip and its extracted copy (as S3, ADR 0005). Nothing stays here after ingest |
| s2_relevance | Owner-written (2026-11-24) | Owner's own | S2 relevance calibration (ADR 0022 §5): 20 question-answer pairs, 10 good, 5 evasive, 5 off-topic | `data/s2_relevance/pairs.txt` | 2026-11-24 | SHA-256 `80b97b79…a733b`. Calibration only; never in tests, never training, never released. Scored by `experiments/s2_relevance/score_pairs.py` (results: ids, labels, scores only) |
