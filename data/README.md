# Dataset register

Git-ignored except this file. Every dataset used by the project gets one row here before it is downloaded.
Sources, priority order and download commands: `docs/plan/FINAL_07_DATA_SOURCES.md`.

| Id | Source | Licence | Role | Local path | Retrieved | Notes |
|---|---|---|---|---|---|---|
| s2_relevance | Owner-written (2026-11-24) | Owner's own | S2 relevance calibration (ADR 0022 §5): 20 question-answer pairs, 10 good, 5 evasive, 5 off-topic | `data/s2_relevance/pairs.txt` | 2026-11-24 | SHA-256 `80b97b79…a733b`. Calibration only; never in tests, never training, never released. Scored by `experiments/s2_relevance/score_pairs.py` (results: ids, labels, scores only) |
