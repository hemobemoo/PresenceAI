# E2 on the C1 test part (run f1): the single scored run and the noise conditions

Summarised from the `f1_test_*_summary.json` and `_clips.csv` files in this folder. Run on a
Colab T4 on 2026-11-17 (UTC): repo commit `e3f9190`, the app's locked dependencies, and
whisper_small converted by `scripts/download_models.py` there (`model.bin` byte-identical to the
laptop's). The ASR device was `cuda`, with the same int8 compute type. The 287 clips matched the
committed manifest. `dirty: true` in the result files comes from `download_models.py` rewriting
`models/MANIFEST.json` in the Colab clone; no code was changed.

**Device check, on the development part:**

| Device | F1 at 0.5 / 1.0 s | Duration error | wpm error |
|---|---|---|---|
| laptop CPU | 0.973 / 0.986 | 0.0799 s | 1.07 % |
| T4 GPU | 0.973 / 0.986 (identical) | 0.0799 s (identical) | 1.01 % |

Only a few punctuation-based position labels differed.

## Verdict: E2 passes (ADR 0009 targets, clean test part, scored once)

| Measure | Test part, 201 clips | Target |
|---|---|---|
| wpm error, all clips | **0.97 %** (fillers 1.12 %, no fillers 0.90 %) | ≤ 5 % |
| Pause F1 at 0.5 s, exclusion zone applied | **0.970** (without: 0.938; 351 true pauses in the zone) | ≥ 0.90 |
| Pause F1 at 1.0 s, exclusion zone applied | **0.982** (without: 0.960; 158 true pauses in the zone) | ≥ 0.90 |
| Duration error | **0.076 s** (1,932 matched pauses) | ≤ 0.10 s |

- True pauses: 1,948, including 50 of Kokoro's own pauses. Detected pauses of at least 0.25 s:
  2,468.
- Six clips have a wpm error above 5 % (the worst is s38_r2 at 16.3 %). They are not
  investigated, because the test part is not used for tuning.

## Pause position (headline measure, ADR 0014 §6)

| True \ detected | sentence | clause | mid-sentence | unknown |
|---|---|---|---|---|
| sentence | 1040 | 79 | 37 | 6 |
| clause | 184 | 259 | 47 | 3 |
| mid-sentence | 133 | 29 | 113 | 2 |

- Mid-sentence pauses labelled as a boundary: **58.9 %**.
- Boundary pauses labelled mid-sentence: **5.1 %**.
- Any mismatch: 26.9 %. Unknown: 0.6 %.
- As on the development part, the mid-sentence errors are the C1 effect noted in ADR 0014 §3:
  a phrase cut at `|` sounds like a phrase end.

## Noise conditions (descriptive, no target)

| Test part | wpm error | F1 at 0.5 / 1.0 s (without exclusion) | Duration error | Mid → boundary / boundary → mid | Measured SNR (median) |
|---|---|---|---|---|---|
| clean | 0.97 % | 0.970 / 0.982 (0.938 / 0.960) | 0.076 s | 58.9 % / 5.1 % | — |
| pink noise, 20 dB | 0.98 % | 0.960 / 0.975 (0.925 / 0.952) | 0.090 s | 56.7 % / 6.1 % | 21.1 dB |
| pink noise, 10 dB | 1.02 % | 0.959 / 0.974 (0.927 / 0.951) | 0.087 s | 55.4 % / 8.0 % | 11.8 dB |

The noise summaries and their ledger entries show `passed: true`. That was a mistake in
`e2.summarise`, which gave a verdict on any test-part run. It is fixed so that only the clean
test part is judged. The files are left as written (AGENTS.md rule 8); the noise conditions have
no verdict.

## Live syllable rate (descriptive)

Pearson r of `pace.syllables_per_s` on the clean test part:

- against the true articulation rate (pauses removed): 0.65;
- against the true wpm: 0.58.

The noise conditions give 0.65 / 0.59 (20 dB) and 0.66 / 0.60 (10 dB).
