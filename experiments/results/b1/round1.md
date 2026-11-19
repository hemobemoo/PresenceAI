# B1 round 1 (2026-11-19): 2 English + 2 Arabic clips, owner's own speech

Descriptive only (ADR 0016): no pass/fail, nothing tuned on B1. Numbers come from
`b1_20261119-190516_*` (all 4 clips). `b1_20261119-190229_*` is the owner's first run: 3 clips,
because `b1_en_01_normal` was skipped. Its annotation had lost its `labels_method` line when the
YAML was edited by hand. Its labels match `label` on every pause `label` can decide, and the
owner decided the rest by ear, so the line was put back. Commit 7354ccb, whisper_small, config
hash 36e2561db2a04161. Pause edges were not hand-checked, so there is no pause F1 or duration
(ADR 0016 amendment).

| Clip | Pauses | wpm (true) | wpm error | WER | WER without hesitations | Words per sentence mark | App position = owner label | Language detected |
|---|---|---|---|---|---|---|---|---|
| ar_01 normal | 56 | 67.9 | 1.96 % | 0.157 | 0.151 | 30.0 | 45 / 56 | ar 0.98 |
| ar_02 normal, English terms | 69 | 60.6 | 0.74 % | 0.200 | 0.194 | 44.7 | all `unknown` (sparse punctuation, ADR 0011) | ar 0.97 |
| en_01 normal | 67 | 80.5 | 0.53 % | 0.016 | 0.016 | 12.5 | 67 / 67 | **ar 0.83, false notice** |
| en_05 defense | 87 | 79.6 | 0.88 % | 0.031 | 0.027 | 17.2 | 78 / 87 | en 0.53 |

## Pause positions: app against the owner's labels

- **190 of 210 pauses agree (90 %)**, counting only pauses where the app gave a position.
  On ar_02 all 69 pauses are `unknown`: Whisper wrote fewer than 1 sentence mark per 30 words.
- **English** (154 pauses): mid-sentence labelled as a boundary: 3 of 97 (3.1 %). Boundary
  labelled mid-sentence: 4 of 57 (7.0 %). Clause labelled as sentence: 2.
- **Arabic, ar_01** (56 pauses): mid-sentence labelled as a boundary: 0 of 42. Boundary labelled
  mid-sentence: 10 of 14 (71 %). Clause labelled as sentence: 1. The app gave no
  `clause_boundary` on either Arabic clip, because Whisper wrote almost no «،».
- **Why the 20 disagree**, per pause, from the cached reports:
  - 14: the app read the same word as the owner's label, but Whisper's punctuation after it
    differs from the owner's.
  - 3: Whisper stretched a word across the pause about equally on both sides.
  - 3: the midpoint rule skipped a stretched word (ADR 0016 open item 6). Putting it on a side
    would fix 1 of these 3.
  - On 22 more pauses the overlap rule would pick another word, but the position would not
    change.

## Other checks

- ADR 0015: no word was removed after the last speech on any clip, and the quiet closings were
  kept.
- Clipping: 0 % and possible clipping 0 % on all 4 clips. SNR 56–57 dB, from one microphone.
- Language check (ADR 0016 re-check): **1 false notice in 4 clips**. en_01 was detected as
  Arabic with probability 0.83; its transcript has no Arabic letters. en_05 was detected as
  English with only 0.53. No `mismatch_min_probability` value separates these: it would have to
  be above 0.83, and the correct English clip reached only 0.53. Proposed: keep 0.5 until all
  20 clips are scored, then decide. The owner decides.
- Rate: wpm error 0.53–1.96 %. WER: English 0.016–0.031, Arabic 0.157–0.200.
