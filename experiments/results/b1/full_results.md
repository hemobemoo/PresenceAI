# B1 full results (2026-11-21): 10 English + 10 Arabic clips, owner's own speech

Run `b1_20261121-143314` (commit b35b134, whisper_small, config hash 36e2561db2a04161).
Descriptive only (ADR 0016): no pass/fail. One value was set after seeing these results, the
language-check threshold (ADR 0016 open item 7, B1-informed). All 20 clips were corrected,
labelled by `label` with every `?` decided by ear, and scored. Pause edges were not
hand-checked, so there is no pause F1 or duration error. Numbers only; no transcript text.

## Per language

| | English (10 clips) | Arabic (10 clips) |
|---|---|---|
| wpm error, mean (max) | 0.32 % (1.19 %) | 0.99 % (2.74 %) |
| WER, mean (range) | 0.8 % (0–3.1 %), a lower bound | 13.3 % (8.4–20.0 %) |
| Pauses | 484 | 492 |
| App position = owner label | 437 / 483 (90.5 %) | 43 / 56 (ar_01 only) |
| Positions `unknown` | 1 | 436 (88.6 %; all of ar_02–ar_10) |
| mid → boundary / boundary → mid | 5.2 % / 8.1 % | 0 % / 75 % |
| False language notices at 0.5 | 4 / 10 | 0 / 10 |
| Words removed after the last speech | 0 | 0 |
| Clipping / possible clipping | 0 / 0 | 0 / 0 |

wpm error mostly reflects word-count differences, because both sides use the same speech span.

## Per clip

| Clip | Pauses | wpm (true) | WER | Position agree | Language detected (p) |
|---|---|---|---|---|---|
| en_01 normal | 67 | 80.5 | 0.016 | 58 / 67 | ar 0.835 (false notice) |
| en_02 normal | 43 | 135.0 | 0.000 | 42 / 43 | ar 0.756 (false notice) |
| en_03 normal | 34 | 135.0 | 0.000 | 30 / 34 | en 0.596 |
| en_04 normal | 39 | 123.7 | 0.004 | 36 / 39 | en 0.662 |
| en_05 defense | 87 | 79.6 | 0.031 | 74 / 87 | en 0.534 |
| en_06 defense | 44 | 126.5 | 0.000 | 43 / 44 | en 0.509 |
| en_07 slow | 50 | 102.2 | 0.004 | 46 / 50 | ar 0.940 (false notice) |
| en_08 fast | 21 | 148.8 | 0.009 | 21 / 21 | en 0.815 |
| en_09 fillers | 57 | 132.8 | 0.015 | 49 / 56 (1 unknown) | en 0.583 |
| en_10 noise | 42 | 121.4 | 0.000 | 38 / 42 | ar 0.794 (false notice) |
| ar_01 normal | 56 | 67.9 | 0.157 | 43 / 56 | ar 0.976 |
| ar_02 normal, English terms | 69 | 60.6 | 0.200 | all unknown | ar 0.970 |
| ar_03 normal, long sentences | 50 | 101.8 | 0.084 | all unknown | ar 0.995 |
| ar_04 normal, long sentences | 51 | 108.0 | 0.123 | all unknown | ar 0.992 |
| ar_05 defense, English terms | 62 | 100.3 | 0.096 | all unknown | ar 0.984 |
| ar_06 defense, both | 41 | 114.6 | 0.171 | all unknown | ar 0.986 |
| ar_07 slow | 43 | 98.1 | 0.089 | all unknown | ar 0.991 |
| ar_08 fast | 22 | 118.0 | 0.154 | all unknown | ar 0.984 |
| ar_09 fillers | 59 | 111.7 | 0.107 | all unknown | ar 0.989 |
| ar_10 noise | 39 | 119.2 | 0.144 | all unknown | ar 0.984 |

## What the numbers rest on (owner's confirmations, 2026-11-21)

- **English positions and WER rest on Whisper's text, confirmed by the owner.** The owner
  changed the punctuation mark of only 15 of about 2,800 English words. They confirmed that
  Whisper's punctuation matches what they said and how they would punctuate it, and that
  `en_02` (unchanged) was listened through and correct. No anchoring check was run.
  - Labels taken from the transcript: 338 / 348 agree (97 %), close to circular.
  - **Independent English evidence: the 135 pauses decided by ear, 99 agree (73 %).** These
    are the hard cases, where Whisper stretched a word into the pause. The ADR 0014 headline
    stays C1.
- **Arabic punctuation is the owner's own**: 12–22 marks per clip against Whisper's 0–5. So
  Arabic positions are independent, but the app gives them only on ar_01. The owner's Arabic
  averages about 36 words per sentence mark (2,351 words, 66 marks), above the
  `max_words_per_sentence_mark` limit of 30 (ADR 0016 open item 8).
- **Noise clips** measured about 60 dB SNR, the same as the quiet clips, so noise
  robustness is not tested.

## Decisions (owner, 2026-11-21)

- `speech.language_check.mismatch_min_probability`: 0.5 → **0.95**, provisional and
  B1-informed. English speech analysed as Arabic is rarely caught. To be checked on L2-ARCTIC
  later, on Kaggle (ADR 0016 open item 7).
- ADR 0011 note and open item; ADR 0013 partly checked (no false alarms; real clipping
  untested); ADR 0014 note; ADR 0015 confirmed; LIMITATIONS updated; round-1 note.
