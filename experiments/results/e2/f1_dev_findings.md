# E2 on the C1 development part (run f1): findings before the test run

Summarised from `f1_dev_clean_20261117-204853_summary.json` and `_clips.csv` (86 clips,
whisper_small, VAD minimum speech 250 ms). Development results are for tuning only; there is no
pass/fail verdict on this part (ADR 0014 §6).

## Headline numbers

| Measure | Value | ADR 0009 target |
|---|---|---|
| wpm error (all clips) | 1.07 % (fillers 1.25 %, no fillers 0.99 %) | ≤ 5 % |
| Pause F1 at 0.5 s, exclusion zone applied | 0.973 (without: 0.934; 151 true pauses in the zone) | ≥ 0.90 |
| Pause F1 at 1.0 s, exclusion zone applied | 0.986 (without: 0.967; 64 true pauses in the zone) | ≥ 0.90 |
| Duration error | 0.080 s | ≤ 0.10 s |

A VAD minimum speech of 100 ms gave F1 0.975 / 0.988 and 0.079 s; the owner kept 250 ms with
no config change (2026-11-17).

## Pause position (matched pauses, true × detected)

| True \ detected | sentence | clause | mid-sentence | unknown |
|---|---|---|---|---|
| sentence | 455 | 17 | 14 | 7 |
| clause | 81 | 103 | 20 | 1 |
| mid-sentence | 56 | 11 | 54 | 2 |

- Headline measure (ADR 0014 §6): **true mid-sentence pauses labelled as a boundary: 55.4 %**
  (67 of 121 with a known label); **true boundary pauses labelled mid-sentence: 4.9 %**
  (34 of 690).
- Any mismatch, sentence ↔ clause swaps included: 25.5 % (extra information).
- The mid-sentence errors are the effect foreseen in ADR 0014 §3: a pause at a `|` mark ends a
  separately generated phrase, so Kokoro gives it phrase-final intonation and Whisper often puts
  a full stop or comma there.

## The three clips above 5 % wpm error

Word-by-word against the script (difflib on lower-case words). None is a counting problem in
our code: rate.wpm counts Whisper's words, and the scripts have no contractions, hyphens or
digits.

| Clip | Error | Script words (+ fillers) | Transcribed | Cause |
|---|---|---|---|---|
| s10_r0 | +10.0 % | 111 | 122 | **Invented words**: after the last sentence Whisper added "thank you very much for joining us thank you" (9 words), plus a stray "and" and "I". A known Whisper failure on the silence at the end of a recording. |
| s34_r4 | −9.6 % | 113 | 102 | **Dropped words** at 197 wpm (am_fenrir): short function words ("and", "I", "a", "if", "this", "can") are missing, and some words are misheard ("decisions" → "to send", "later" → "layer"). |
| s06_r3 | −8.7 % | 110 + 2 | 102 | **Dropped and merged words** ("my topic" → "talk", "and write" → "right", "class" → "glass"); the two fillers were not transcribed (base Whisper, ADR 0012). |

No code change: these are speech-recognition errors, which E2 is meant to measure.

## Live syllable rate (descriptive)

Pearson r of `pace.syllables_per_s` against: post-analysis wpm 0.48; true wpm 0.49; **true
articulation rate (pauses removed) 0.65**.
