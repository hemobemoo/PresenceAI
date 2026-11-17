# 0011 — Pauses and speech rate: VAD, pause positions, rate windows, syllable pace, confidence

- Status: **accepted** by the owner on 2026-11-17 (slice 2 plan, with changes 1–3 below). The C1
  split values (`dev_fraction: 0.3`, `seed: 20261117`) were confirmed by the owner the same day,
  with stratification added (see "C1 split").
- Date: 2026-11-17
- Decider: project owner
- Plan reference: docs/plan/FINAL_04 §2 (rate, pauses, pace), §5 (Arabic), §6 (confidence);
  FINAL_10 E2; ADR 0009 (E2 targets); ADR 0010 (no volunteer study)
- Config keys affected: `configs/default.yaml: speech.*`; `configs/evaluation.yaml: e2.split`

## Context

P1 slice 2 measures pauses and speech rate. Every threshold must be in `configs/` with an ADR
(AGENTS.md rule 7). The owner approved the proposed values with three changes on 2026-11-17:
commas get their own position, sparse punctuation gives `unknown` instead of `mid_sentence`, and
VAD values are tuned only on a development part of C1.

## Decision

### Voice activity detection (`speech.vad`)

Silero VAD, which ships inside faster-whisper (`silero_vad_v6.onnx`, MIT), so there is no
download and no new dependency.

| Key | Value | Why |
|---|---|---|
| `threshold` | 0.5 | Silero default |
| `min_speech_duration_ms` | 250 | Silero default (owner). 100 ms is used only if the C1 development part shows it is better |
| `min_silence_duration_ms` | 100 | Below the smallest pause threshold, so 0.25 s pauses are kept |
| `speech_pad_ms` | 0 | Padding widens speech regions and would shorten every pause by twice the padding |

faster-whisper's own VAD filter stays off during transcription (slice 1), so pauses are measured
only here.

### Pauses (`speech.pauses`)

- A pause is a gap between VAD speech regions of at least 0.25 s. Counts are reported at
  ≥ 0.25, ≥ 1.0 and ≥ 2.0 s (FINAL_04 §2), with the mean, the 90th percentile and the longest
  pause.
- **Position** comes from the last word before the pause (closing quotes and brackets are
  ignored):
  - `sentence_boundary` after `. ! ? ؟ …`;
  - `clause_boundary` after `, ، ; ؛`, reported separately and **not penalised**;
  - `mid_sentence` otherwise. Only this position may count against the speaker in later slices;
  - `unknown`, with confidence low, **when the transcript has fewer than one sentence mark per 30
    words** (`max_words_per_sentence_mark: 30`). Below that, boundaries cannot be placed
    reliably. This applies to any language; Arabic is the main case. `unknown` is also used
    when no recognised word sits on one side of the pause.
- A pause's confidence follows the lower word probability of its two neighbouring words, using
  the bands below.
- Every count has evidence: each pause's time span, plus the quoted words on either side.

### Rate (`speech.rate`)

- `rate.wpm` = words ÷ speaking time, from the first speech onset to the last offset. It is also
  given in 10 s windows with a 5 s hop, counting words by their midpoint.
- `rate.articulation_wpm` = words ÷ time inside speech regions.
- Below 10 s of speech, rate and pace are reported as **"not measurable"**, never as a low score.

### Syllable pace (`speech.syllables`)

The project's own implementation of the method published in N. H. de Jong and T. Wempe,
"Praat script to detect syllable nuclei and measure speech rate automatically," *Behavior
Research Methods*, 41(2), 385–390, 2009, doi:10.3758/BRM.41.2.385:

- an intensity peak is a syllable nucleus if it rises at least **2 dB** above the preceding dip
  (`min_dip_db`; the paper and both versions of the authors' script);
- and if it lies above the 0.99-quantile intensity minus **25 dB** (`silence_threshold_db`,
  `reference_quantile`). This threshold is from the authors' v2 script (2010, with H. Quené and
  I. Persoon), which replaced the original median-based threshold;
- the intensity window is 64 ms with a 16 ms hop, matching Praat's settings at the scripts' 50 Hz
  pitch floor;
- speech regions from the VAD stand in for the original's pitch-based voicing check. pYIN comes
  in slice 4 and may replace it.

The authors' scripts are GPL-3. No code is taken from them; only the published method and its
parameter values are used. The method was validated on **Dutch** speech, so for Arabic the
syllable pace is **information only** (confidence low; the method text says so).

### Confidence bands (`speech.confidence`, FINAL_04 §6)

| Band | Mean word probability | Speaking time |
|---|---|---|
| high | ≥ 0.85 | ≥ 30 s |
| medium | ≥ 0.70 | ≥ 10 s |
| low | otherwise | otherwise |

A measure that does not use the transcript (syllable pace) uses the speaking-time part only.

### C1 split (`configs/evaluation.yaml: e2.split`)

C1 (slice 5) is split once, before any tuning, by **script**, so that all clips made from one
script fall in the same part. The split uses a fixed seed:

- `dev_fraction: 0.3` (about 90 of about 300 clips) and `seed: 20261117`, **confirmed by the owner
  on 2026-11-17**.
- **Stratified** (owner, 2026-11-17): both parts must cover the full speaking-rate range of C1
  (110–200 wpm, FINAL_10 §2) and every Kokoro voice used. Scripts are grouped by the speaking-rate
  band and voice of their clips, and each group is split 30 / 70 with the same seed. The rate bands
  and the voice list are fixed in the C1 generation plan (slice 5) before the split runs.
- **C1 as built (ADR 0014 §5):** each script has one voice and is rendered in every rate band, so
  the split is stratified by voice (3 of each voice's 10 scripts to development) and every band
  is covered by construction; the count check below confirms it.
- **Check:** the split script prints the number of clips per part by rate band and by voice, and
  stops with an error if any band or voice is missing from either part.
- VAD values (`speech.vad`) and the syllable values are tuned **only on the development part**.
- E2 pass/fail against ADR 0009's targets is scored **only on the held-out test part**. Changes
  made after seeing test results are reported as such.

## Consequences

- `SpeechReport` is now schema version 2. It adds speech regions, pause events, rate windows,
  `measurements` (every one a `Measurement` with evidence and confidence) and `not_measurable`
  (measure → reason).
- Smoke test (2026-11-17, laptop CPU, whisper_small): a 3-minute public DisfluencySpeech clip
  gave 193 wpm, 219 articulation wpm, 4.7 syllables/s and 34 pauses (11 at sentence boundaries,
  3 at clause boundaries, 20 mid-sentence). The run took 45 s. It shows the pipeline works, not
  that it is accurate. Accuracy is E2's job.
- **Watch in E2 (owner, 2026-11-17):** in the smoke test, 20 of 34 pauses were `mid_sentence`.
  The DisfluencySpeech transcripts may be lightly punctuated, so punctuation gaps could inflate that
  share. E2 therefore reports the share of pauses labelled `mid_sentence` next to the C1 ground
  truth (pauses inserted inside vs. between sentences), per clip and overall, on the test part.
  It also reports how often a pause's position label is wrong.
- Re-check `min_speech_duration_ms` (250 vs 100) and `max_words_per_sentence_mark` on the C1
  development part and on the owner's Arabic B1 clips. Any change needs an update to this ADR.
