# 0013 — Loudness and pitch variation

- Status: **accepted** by the owner on 2026-11-17 (slice 4 plan, all seven proposed values, plus
  changes 1–5). The plateau clipping rule added during implementation (§3) was confirmed by the
  owner on 2026-11-17 **as provisional**, and is reported apart as "possible clipping" with low
  confidence.
- Date: 2026-11-17
- Decider: project owner
- Plan reference: docs/plan/FINAL_04 §1 (noise), §2 (loudness, pitch variation); FINAL_05 §1
  (`vocal_variety` opt-in); FINAL_02 §3.1 (live meter); ADR 0011 (bands); ADR 0012 (pitch track,
  fillers)
- Config keys affected: `configs/default.yaml: speech.loudness`, `speech.pitch_variation`,
  `speech.pitch.fmax_hz` (400 → 600), `speech.pitch.limit_band`; `configs/engine.yaml: opt_in`;
  `src/presenceai/lang/{en,ar}/verdict_terms.yaml`

## Context

P1 slice 4 measures loudness and pitch variation. Both are easy to turn into unfair or
misleading judgements:

- absolute loudness depends on the microphone and distance;
- browser processing flattens loudness;
- absolute pitch differs by voice;
- pitch trackers make octave errors;
- "monotone" is a verdict FINAL_04 §2 forbids unless the user chose that goal.

## Decision

### 1. Loudness is relative only

- **One level function**, `loudness.frame_dbfs`, gives the level of a 20 ms frame. The live meter
  (P6) calls it on each 20 ms chunk. The report's 1 s windows are power means of the same frames,
  so live and post-analysis numbers are identical. A test checks this.
- **Measures**, using only 1 s windows that are at least 50 % VAD speech:
  - `loudness.variation_cv`: coefficient of variation of the window RMS amplitude (FINAL_04);
  - `loudness.spread_db`: p90 − p10 of window levels, information only;
  - `loudness.clipping_share`;
  - `loudness.change_from_baseline_db`: only with `presenceai analyze --baseline earlier.json`
    (one of the user's own reports), until profiles exist in P3. Confidence is at most medium,
    because a change of microphone or distance cannot be detected.
- **An absolute level never becomes a measurement.** It is kept only in `LoudnessProfile`
  (median speech dBFS, noise floor, signal-to-noise ratio, clipped runs), which nothing scores.

### 2. Browser processing (owner, 2026-11-17)

Nothing in the repository turns off the browser's `autoGainControl`, `noiseSuppression` or
`echoCancellation`. Auto-gain flattens loudness variation and makes baseline comparisons
unreliable.

- **Live capture (P6) must request all three as off.** A P6 test must check the constraints sent
  to the browser.
- Every report has a `capture` field with each setting as `on`, `off` or `unknown`. Uploaded
  files are `unknown`; `--capture-processing off` declares a recording made with all three off.
- When any of them is `on` or `unknown`, `loudness.variation_cv`, `loudness.spread_db` and
  `loudness.change_from_baseline_db` are capped at **low** confidence, and the method text says
  why. Clipping is not capped.

### 3. Clipping on the original samples

- Clipping is detected on samples decoded at the file's own rate and channels
  (`audio.load_native`), before resampling to 16 kHz.
- **Rule 1, clipping** (approved): `|x| >= 0.999` in runs of at least 3 samples. Reported as
  `loudness.clipping_share` and counted in `LoudnessProfile.clipped_runs`.
- **Rule 2, possible clipping** (owner confirmed as provisional, 2026-11-17). The owner's MP3
  test showed that LAME lowers clipped peaks to about 0.97, so rule 1 found nothing in a clipped
  MP3. A clipped signal still has a plateau. When a channel's peak is at least 0.9 of full scale,
  and at least 4 times as many samples lie within 2 % of the peak as in the 2–4 % band below it,
  those top-band samples are possible clipping (same 3-sample runs).
  - **Reported apart** as `loudness.possible_clipping_share` (and `possible_clipped_runs`),
    **always low confidence**, never added to rule 1. Samples rule 1 already counts are left out.
  - Config: `clip_peak_min: 0.9`, `clip_pileup_band: 0.02`, `clip_pileup_ratio: 4.0`.
- **Ratios measured on synthetic signals, 2026-11-17** (`loudness.pileup_ratio`):

  | Signal | Ratio |
  | --- | --- |
  | hard-clipped sine, after MP3 | 13–173 |
  | unclipped 0.95 sine (the hardest unclipped case) | 2.3 |
  | loud speech-like signal through a look-ahead limiter, not clipped (owner's safeguard 2), WAV | 0.86–1.20 |
  | the same, after MP3 (128 kbit/s) | 0.01–0.20 |
  | hard-clipped speech-like signal, WAV | 39–50 |
  | the same, after MP3 | 0.08–0.15 (its peaks overshoot to 1.04–1.06, so rule 1 catches it) |

  The limiter cases cover 3 seeds, drive +6 to +18 dB, ceilings 0.891, 0.944 and 0.98 (−1, −0.5
  and −0.18 dBFS), attack 1–5 ms and release 1–80 ms. None came near 4, so mastered podcast or
  video audio should not be flagged. `test_a_limited_but_unclipped_signal_is_not_possible_clipping`
  keeps this. **Re-check on real public clips** once they are downloaded, and record the ratios
  here.

### 4. Noise and confidence

- The noise floor is the 10th percentile of 20 ms non-speech frames. The signal-to-noise ratio
  is the median speech level minus that floor.
- Below 10 dB: loudness and pitch are "not measurable in this recording". From 10 to 20 dB:
  low confidence. With less than 1 s of non-speech the noise is unknown and confidence is at most
  medium.
- Loudness variation needs at least 10 s of speech windows (the ADR 0011 rule).

### 5. Pitch variation: semitones from the speaker's own pitch

- The slice 3 pitch track is **reused**; a test fails if pYIN runs more than once per recording.
- **Frames used:** voiced, inside VAD speech, outside filler stretches (lexicon hesitations and
  acoustic candidates from slice 3, ± 50 ms).
- Each frame is `12·log2(f0 / median f0)`, so the same movement gives the same number for any
  voice. **No absolute pitch in Hz is reported.**
- Measures: `pitch.sd_semitones` and `pitch.range_semitones` (p90 − p10), both **information
  only**.
- **Confidence:**
  - under 5 s of usable voiced speech: "not measurable";
  - under 15 s: low;
  - otherwise the ADR 0011 speaking-time bands;
  - capped by the noise rules in §4.

### 6. Pitch range: fmax 400 → 600 Hz (owner, 2026-11-17)

- A 400 Hz ceiling cut off emphasised peaks of higher voices. Measured with real pYIN on a 340 Hz
  voice moving ±6 semitones (peaks ≈ 480 Hz): 3.02 semitones at fmax 400 against 4.22 for the
  same movement at 120 Hz, about 29 % low. At fmax 600 the two give 4.23 and 4.22.
- **Timing:** on the 3-minute clip, interleaved runs, three each, gave a median of 12.1 s at
  400 Hz and 11.1 s at 600 Hz. That is no slower (within the machine's run-to-run noise, ADR
  0012 §7), so fmax is 600 Hz.
- The share of usable frames within 2 % of fmin or fmax (`limit_band`) is reported as
  `at_limit_share`. Above 5 % (`max_at_limit_share`), confidence is low and the method says
  "at the pitch range limit".

### 7. Octave errors

- A frame more than 9 semitones from the running median (± 0.25 s) is folded by 12 semitones if
  that lands within 3 semitones. Otherwise it is dropped. Above 20 % corrected frames, confidence
  is low.
- **This rule can also remove real large jumps** of more than 9 semitones within half a second
  (for example a sudden emphatic rise). The `octave_folded` and `dropped` counts in
  `PitchDiagnostics` are how we will notice. If real recordings show many dropped frames, the
  limit is revisited here. On the 3-minute public clip, 13 of 12,092 frames were folded and none
  dropped.

### 8. No verdicts; vocal_variety opt-in

- `lang/en/verdict_terms.yaml` lists words such as monotone, flat, dull, boring and robotic.
  `lang/ar/verdict_terms.yaml` lists words such as رتيب, رتابة, ممل, باهت, آلي and على وتيرة
  واحدة.
- **Whole words, before Arabic letter normalisation** (owner, 2026-11-17,
  `coach.guardrails.contains_verdict_term`). Only diacritics and tatweel are removed, and an
  attached ال، ب، و، ف، ل، ك may come first. No suffix is stripped, so the lists name each
  inflected form. The banned-term normalisation had made «آلي» ("robotic") match the common word
  «إلى» ("to"). Now it does not, and neither does «آليات» ("mechanisms"). Because «آلي» also
  means "automatic", our own Arabic text uses «تلقائي» for that.
- A safety test checks every report string, in English and Arabic, and every language-pack
  string.
- `configs/engine.yaml: opt_in: [vocal_variety]`: the engine (P5) may use this skill only when
  the user chose it.

## Consequences

- `SpeechReport` is schema version 4. It adds `capture`, `loudness_profile`,
  `pitch_diagnostics`, `baseline_source`, and the `loudness.*` (including
  `loudness.possible_clipping_share`) and `pitch.*` measurements.
- Loudness variation, spread and baseline change are always low confidence on uploaded files,
  including all public-video evaluation data, because their capture settings are unknown.
- Smoke test (3-minute public clip, capture unknown): signal-to-noise 55 dB; no clipping;
  loudness variation 0.29 (low, capture unknown); pitch spread 3.1 semitones (high).
- Re-check: the plateau ratio on real public clips (mastered and clipped); the octave limit on real
  speech (dropped counts); fmax on real higher voices (`at_limit_share`).
- **B1, partly checked** (2026-11-21, 20 clips, one laptop microphone, SNR 55–64 dB): clipping
  share 0, possible-clipping share 0 and no plateau ratio on every clip, so there were **no
  false alarms**. Detection of **real** clipping is still untested, so the public clipping
  clips (ADR 0016 open item 2) stay open.
