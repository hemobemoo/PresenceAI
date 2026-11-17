# 0014 — C1 synthetic speech and the E2 protocol

- Status: **accepted** by the owner on 2026-11-17 (slice 5 plan with additions 5–9). Fixed
  **before** C1 is generated and before E2 runs; nothing here is changed after seeing E2 results.
- Date: 2026-11-17
- Decider: project owner
- Plan reference: docs/plan/FINAL_07 §C (C1); FINAL_10 §1 (E2); ADR 0009 (E2 targets); ADR 0011
  (pauses, rate, C1 split); ADR 0012 (E3 data); ADR 0013 (noise)
- Config keys affected: `configs/evaluation.yaml: e2` (match, exclusion, noise, wpm_judged_on),
  `c1` (new)

## Context

E2 checks speech rate, pauses and pause durations against exact ground truth. FINAL_07 plans C1
as ~300 Kokoro clips "from scripts with known word counts, inserted silences, inserted filler
tokens". This ADR fixes how C1 is built, how truth is measured, and how E2 is scored.

## Decision

### 1. Kokoro runs only on Colab, in its own environment

- Kokoro-82M's weights and voices are Apache-2.0. The `kokoro` package (Apache-2.0) depends on
  `misaki[en]`, which pulls in **`phonemizer-fork` (GPL-3.0+)** and **`espeakng-loader`** (MIT,
  bundling espeak-ng, GPL-3.0). `kokoro/pipeline.py` imports `misaki.espeak`, and with it both
  packages, when it is imported, even if espeak-ng is never called.
- So `kokoro` **never enters the app's `pyproject.toml` or `uv.lock`** (AGENTS.md rule 6). The
  generator `experiments/c1_kokoro/colab/generate_c1.py` runs on Colab with its own pip installs
  (`configs/evaluation.yaml: c1.generator.pip`). Only the audio and the truth files come back.
- **espeak-ng is never called** (owner, 2026-11-17):
  1. before synthesis, every script word is looked up in misaki's own word lists (`us_gold`,
     `us_silver`, and `gb_*` for the British voices); a word missing from them stops the run;
  2. misaki's espeak fallback is replaced by a function that raises, so a call stops the run;
  3. every word token must get phonemes; misaki 0.9.4 puts no rating on word tokens, so the
     check records whether each word came from misaki's gold list, silver list or its rules
     (e.g. -s, -ing). The result is written to `resources/c1/lexicon_check.json`, and a fast
     CI test fails if any script word is missing from it.
- Downloads (approved 2026-11-17), at revision `f3ff3571…`, each checked against the SHA-256 in
  the Hub's LFS metadata: `kokoro-v1_0.pth` and six voice files. `config.json` is not an LFS
  file and is checked by its git blob id. spaCy's `en_core_web_sm` 3.8.0 (MIT) comes from
  spaCy's own release page, which publishes no digest: its SHA-256 is recorded on the first
  download and pinned in `c1.generator.spacy_model.sha256` for later runs (recorded in Colab run
  t1: `1932429d…`).

### 2. Scripts

- 60 short presentation-style scripts, 105–125 words, written for PresenceAI in this slice
  (`resources/c1/scripts.yaml`). No third-party text, released with the code
  (`docs/data/DATA_SOURCES.md`). The owner reviews a sample of 10; automated checks cover all:
  letters and `, . ? ! |` only (no digits, apostrophes or hyphens, so the word count is
  unambiguous), no listed heteronyms, at least one sentence mark per 20 words (so ADR 0011 never
  gives `unknown`), word count in range, 10 scripts per voice, at least two `|` marks, each inside a sentence
  between two words, and no banned or verdict words.
- `|` marks a mid-sentence split point. Phrases end at `|`, at commas and at sentence marks.

### 3. Construction and ground truth (owner's points 1, 2, 7)

- **Voices:** `af_heart`, `af_bella`, `am_michael`, `am_fenrir` (American), `bf_emma`,
  `bm_george` (British). Each script has one voice; each is rendered 5 times at target rates
  119, 137, 155, 173 and 191 wpm, with new gaps drawn each time: 300 clips.
- **Phrases are generated separately** and joined with silences of exact length. Each boundary
  gets a pause (log-uniform 0.30–3.00 s) with probability 0.6 after a sentence, 0.4 after a
  comma and 0.5 at a `|` mark; otherwise a short gap (0.05–0.20 s), which is not a pause and
  tests false alarms. The true position is the boundary type.
- **Trimming threshold (owner's point 7):** each phrase is trimmed to the first and last 5 ms
  frame within **55 dB of the phrase's loudest frame**, with no padding. Quiet onsets and
  endings (s, f, h) lie 30–45 dB below a vowel, so they are kept; a test builds phrases with
  such edges and checks that no more than one frame is cut.
- **Kokoro's own pauses:** silent stretches of at least 0.10 s inside a phrase are reported
  (count and durations). Those of 0.25 s or more are added to the truth as `mid_sentence`
  pauses with `source: kokoro`.
- **Noise floor:** after joining, white noise at −60 dBFS covers the whole clip, so pauses are
  not digital silence.
- **True speaking rate, measured on the finished clean audio** (owner's point 2): words ÷ (last
  speech offset − first speech onset), the definition of `rate.wpm` (ADR 0011). Kokoro's speed
  setting is not used as truth. A render more than 9 wpm from its target is re-rendered with a
  corrected speed, at most twice; then it is kept if within 110–200 wpm, otherwise dropped and
  counted. Rate bands (110–140, 140–170, 170–200) use the measured value.
- **Clip length** 35–75 s.
- **Limitation:** a pause at a `|` mark ends a separately generated phrase, so it may sound like
  a phrase end, not a hesitation, and Whisper may put a full stop there. E2's position report
  shows this. Stated in `docs/cards/LIMITATIONS.md`.

### 4. Fillers are counted, but never used for E3 (owner's point 3)

- About a third of the clips get one or two fillers (`um`, `uh`), each generated as its own
  phrase after a boundary's gap and followed by a short gap. They count as words in the true
  wpm, because `rate.wpm` counts every transcript word.
- **C1 fillers are not used for E3.** E3 stays on DisfluencySpeech, AMI and PodcastFillers
  (ADR 0012): synthetic fillers spoken alone by a TTS voice are not real hesitations.

### 5. Split

- By script, seed 20261117, 30 % development (ADR 0011). Each voice's 10 scripts are split 3 / 7,
  so 18 scripts (90 clips) are development and 42 (210 clips) test. Every script is rendered
  across all rate bands, so stratifying by voice also covers every band. This replaces ADR
  0011's "grouped by rate band and voice" for C1: with one voice per script and every script in
  every band, the band grouping adds nothing.
- The split script prints clips per part by band and by voice and stops on any empty cell.

### 6. E2 scoring (owner's points 4, 5, 6)

- **Targets are ADR 0009's, unchanged, and are not changed after seeing results.** A miss is
  recorded as a fail. Pass/fail uses the test part only; tuning (ADR 0011's VAD and
  `max_words_per_sentence_mark` values) uses only the development part. The test part is scored
  once per config version, and every run writes its git commit.
- **Pipeline:** the app's own functions (ASR, VAD, pauses, rate), with the transcript cached per
  clip so a VAD change re-runs only VAD and pauses. pYIN is not needed for E2.
- **Matching:** a detected pause matches a true pause if they overlap and their onsets differ by
  at most **0.20 s**; one-to-one, largest overlap first. Only true pauses of at least 0.25 s.
- **F1 at 0.5 s and 1.0 s:** a true pause within **±0.10 s** of the threshold counts neither as
  a hit nor as a miss, and a detected pause matched to it is not a false alarm. E2 **reports how
  many true pauses fall in that zone**, and **F1 without the exclusion** next to it (owner's
  point 5). The target is judged on the F1 with the exclusion, as approved.
- **Duration error:** mean absolute error over all matched pauses.
- **wpm error: judged on all clips** (owner's point 6). The filler / no-filler breakdown is
  extra information. A clip without a measured rate counts as a 100 % error.
- **Pause position, headline measure** (owner, 2026-11-17, fixed on the development part
  before the test run): boundary vs mid-sentence errors, reported separately in both
  directions, as shares of the matched true pauses of each kind:
  - true mid-sentence pauses labelled as a boundary (`mid_as_boundary_rate`);
  - true boundary pauses (sentence or clause) labelled mid-sentence (`boundary_as_mid_rate`).

  Sentence ↔ clause swaps are not errors here, because neither label counts against the
  speaker (ADR 0011). `unknown` labels are left out of both rates and reported apart. The full
  true × detected table is reported, and the overall share of any mismatch (`position_error_rate`)
  is kept as extra information.
- Also reported: per-clip values; the `mid_sentence` share against the truth (ADR 0011);
  Kokoro's own pauses; the live syllable rate against post-analysis wpm and, descriptively,
  against the true articulation rate (pauses removed).

- **Development findings before the test run** (owner, 2026-11-17):
  `experiments/results/e2/f1_dev_findings.md`. The three development clips above 5 % wpm error
  are speech-recognition errors (an invented closing "thank you…", dropped function words at a
  fast rate, merged words), not a counting bug, so the code is unchanged. VAD minimum speech
  stays 250 ms.

### 7. Noise condition (descriptive, no target)

A second copy of the test part with pink noise at **20 dB and 10 dB** signal-to-noise ratio:
speech power over the true speech spans divided by noise power. Slice 4's measured ratio is
reported next to it. Seed 20261117 per clip.

### 8. Where things run

- Generation: Colab T4. Colab's kernel is Python 3.13 and kokoro 0.9.4 needs < 3.13, so the
  generator runs in its own venv on the image's Ubuntu Python 3.12 (nothing extra downloaded).
  The first run checks all 60 scripts against the word lists and times
  5 clips; the rest is generated only after the owner has seen that timing.
- E2 and the noise runs need no GPU and no laptop-specific value; only E8 timing does. They can
  run on a Colab or Kaggle CPU with the app's pipeline (the whisper-small CTranslate2 int8 model
  converted as in ADR 0008, and the C1 clips). The run records the CPU model and package
  versions, because int8 kernels can differ slightly between CPUs.

## Consequences

- New: `schemas/c1.py`, `synth/` (construction, scripts, split), `evaluation/e2.py`,
  `experiments/c1_kokoro/`, `resources/c1/`.
- C1 audio (~350 MB FLAC) stays in git-ignored `data/c1/`; the truth files, manifest
  (SHA-256 of every clip) and split are committed.
- Re-check after the first runs: the trimming threshold against Kokoro's real silence level
  (reported by the timing run), and how often Kokoro adds pauses of its own.
