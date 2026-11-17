# 0012 — Fillers, discourse markers and repetitions

- Status: **accepted** by the owner on 2026-11-17 (slice 3 plan, with changes 1–8). The acoustic
  filler values are **provisional** until they are tuned on the data named below.
- Date: 2026-11-17
- Decider: project owner
- Plan reference: docs/plan/FINAL_04 §2 (fillers, repetitions), §5 (Arabic fillers, **changed by
  this ADR**); FINAL_06 §3 (T1); FINAL_10 E3; ADR 0011 (confidence bands)
- Config keys affected: `configs/default.yaml: speech.pitch`, `speech.fillers`,
  `speech.repetitions`; `configs/models.yaml: asr.*.verbatim`; `configs/evaluation.yaml: e3`;
  `src/presenceai/lang/{en,ar}/fillers.yaml`

## Context

P1 slice 3 measures fillers and repetitions. FINAL_04 §2 asks for two filler detectors, a
lexicon on the transcript and acoustic candidates, combined so that confidence is high when they
agree. Two facts shape the design. Base Whisper drops many fillers: on a public DisfluencySpeech
clip, whisper-small left out an "uh" that turbo kept. And the verbatim model T1 does not exist
until P4.

## Decision

### 1. Hesitations vs discourse markers

- **Hesitations** are counted as fillers. English: uh, um, uhm, erm, er, ah, hmm, mm. Arabic: امم,
  اه, ممم, آآ.
- **Discourse markers** are reported separately, as information only, and never penalised.
  English: like, you know, I mean, so, basically, actually. Arabic: يعني, طيب. Every occurrence is
  counted, including ordinary uses ("I like it"), so marker counts are labelled **upper bounds**.
- **Change to FINAL_04 §5:** يعني and طيب move from the Arabic filler list to the discourse-marker
  list. A doubled marker ("يعني يعني") is counted as a **repetition**.
- Lists: `lang/en/fillers.yaml` and `lang/ar/fillers.yaml`.

### 2. Normalisation, shared by fillers and repetitions

Implemented in `text/normalize.py`:

- every language: case-fold, and strip punctuation and symbols from both ends of a token;
- Arabic also: remove diacritics and tatweel, unify alef forms (أ إ آ ٱ → ا), and collapse
  stretched letters (طيييب → طيب);
- for comparing a token with the filler lists only, repeated letters in any language are also
  collapsed ("Ummm" = "um"). Repetition matching does not do this, so "too" and "to" stay
  different.

### 3. Lower bounds and confidence caps

`verbatim: false` is set on both base Whisper models in `configs/models.yaml`, and
`verbatim: true` on T1. With a non-verbatim model:

- the filler and repetition counts are **lower bounds**, and the method text says so;
- confidence is capped at **medium for English and low for Arabic**
  (`speech.fillers.confidence_cap_without_verbatim`), on top of the ADR 0011 bands.

Rates are per 100 non-filler words.

### 4. Acoustic candidates (PROVISIONAL values)

A candidate is voiced sound inside a VAD speech region that no recognised word covers, with
steady pitch:

| Key | Value |
|---|---|
| `min_duration_s` | 0.15 |
| `min_voiced_fraction` | 0.7 |
| `max_pitch_sd_semitones` | 1.0 |
| `max_unvoiced_gap_s` | 0.05 (short unvoiced dropouts merged inside one candidate) |
| `agreement_tolerance_s` | 0.05 (a candidate this close to a lexicon hesitation agrees with it) |

The report shows a **range**: the lower end is the lexicon count, and the upper end adds the
acoustic candidates that no lexicon filler explains. The upper end always has low confidence.

**Tuning is allowed only on:**
- DisfluencySpeech **train and validation** splits (Apache-2.0; its test split is used only for
  scoring);
- **AMI development sessions held out from E3.**

AMI is split by **session**, not single meeting, because the four meetings (a–d) of a session
have the same speakers. The split is stratified by recording site and was drawn once on
2026-11-17, before any download, with `random.Random(20261117)`, choosing one session per site
from the sorted disfluency-layer pool:

- **development (tuning):** ES2005, IS1001, TS3005 (12 meetings);
- **test (E3 scoring only):** ES2002, ES2006, ES2008, IS1003, IS1009, TS3006, TS3007.

**PodcastFillers stays evaluation-only** and is never used for tuning or training. Downloads
need the owner's approval first. AMI follows `docs/data/DATA_SOURCES.md` §4: official Edinburgh
files only, and the `LICENCE` file in the v1.6.2 zip is read before any other file is used.

### 5. Word timing can hide fillers (owner's check)

A hesitation directly next to a word is invisible to the acoustic detector if Whisper stretches
that word's timestamps over it. A unit test pins this down. The real public clip shows it:

- **whisper-small** wrote "But but boy" and gave the second "but" 6.98–7.80 s, which covers the
  hesitation. The detector found nothing there; the repetition was still found.
- **turbo** wrote "But, uh, but" and placed "uh" at 7.08–7.46 s. The lexicon found it, and an
  acoustic candidate right after it (7.47–7.68 s) agreed.

E3 measures how often this happens.

### 6. Repetitions (`speech.repetitions`)

- A 1–3-word phrase said again, with the second copy starting within 2.0 s of the first copy's
  end (FINAL_04 §2). The longest phrase wins, and a run ("I I I") is one event with three copies,
  counted as two repetitions.
- Hesitations are skipped: "the uh the" is a repetition, and "uh uh" is two fillers.
- Matching uses the normalisation in §2, Arabic rules included.

### 7. Pitch track (`speech.pitch`)

- pYIN (librosa, ISC; `uv add librosa`; licence check 0 FAIL, 0 WARN) runs **once per
  recording**, with a limited range of 65–400 Hz, a 1024-sample frame (64 ms) and a 160-sample
  hop (10 ms).
- About 5 s per 3 minutes of audio on the laptop, plus about 20 s for numba to compile on the
  first call of a run.
- The track is kept on the report object for slice 4 (pitch variation). It is not written to the
  JSON report.

### 8. Left out until T1 (P4)

**False starts** (cut-off words, "th- the") need a verbatim transcript. They are not measured
until T1 exists (FINAL_04 §2 "T1 transcript marks"). This is stated in `LIMITATIONS.md`.

### 9. E3 additions (`configs/evaluation.yaml: e3`)

- `base_initial_prompt` is an **extra ASR condition, not the default**: Whisper is given a
  filler-rich prompt, fixed in the E3 config before E3 runs.
- E3 reports **precision** next to recall and F1 for every detector and condition, so that
  fillers the prompt makes Whisper invent show up as false positives.

## Consequences

- `SpeechReport` is schema version 3. It adds filler events, acoustic candidates and repetition
  events, the measurements `fillers.*`, `discourse_markers.*` and `repetitions.*`, and the pitch
  track (in memory only).
- Re-check after tuning: the acoustic values in §4 on the development data, then this ADR moves
  them from provisional to tuned. The confidence caps are lifted only when T1 is in use
  (`verbatim: true`).
