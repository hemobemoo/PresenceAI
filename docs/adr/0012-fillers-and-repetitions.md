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

**How the result is reported** (changed by the owner on 2026-11-17, before merge):

- **Base Whisper (`verbatim: false`):** "**at least N per 100 words**". The value is the
  lexicon rate, the unit is `per 100 words (at least)`, the method text starts "at least N per
  100 words", and the confidence is capped (§3). **No range is claimed.** On the real 3-minute
  clip, the range 0.86–1.54 per 100 words did not contain the true rate of about 3.0 (18 uh/um in
  609 reference words), so calling it a range would mislead. Acoustic candidates without a
  lexicon filler are shown separately as `fillers.acoustic_only_candidates`: extra candidates,
  low confidence, not counted as fillers.
- **Verbatim model (T1, `verbatim: true`):** a **range**. The lower end is the lexicon rate
  (`fillers.per_100_words`) and the upper end (`fillers.per_100_words_upper`, low confidence)
  adds the acoustic-only candidates. Whether that range contains the true rate is checked in E3
  before any wording relies on it.

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

The session split is **approved by the owner (2026-11-17)**. Its speaker separation is still to be
checked:

- **Speaker check, to do after the AMI download (which needs the owner's approval):** read the
  speaker (participant) list in the corpus metadata that comes with the annotation package and
  confirm that **no speaker appears in both a development and a test session**. Grouping
  meetings by session keeps each session's speakers together, but it does not prove that no
  person took part in two sessions, so this is checked, not assumed.
  - **Result:** _not yet checked; recorded here after the download._
  - If any speaker is shared, the split is redrawn with the shared speaker's sessions on one side,
    before any tuning, and this ADR records the new split.

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
  recording**, with a limited range of 65–400 Hz (raised to 600 Hz by ADR 0013), a 1024-sample frame (64 ms) and a 160-sample
  hop (10 ms).
- Cost per 3 minutes of audio on the laptop (i7-1255U: 2 performance + 8 efficiency cores),
  single-threaded: 5 s when the machine was idle, 21–23 s with ~37 % background load
  (browsers). The same code and file were used for both; the difference is which core Windows
  runs pYIN on. Numba also compiles once per process, adding ~9–19 s to the first call.
- **Per-stage breakdown** (3-minute public clip, one run, 2026-11-17, ~37 % background load):

  | Stage | Time |
  |---|---|
  | ASR, whisper-small (incl. decode) | 58.0 s |
  | pYIN, first call (incl. compile) | 31.7 s |
  | VAD (Silero) | 1.4 s |
  | imports + model load | 1.1 s |
  | second audio decode | 0.25 s |
  | pauses, rate, syllables, fillers, repetitions, report | ~0.2 s together |

  Apart from ASR and pYIN, every stage together takes about 3 s, so nothing else needed fixing.
  The 45 s → 99 s difference between the slice 2 and slice 3 smoke runs is pYIN plus slower ASR
  under a busier machine. Options if pYIN becomes a problem: run it only over VAD speech regions,
  or run it alongside ASR. Neither is done yet.
- Tests that run real pYIN are marked `slow`, so the fast suite does not pay the compile cost.
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
