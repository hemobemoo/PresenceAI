# 0016 — Arabic basics: report language, language check and the B1 pilot set

- Status: **accepted** by the owner on 2026-11-18 (decisions a–c as proposed, plus extra
  instructions 1–4). `mismatch_min_probability` is **provisional** until the B1 re-check.
- Date: 2026-11-18
- Decider: project owner
- Plan reference: FINAL_12 §2 (P1: "Arabic language pack basics", study "Record pilot set B1");
  FINAL_04 §5 (language detection on the first 30 s, notice not error); FINAL_07 §B; FINAL_10 §2
  (E1, E2); AGENTS.md ("English and Arabic UI strings live in `src/presenceai/lang/`")
- Config keys affected: `configs/default.yaml: speech.language_check.window_s`,
  `speech.language_check.mismatch_min_probability` (both new)

## Context

After slice 5, three P1 pieces were missing:

1. About 40 method texts and "not measurable" reasons were written in English inside
   `speech/*.py`. That breaks the AGENTS.md rule that user-facing text lives in `lang/`, and there
   was no Arabic version.
2. FINAL_04 §5 asks for language detection on the first 30 s, with a notice on a mismatch. The
   chosen language went straight to Whisper and was never checked.
3. Several decisions taken on synthetic speech (ADRs 0011, 0013, 0014, 0015) wait for a check on
   real speech, and Arabic E2 has no public source with timing ground truth
   (`docs/data/DATA_SOURCES.md` §3). Pilot set B1, the owner's own recordings, had no size, plan
   or annotation format. FINAL_07 still said 30–40 clips.

## Options considered

**a. Report language.**
1. The recording's language (the language the user chose for the analysis). One setting; a
   report never mixes languages; the Arabic wording is checked against the Arabic verdict and
   banned lists. *Chosen.*
2. A separate UI language. More flexible, but it needs a UI choice that only exists in P3.

**b. When a detected language counts as a mismatch.**
1. Any other top language. Short or code-switched clips would give false notices.
2. Another language with probability ≥ 0.5. Whisper's own detection threshold is 0.5 as well
   (faster-whisper 1.2.1 `language_detection_threshold`). *Chosen, provisional.*
3. A higher value such as 0.8. Fewer false notices, more missed mismatches. There is no data yet
   to prefer it.

**c. B1 size.**
1. 30–40 clips (FINAL_07). About twice the recording and correcting work for one speaker.
2. 10 EN + 10 AR clips of 2–3 minutes. It matches E1's 20 hand-corrected clips (FINAL_10) and
   covers each check below at least once per language. *Chosen.*

**d. Annotation format.**
1. Annotate from scratch (word times, pauses). Many hours per clip.
2. Pre-fill everything from the app and only correct (owner's instruction). The transcript goes
   in a YAML file; pauses go in an Audacity label track, so they can be moved on the waveform.
   *Chosen.*

## Decision

**Report texts.** Every method text, "not measurable" reason, note, unit label and the
language notice is in `src/presenceai/lang/{en,ar}/report.yaml`, with named placeholders.
`lang.text(lang, key, **values)` fills them in and fails on a missing or extra value. Speech
code uses keys only. The report is written in the recording's language. The English texts are
unchanged word for word. `Measurement.unit` stays a stable code (e.g. `words/min`);
`lang.unit_label` gives its display label. Tests: both files have the same keys and the same
placeholders; no verdict or banned word in either language; no sentence-like literal in the
speech modules (`tests/unit/test_report_texts.py`). The Arabic file is a draft translation for
a native-speaker review, as `lang/ar/ui.yaml` is.

**Language check** (`speech/language_check.py`, `SpeechReport.language_check`, schema 6):

```yaml
speech:
  language_check:
    window_s: 30.0
    mismatch_min_probability: 0.5   # PROVISIONAL; re-check on B1
```

- Whisper's `detect_language` runs on the first 30 s of speech, counted from the first VAD
  onset so that leading silence does not dilute it. No VAD filter inside Whisper.
- A mismatch is: detected language ≠ chosen language **and** probability ≥ 0.5. The report then
  carries a notice in the report language, with the detected language and its probability.
- It is a notice only. The analysis still runs in the chosen language, nothing stops, and no
  measurement or score changes (`tests/unit/test_language_check.py`). With no speech there is no
  check (`language_check: null`).

**Arabic smoke test** (`tests/integration/test_arabic_smoke.py`, no models): a fake Arabic
recording runs through the whole `analyze` path. Fillers are capped at low confidence, markers
are counted apart, «يعني يعني» is a repetition, pause positions are `unknown` with sparse
punctuation, syllable pace is information only, and every report text is Arabic with no
verdict or banned word. A slow real-model test waits for the first B1 Arabic clip.

**B1: 20 clips, 10 English and 10 Arabic, 2–3 minutes each**, recorded and corrected by the
owner only. The clip plan, setup, file names and correcting steps are in
`docs/data/B1_RECORDING_GUIDE.md`:

- clips 1–4 normal; 5–6 project defense; 7 slow, with deliberate mid-sentence pauses; 8 fast;
  9 fillers and one repetition; 10 mild steady noise. Every clip ends with a short, slightly
  quieter closing ("Thank you." / «شكراً») for the ADR 0015 check;
- Arabic adds natural «يعني» markers, one doubled «يعني يعني» in clip 9, English terms
  mid-sentence in clips 2, 5 and 6, and long sentences in clips 3, 4 and 6;
- laptop microphone, quiet room, 30–50 cm, audio only, WAV;
- the files stay on the owner's laptop in `data/b1/` (git-ignored): never GitHub, Drive, Kaggle
  or Colab.

**B1 annotation and scoring** (`schemas/b1.py`, `evaluation/b1.py`,
`experiments/b1_real_speech/run.py`):

- `prefill` writes `<clip>.yaml` (Whisper transcript, VAD speech start and end, words removed by
  ADR 0015) and `<clip>.labels.txt` (detected pauses with their position labels, `?` where the
  position is unknown). The owner only corrects. A corrected file is never overwritten;
- `score` uses the E2 matching and F1 rules unchanged (`evaluation.e2`, ADR 0014 §6) and writes
  numbers only, never transcript text, to `experiments/results/b1/`. Per clip and per language:
  wpm error, WER (E1), pause F1 at 0.5 and 1.0 s, duration error, the position table and both
  position errors, words per sentence mark, removed words and how many were real, clipping
  share, possible clipping share, the plateau ratio, SNR, and the detected language and its
  probability;
- B1 is **descriptive only**: no pass/fail, and no threshold is tuned on it. Arabic E2 is
  reported for one speaker.

## Consequences

- FINAL_07 §B now says B1 is 20 clips (10 EN + 10 AR). E1's "20 hand-corrected clips of the
  owner (B1)" in FINAL_10 is unchanged and now matches.
- Reports are schema 6. A baseline report from schema 5 must be made again.
- **Re-check on B1:** `mismatch_min_probability`. All 20 clips are in the chosen language, so
  every notice on B1 is a false notice. If there are any, especially on the code-switching
  clips, propose a new value here; the owner decides.
- What would show a decision is wrong: a false language notice on B1; Arabic report text that a
  native speaker reads as a judgement; correcting taking far longer than 20 minutes per clip.

### Open items

1. **TED-LIUM 3** (approved later, test set only): only when E1 or E2 on real speech is
   scheduled. Downloaded and run on Kaggle or Colab, **never on the owner's laptop**. Not part
   of this slice.
2. **Public clips for the ADR 0013 clipping check** (approved later): a small set of about 20
   short clips from a source already listed in `docs/data/DATA_SOURCES.md`. Until then, the
   clipping check (`loudness.clipping_share`, `possible_clipping_share` and the plateau ratio)
   runs on the B1 clips as the first real-speech check, **one microphone only**, and is reported
   as such.
3. **Encryption of B1 at rest.** AGENTS.md rule 5 encrypts the media the app writes. The B1
   recordings are written by Windows Sound Recorder, outside the app, and the app has no media
   encryption module yet (`storage/` is empty). Until the owner decides, they are plain WAV files
   in the git-ignored `data/b1/` on the laptop only. The pre-filled annotations contain the
   transcript and stay in the same folder.
4. A native-speaker review of `lang/ar/report.yaml` (with `lang/ar/ui.yaml`), before any user
   sees Arabic reports.
5. The slow real-model Arabic test runs once the first B1 Arabic clip exists.
