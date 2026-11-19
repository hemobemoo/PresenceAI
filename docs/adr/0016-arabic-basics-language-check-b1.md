# 0016 — Arabic basics: report language, language check and the B1 pilot set

- Status: **accepted** by the owner on 2026-11-18 (decisions a–c as proposed, plus extra
  instructions 1–4). Amended 2026-11-19: pause labels from the owner's transcript, no pause
  F1 or duration on B1 (see "Amendment"). On 2026-11-18 the owner also accepted English unit
  codes with translated labels and report schema 6, and decided open item 3 (B1 storage,
  option A).
  `mismatch_min_probability` is **provisional** until the B1 re-check.
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
   in a YAML file; the detected pauses go in a labels file (`start<TAB>end<TAB>label`).
   *Chosen.* Pause labels come from the owner's transcript punctuation (see "Amendment").

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
  ADR 0015) and `<clip>.labels.txt` (the detected pauses, labelled by `label` from the
  owner's punctuation; see "Amendment"). The owner only corrects. A corrected file is never
  overwritten;
- `score` uses the E2 matching and F1 rules unchanged (`evaluation.e2`, ADR 0014 §6) and writes
  numbers only, never transcript text, to `experiments/results/b1/`. Per clip and per language:
  wpm error, WER (E1), pause F1 at 0.5 and 1.0 s and duration error (only with hand-checked
  pause edges, so not on B1 as recorded; see "Amendment"), the position table and both
  position errors, words per sentence mark, removed words and how many were real, clipping
  share, possible clipping share, the plateau ratio, SNR, and the detected language and its
  probability;
- B1 is **descriptive only**: no pass/fail, and no threshold is tuned on it. Arabic E2 is
  reported for one speaker.

## Amendment (owner, 2026-11-19): pause labels from the owner's transcript

The owner uses Windows Sound Recorder only and will not install Audacity, so pause edges are
not moved by ear. The annotation workflow changes:

- `prefill` writes every pause label as `?`. The pause times stay exactly as the app detected
  them: `label` never changes them, adds pauses or deletes them.
- `run.py label <clip>` (or `--all`) fills `<clip>.labels.txt` from the owner's corrected
  `transcript`:
  - The owner's words are aligned to Whisper's timed words (the report cached at pre-fill, in
    `data/b1/reports/`) by a word-level edit-distance alignment on normalised tokens. A
    misheard word still pairs up by position. Among equally short alignments, the one with
    the most exact matches wins.
  - Each pause takes the punctuation after the last of the owner's words before it, by word
    midpoint, as in `pauses.find_pauses`: `. ? ! ؟` → `sentence`; `، , ؛ ;` → `clause`;
    none → `mid`.
  - **Words that overlap a pause** (owner, 2026-11-19, second change). Whisper often stretches
    a word into the silence next to it, so its midpoint falls inside the VAD pause. On the
    first two Arabic clips this made every `?` (46 of 56 and 48 of 69 pauses). Such a word now
    goes to the side where more of it lies **outside** the pause, if that part is longer by
    more than `b1.labels.overlap_margin_s` (`configs/evaluation.yaml`, **0.10 s, provisional**).
    A word about equally on both sides, or (almost) all inside the pause, still gives `?`.
    Words whose midpoint is outside the pause keep the midpoint rule. The first clips gave these
    `?` counts:

    | Margin (s) | ar_01 (56 pauses) | ar_02 (69) | en_05 (87) |
    |---|---|---|---|
    | old rule (midpoint only) | 46 | 48 | 71 |
    | 0.20 | 20 | 21 | 54 |
    | **0.10** | **10** | **10** | **27** |
    | 0.05 | 7 | 6 | 19 |

    The margin was not chosen to reach a count. A remaining `?` is decided by ear with
    `listen`.
  - Words the owner added (e.g. «امم») have no timing and sit between their aligned
    neighbours. When such a word, or a word Whisper put inside the pause, sits at the pause,
    or when none of the owner's words is timed on one side, the label is `?`. `label` prints
    the pause number, its time and 3 words on each side.
  - A labels file that `label` filled completely is kept unless `--force` is given. A file
    from `prefill`, or one that still has a `?`, is relabelled.
- `run.py listen <clip> <n>` plays 3 s before to 1 s after pause *n*, so a `?` can be decided
  by ear. It uses the standard library only: `wave` and `winsound`, with a temporary WAV slice
  that is deleted afterwards. A file `wave` cannot read (e.g. FLAC) is decoded by the app's
  own PyAV loader. There is no new dependency.
- The annotation records `labels_method: transcript_punctuation` and
  `pause_edges_checked: false`.
- `score` reports pause **positions** (ADR 0014 headline) from these labels. It does **not**
  report pause F1 or pause duration error for clips with `pause_edges_checked: false`; it
  lists them under `pause_edges_not_hand_checked`. Rate, WER, ADR 0015 removed words,
  clipping and the language check are unchanged.
- Consequence: **B1 does not measure pause detection (F1) or pause duration**, because the
  pause edges are the app's own. Arabic pause timing therefore has **no real-speech check**:
  no public Arabic source has timing ground truth (`DATA_SOURCES.md` §3). This is in
  `docs/cards/LIMITATIONS.md`. Position ground truth is the owner's punctuation, which follows
  meaning rather than where the owner paused.

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
3. **Encryption of B1 at rest: decided** (owner, 2026-11-18, option A).
   - B1 holds only the owner's own voice, recorded outside the app with Windows Sound Recorder.
   - The recordings stay as **plain WAV files** in the git-ignored `data/b1/` on the owner's
     laptop only. They are never committed, never synced to OneDrive, and never uploaded to
     Kaggle, Colab, Hugging Face or any other cloud.
   - The pre-filled and corrected annotations (`data/b1/annotations/`) and the cached reports
     (`data/b1/reports/`) contain the transcript, so the same rule applies to them.
   - Encryption at write time (Fernet, `MEDIA_KEY`, FINAL_11 §2, AGENTS.md rule 5) still
     applies to **all media the app itself records**. This exception covers only B1 files made
     outside the app.
   - The owner may turn on Windows device encryption if the Windows edition offers it. This is
     optional and not part of the decision.
4. A native-speaker review of `lang/ar/report.yaml` (with `lang/ar/ui.yaml`), before any user
   sees Arabic reports.
5. The slow real-model Arabic test runs once the first B1 Arabic clip exists.
6. **The app's own pause rule** (`pauses.find_pauses`) uses word midpoints only. When Whisper
   stretches a word into a pause, the app skips that word and takes the word before it as
   `word_before`, so a reported position can come from the wrong word. **Checked, no change**
   (owner's request, 2026-11-19; **B1-informed**: the question came from B1 round 1, the check
   ran on C1 dev only, never on B1 or the C1 test part):
   - B1 round 1 (`experiments/results/b1/round1.md`): 20 of 210 positions disagree with the
     owner's labels. The midpoint rule skipped a stretched word in 3 of them, and the B1
     overlap rule would fix 1.
   - C1 dev (`experiments/e2_rate_pauses/word_side_dev.py`,
     `experiments/results/e2/f1_dev_word_side.csv`, 821 matched pauses):

     | Rule | Agree | mid→boundary | boundary→mid | word_before changed |
     |---|---|---|---|---|
     | **midpoint (app, kept)** | **612** | **0.554** | **0.049** | 0 |
     | overlap 0.05, unplaced skipped | 603 | 0.562 | 0.061 | 96 |
     | overlap 0.10, unplaced skipped | 606 | 0.562 | 0.057 | 68 |
     | overlap 0.20, unplaced skipped | 610 | 0.562 | 0.051 | 18 |
     | overlap 0.10, unplaced → `unknown` | 352 (375 unknown) | 0.576 | 0.052 | 68 |

   - No variant is better than the midpoint rule on either position error, so
     `pauses.find_pauses` stays as it is. B1 `label` keeps its own overlap rule, because it
     places the owner's words, not the app's. The B1 re-score after this check is identical
     to round 1, since the app did not change.
7. **Language check, B1 round 1:** 1 false notice in 4 clips (an English clip detected as
   Arabic, probability 0.83; another English clip reached only 0.53 for English). No
   `mismatch_min_probability` value separates these. It stays at 0.5 until all 20 clips are
   scored, and then the owner decides.
