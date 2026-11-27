# 0028 — E3 protocol: filler and repetition scoring, decoding, data access (PR-3)

- Status: **proposed** (owner's answers E1–E8 and additions 1–6 of 2026-11-27 written in; waits
  for the owner's acceptance). Values marked PROVISIONAL may change only with a new decision
  recorded here.
- Date: 2026-11-27
- Decider: project owner
- Plan reference: FINAL_10 E3; ADR 0012 (fillers, AMI split, `initial_prompt`); ADR 0026 §4 (T1
  pass rule); ADR 0027 (T1 text style, segments, pins, builder pass line)
- Config keys affected: `configs/evaluation.yaml: e3` (model, decoding, initial_prompt, detector,
  wer_normaliser, ami, podcastfillers, repetitions, bootstrap, disfluencyspeech_text_check) and
  `public_data.podcastfillers` (new)

## Context

ADR 0026 §4 defines T1's pass rule against two baselines, base and base + `initial_prompt`. PR-3
measures those baselines once, with a protocol fixed **before** the run so that T1 runs later are
scored the same way. All of it runs on Colab (ADR 0026 §1).

## Decision

### 1. ASR (owner addition 1, E6)

- Model: **whisper-small**, the app's own. On Colab, `openai/whisper-small` is fetched at the
  revision in `configs/models.yaml`, every source file is checked against `models/MANIFEST.json`,
  it is converted with the converter versions in `configs/models.yaml` (`ct2-transformers-converter`,
  int8, the same `copy_files`), and **every converted file must match the app's SHA-256** (the
  converter's two JSON files after line-ending normalisation, §7). Otherwise the run stops (`model_differs_from_app`). Turbo is report-only and is not in this run.
- Decoding: **the app's settings, unchanged** (`configs/default.yaml: asr`): compute type int8,
  beam size 5, temperature 0.0, `condition_on_previous_text` false; plus language "en", word
  timestamps on, no VAD filter (as `speech/asr.py`). These are `e3.decoding`. The same settings
  are used for base, base + `initial_prompt`, and **every later T1 run**. A test checks that
  `e3.decoding` equals `default.yaml: asr`.
- Runtime: faster-whisper 1.2.1 and CTranslate2 4.9.0 (the app's `uv.lock` versions) on a Colab
  T4. The laptop runs int8 on CPU; Colab runs the same int8 model on the GPU. The small numeric
  difference this can cause is the same for base and T1.

### 2. Conditions

- **base**: no prompt.
- **base + `initial_prompt`** (owner E4): `"Umm, let me think, uh, like, hmm... Okay, so, um,
  here's what I was, uh, thinking."` (`e3.initial_prompt`, fixed before the run).

### 3. Detector (owner E5, addition 6)

- **Lexicon only** in PR-3. Fillers follow the T1/E3 rule (`t1text.is_filler`: the app's
  `filler_key` and hesitation list, backchannels zero, one-letter partials not fillers) on
  **both** reference and hypothesis.
- **T1's pass rule (ADR 0026 §4) is scored with the lexicon detector: lexicon on base vs lexicon
  on T1** (and vs lexicon on base + `initial_prompt`).
- Acoustic and fusion detectors wait until their provisional values are tuned on the dev data
  (ADR 0012 §4). CrisperWhisper (upper bound) also comes later.
- Follow-up when T1 is wired into the app: the app's lexicon (`speech/fillers.py`) needs the same
  one-letter-partial rule, because a verbatim model writes "m-" (ADR 0027 §5).

### 4. Sets and scoring

| Set | Unit | Reference | Matching | Deciding |
|---|---|---|---|---|
| DisfluencySpeech test (250 clips) | clip | T1 text builder on `transcript_annotated` (ADR 0027 §1) | word alignment | no (one speaker) |
| AMI test sessions (7 × 4 meetings) | per-speaker T1 segments, **20 % crosstalk filter** (owner E3) | AMI words, T1 spelling | word alignment | **yes** |
| AMI test, crosstalk kept | the same segments without the crosstalk filter | as above | word alignment | no (secondary, owner addition 3) |
| PodcastFillers test episodes | whole episode | annotated **Uh/Um** events (`fullvoc_label`) | **time**: an "uh"-family hypothesis word (uh, um, uhm, er, erm) matches an event widened by **±0.2 s** (PROVISIONAL), one-to-one | **yes** |
| LibriSpeech test-clean | utterance | its transcripts | WER | WER condition only |

- **Word alignment**: unit-cost edit distance over lower-cased words without edge punctuation (the
  hyphen of a partial is kept). An aligned pair of fillers is a TP; a hypothesis filler aligned
  to anything else or inserted is an FP; a reference filler aligned to anything else or deleted is
  an FN. The filler type need not match.
- **PodcastFillers precision is a lower bound**: its annotation may not mark every filler, and
  hmm, mm and ah are not scored there (they are not labelled as fillers).
- The AMI report gives the **share of test speech excluded by the crosstalk filter**.
- **Repetitions (owner E7):** reference repetitions come from the AMI disfluency layer (elements
  whose type contains "repet"), on the deciding AMI segments. A reference repetition is **kept** if
  every one of its words is aligned to the same word in the hypothesis. ADR 0026 §4 condition 4 is
  "repetitions kept on AMI test ≥ base". This is a transcript-faithfulness measure (does the
  model keep both copies?), not the app's repetition detector. The layer's format is **probed
  first** (`probes/ami_disfluency_probe.json`), and the parser is fixed before any repetition
  number is used if the probe disagrees.
- **WER normaliser (owner addition 2):** lower case, punctuation removed (apostrophes kept),
  **fillers not removed** (an invented filler on read speech is a real error). The same for base
  and T1.
- **Bootstrap:** 95 % percentile intervals, 1,000 resamples, seed 20261128, resampling whole units:
  clip (DisfluencySpeech), meeting (AMI), episode (PodcastFillers), speaker (LibriSpeech).

### 5. Data access

- **DisfluencySpeech builder check first** (ADR 0027 §6): stage 0 reports the four counts per
  split; pass = at most 50 of 5,000 rows `words_differ`; otherwise the run stops.
- **AMI**: the pinned annotation zip; the test headsets (7 sessions × 4 meetings × 4 headsets) are
  hashed, and the hashes are added to `configs/pins/` after the run.
- **PodcastFillers (owner E1, E8, addition 4)**: evaluation only. `PodcastFillers.csv` is checked
  against Zenodo's md5. The split zip (4 parts, 25.2 GB) is read by **HTTP range requests**: the
  central directory comes from the last part, then **only the licence files and the test-episode
  WAVs** are read. **Licence files first**: `LICENSE_ANNOTATIONS_ADOBE_RESEARCH_LICENSE.txt` and
  `LICENSE_AUDIO.csv`, recorded before any audio is used. Each extracted WAV is checked against
  its **CRC32** from the zip index, and its **SHA-256 is pinned** after the run, because no
  whole-zip hash exists this way. If Zenodo refuses ranges, all four parts are downloaded and
  checked against Zenodo's md5s.
- **LibriSpeech test-clean**: the pinned archive (ADR 0027 §6).
- Hypotheses and item lists (corpus text and research-only annotation times) stay in the
  git-ignored `data/e3/<run>/`; metrics, probes, pins, checks and licence texts are committed.

### 6. Run `20261128-b1` (Colab T4, 2026-11-28): stopped in stage 1

Record: `experiments/results/e3/base_20261128-b1/`. 42 s, status `error`.

- **Stage 0 passed**: DisfluencySpeech builder check, 20 of 5,000 rows `words_differ` (line 50).
  Counts per split in ADR 0027 (Consequences).
- **Stage 1 failed before any model file was built:** `/usr/bin/python3.12 -m venv` exited 1.
  Colab's Python 3.12 has no `ensurepip`, so it cannot make a venv. **Fix:** the converter
  environment is made with **uv**, pinned to the laptop's uv version (0.13.7 when built), as
  `scripts/download_models.py` does locally: `uv venv --python 3.12`, then `uv pip install` of the
  converter versions in `configs/models.yaml`. uv needs no `ensurepip` and fetches a CPython 3.12
  if needed. The model-identity check (every converted file = the app's SHA-256) is unchanged.
- The next run is a new build (`20261128-b2`, new Drive folder); stage 0 runs again.

### 7. Run `20261128-b2` (Colab T4, 2026-11-28): the model check, and line endings

Record: `experiments/results/e3/base_20261128-b2/` (`model_check.json`). Status
`model_differs_from_app`; the run stopped before any ASR, as designed.

- Stage 0 passed again (20 of 5,000 `words_differ`, identical to b1).
- The uv environment worked: CPython 3.12.3, ctranslate2 4.9.0, transformers 4.57.6, torch
  2.15.0+cpu, as pinned.
- **Every source file matched `models/MANIFEST.json`.** Of the converted files, **`model.bin`
  (the weights), `tokenizer.json` and `preprocessor_config.json` matched byte for byte**. Two did
  not: `config.json` and `vocabulary.json`, the JSON files the converter writes itself.
- **Cause: line endings only.** On the laptop, the same files with `\r\n` turned into `\n`
  (239 and 51,866 lines) have exactly the SHA-256 that Colab produced. The laptop is Windows, so the
  converter's text-mode writes gave CRLF there and LF on Colab. The model is the same.
- **Change to §1:** those two files are compared **after normalising `\r\n` to `\n`**, against
  pins taken from the laptop's files (`e3.model_text_files_lf_sha256`: `config.json b55496ac…`,
  `vocabulary.json 5ad28279…`). **Every other converted file is still compared byte for byte**
  with `models/MANIFEST.json`. A test fails if the rule covers any other file, and (where the app
  model is present) if the pins differ from the laptop's files normalised.
- The next run is a new build (`20261128-b3`).

### 8. Run `20261128-b3` (Colab T4, 2026-11-28): model, AMI and licences done; two formats

Record: `experiments/results/e3/base_20261128-b3/`. 1,443 s, status `error` in the data stage.

- **Model identical to the app** (the §7 rule): `model.bin`, `tokenizer.json`,
  `preprocessor_config.json` byte for byte; `config.json`, `vocabulary.json` after line-ending
  normalisation.
- **AMI test headsets:** all 112 downloaded and hashed (7.27 GB). The server answered 503 17 times,
  and the retries handled every one. Pinned in `configs/pins/ami_test_headsets.json`;
  `public_data.ami_headset.sha256_files` now lists the training and test pin files (they must
  agree where they overlap; they do not overlap).
- **PodcastFillers range requests work:** the split zip's index and both licence files were read
  without downloading the 25 GB archive.
  - `LICENSE_ANNOTATIONS_ADOBE_RESEARCH_LICENSE.txt`: the Adobe Research License grants use
    "for noncommercial research purposes only" (§1.1). That fits evaluation-only use (never
    training, never tuning). Full text in `licence/podcastfillers/`.
  - `LICENSE_AUDIO.csv`, 199 episodes: CC BY 3.0 100, CC BY-SA 3.0 79, CC BY-ND 3.0 20. **No
    non-commercial audio.** Transcribing for evaluation, without redistributing or releasing
    derived audio, fits all three.
- **Stop 1, PodcastFillers CSV:** `KeyError: 'fullvoc_label'`. The column name from the website
  was not a key as the file was read. Fix: columns are found by normalised name from a short alias
  list (BOM, spaces and case ignored; `e3score.pf_columns`, tested). A probe records the header and
  the label and split value counts (no annotation rows). If a column is still missing, the run stops
  with `podcastfillers_columns_unknown` **before** any large download.
- **Stop 2 (found in the probe), AMI disfluency layer:** 160 files with 33,032 `<dsfl>` elements
  and **no type attribute**; each has `nite:child` word links (26,485) and `nite:pointer` links
  (69,773, attribute `role`). The type is behind the pointers. Fix: the parser follows each
  pointer into the AMI ontology files (`ontologies/…`); a node's label is its attribute values
  plus its ancestors'. A disfluency is a **repetition** if a label it points to contains "repet".
  The probe now records pointer roles, link targets, ontology files and resolved labels. If no
  repetition resolves, repetitions are reported as **not scored** (with the reason) and the
  filler scoring still runs. Repetitions are never guessed.
- **Order:** the cheap checks (PodcastFillers columns, AMI types) now run first, then the PodcastFillers
  WAVs, the AMI headsets and LibriSpeech.
- **Tip for the next run:** if the b3 Colab runtime is still connected, running the new notebook in
  the same runtime reuses the model and the AMI headsets already on its disk (only results are
  checked against the build).

## Consequences

- PR-4 (T1-r1) and later T1 runs are scored by this exact protocol; only the model changes.
- Open after the run: the DisfluencySpeech builder check result, the model-identity check, the
  disfluency-layer probe, PodcastFillers licence texts and access path, the AMI test-headset and
  PodcastFillers WAV pins, and the base and base + `initial_prompt` numbers.
- **This decision is wrong if** the probe shows the disfluency layer marks repetitions in a way
  the parser does not read, or if the converted model differs from the app's (then E3 would
  measure another model).
