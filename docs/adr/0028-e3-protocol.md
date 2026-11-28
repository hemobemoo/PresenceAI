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

### 9. Run `20261128-b4` (Colab T4, 2026-11-28) and the Drive dataset cache

Record: `experiments/results/e3/base_20261128-b4/`. 5,646 s, status `error` in the data stage.

- **PodcastFillers solved.** The label column is `label_full_vocab` (the alias list found it).
  The test split has **20 episodes and 5,044 scored Uh/Um events**. All 20 test WAVs (1.93 GB)
  were read by range requests, CRC32-checked and hashed, then **pinned** in
  `configs/pins/podcastfillers_test_wav.json`. Label counts over all splits: Uh 17,907,
  Um 17,078, Words 12,709, Repetitions 9,024, Breath 8,288, …
- **AMI disfluency types resolved, but there is no repetition type.** The pointers
  (`dsfl-type` 26,045) lead into `ontologies/dsfl-types.xml`, whose types are: Hesitation 6,318,
  Reparandum 5,883, Reparans 4,779, Discourse Marker 2,989, Disruption 2,267, Slip of the tongue
  1,455, Omission 1,159, Stuttering 859, Explicit Editing Term 211, Other 125. The other pointer
  roles (`da-ref` to dialogue acts, `extension`) are not types. So "repet" matched nothing (0
  repetitions), and repetitions would have been reported as not scored. **Open, owner's
  decision:** how a repetition is defined from Reparandum/Reparans (see Consequences).
- **Stop:** AMI test headset `IS1003a.Headset-3.wav` still got HTTP 503 after its 6th retry (about
  16 min in total). 51 of the 112 test headsets had arrived.
- **Owner's decision (2026-11-28), amends ADR 0026 D2: a private Drive cache of pinned public
  datasets.** The owner has 400 GB of Google Drive until October 2027.
  - **What:** only pinned public files the owner approved (AMI, PodcastFillers test WAVs,
    LibriSpeech, the AMI annotation zip), in `MyDrive/presenceai_data/<set>/`, private, not
    shared, never redistributed. **Never the owner's own recordings** (ADR 0016).
  - **How:** `fetch_public_data.cache_get` / `cache_put`. A cached copy is used only if its
    SHA-256 matches the pin (a bad copy is deleted and fetched again). A downloaded pinned file is
    saved to the cache **as soon as it verifies**, so a later failure loses nothing. Unpinned
    files are never cached. Files are copied to Colab's local disk before use. Tested.
  - **First fill:** after b4, a one-off cell copied the runtime's verified files: 51 AMI test
    headsets, 20 PodcastFillers WAVs and the AMI annotation zip (every one matched its pin; none
    skipped).
  - **Delete list:** `data/README.md` lists each cached set; the owner deletes a set when it is
    no longer needed, and all of them by the end of the project.

### 10. Repetition definition (owner, 2026-11-28)

The owner accepted the proposed definition on the condition that it is **computed automatically,
with no manual labelling**. It is.

- **A repetition is a Reparandum whose normalised words equal the start of its Reparans** ("the
  the cat": reparandum "the", reparans "the"). Normalised means lower case with edge punctuation
  removed; a partial word keeps its hyphen, so "th- the" is a false start, not a repetition. Key:
  `e3.repetitions.definition: reparandum_repeated_in_reparans`.
- **Pairing a Reparandum with its Reparans:** a Reparans in the same parent `<dsfl>`, the first
  one after it; if they are not nested, the first Reparans that starts within **2 words**
  (`e3.repetitions.max_gap_words`, PROVISIONAL; room for an "uh" or "I mean") after the
  Reparandum ends. The probe counts how each pair was made (`repair_pairing`: same parent,
  following, unpaired) and how many repetitions result, so the next run shows which structure AMI
  uses before any number is relied on.
- Any element whose type itself contains "repet" still counts (none exist in AMI v1.6.2).
- ADR 0026 §4 condition 4 ("repetitions kept on AMI test ≥ base") is scored with this definition.
- Tested on hand-written nested and flat examples: a repeat is found; a correction ("go uh went")
  and a false start ("th- the") are not; a Reparans 7 words later is not paired.

### 11. Run `20261128-b5` (Colab T4, 2026-11-28): complete, the base numbers

Record: `experiments/results/e3/base_20261128-b5/` (`e3_metrics.json`). 12,209 s (3.4 h), status
`ok`, every stage done, no resume needed. Built from `cb1cdca`.

- **Checks:** stage 0 passed (20 of 5,000 `words_differ`, as b1–b4); the model was identical to the
  app (§7 rule); 11,408 items transcribed per condition (AMI 8,518 segments, DisfluencySpeech 250,
  PodcastFillers 20 episodes, LibriSpeech 2,620).
- **Data:** all 112 AMI test headsets matched their pins (51 from the Drive cache, 61 downloaded,
  8 retries in total); 19 PodcastFillers WAVs came from the cache, all 20 matched their pins.
- **Repetitions (§10), from the test meetings' disfluency layer:** 5,883 Reparanda; 4,774 paired
  with a Reparans in the same parent, 59 with a following one, 1,050 unpaired; **2,957
  repetitions**. 1,465 lie inside the deciding AMI segments, 2,746 with crosstalk kept.
- **Base numbers** (lexicon detector; F1 with 95 % bootstrap intervals):

| Set | base | base + `initial_prompt` |
|---|---|---|
| **AMI test** (deciding) | F1 0.079 [0.059, 0.102], P 0.84, R 0.04 | **F1 0.815 [0.790, 0.841]**, P 0.91, R 0.74 |
| AMI repetitions kept | 250 of 1,465 (0.171) | 685 of 1,465 (0.468) |
| **PodcastFillers test** (deciding) | F1 0.031 [0.013, 0.058], P ≥ 0.78, R 0.016 | **F1 0.038 [0.017, 0.066]**, P ≥ 0.81, R 0.019 |
| AMI test, crosstalk kept | F1 0.103 [0.078, 0.130] | F1 0.761 [0.736, 0.786] |
| DisfluencySpeech test | F1 0.059 [0.015, 0.113] | F1 0.942 [0.910, 0.967] |
| LibriSpeech test-clean WER % | 3.89 [3.42, 4.36] | 4.06 [3.58, 4.56] |

  The crosstalk filter excluded **38.6 %** of AMI test speech (about 20 % in training, ADR 0027).
- **Finding: the prompt reaches only the first 30 s of long audio.** With
  `condition_on_previous_text` false, faster-whisper 1.2.1 resets the prompt after the first
  window (`transcribe.py`, `prompt_reset_since`). AMI segments and DisfluencySpeech clips are
  short, so the prompt covers them; a PodcastFillers item is a whole episode, so the prompt covers
  its first 30 s only. That is why the prompt lifts AMI from 0.08 to 0.81 but PodcastFillers only
  from 0.031 to 0.038. The app decodes a whole answer the same way, so for answers longer than
  30 s the app also gets the prompt only at the start. This is the app's real behaviour, not a
  scoring error. **Open, owner's decision** (Consequences).
- **Pin-name fix:** in `configs/pins/podcastfillers_test_wav.json` one name ("Free Podcast
  Day_Home Grown Girls – Apr 7 2018") had its "–" stored mis-decoded (cp1252), so b5 found no
  pin for it: it was CRC32-checked but not compared with its pin and not cached. Its SHA-256
  equals the pinned value (`97b878ea…`), so the data is unchanged. Fixed: the name is clean UTF-8;
  the run now **stops on any PodcastFillers WAV or AMI test headset without a pin**; a test fails
  if a pin file holds mis-decoded text.

## Consequences

- **Owner's decision before PR-4, the PodcastFillers baseline with the prompt:**
  - (a) *recommended:* keep the protocol as it is. It measures what the app's settings really do;
    on PodcastFillers, "T1 vs base + prompt" is then close to "T1 vs base", and the AMI test
    carries the prompt comparison.
  - (b) add a reported-only condition that repeats the prompt in every 30-s window (decoding
    split windows), to show what a prompt could do on long audio. It needs one more Colab run over
    the 20 episodes; it is not an app setting, so it would not enter the pass rule.
- **Open: whether the app should repeat the prompt** for answers longer than 30 s is a separate
  app question (FINAL_10 E6 / ADR 0012), not part of E3.

- PR-4 (T1-r1) and later T1 runs are scored by this exact protocol; only the model changes.
- Closed by runs b1–b5: the DisfluencySpeech builder check, the model-identity check, the
  disfluency-layer probe, the PodcastFillers licence texts and access path, the AMI test-headset
  and PodcastFillers WAV pins, and the base and base + `initial_prompt` numbers (§11).
- **This decision is wrong if** the probe shows the disfluency layer marks repetitions in a way
  the parser does not read, or if the converted model differs from the app's (then E3 would
  measure another model).
