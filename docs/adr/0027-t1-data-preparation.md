# 0027 — T1 data preparation: verbatim text style, AMI segments, licence-first data run

- Status: **accepted** (owner, 2026-11-27; answers P1–P4, runs d2 and d3, and the builder pass
  line in §6 written in). Values marked PROVISIONAL may change only with a new decision recorded
  here.
- Date: 2026-11-27
- Decider: project owner
- Plan reference: ADR 0026 §3 (T1 data), §1 (ground rules); FINAL_06 §3; FINAL_10 E3;
  `docs/data/DATA_SOURCES.md` §4 (AMI rules); ADR 0005 (S3), ADR 0012 (E3 split, speaker check)
- Config keys affected: `configs/evaluation.yaml: t1` (new), `configs/evaluation.yaml:
  public_data` (new)

## Context

ADR 0026 approved T1-r2 on AMI meetings outside the ten E3 sessions. Those meetings have only the
word-level orthographic transcript, no disfluency layer. PR-2 must turn them, and
DisfluencySpeech, into one training text style, with a fixed amount of speech, without
downloading anything to the laptop, and with the AMI licence read before any other AMI file.

## Decision

### 1. One verbatim text style (owner P2, P3)

Used for every T1 training source **and for E3's DisfluencySpeech references** (PR-3 builds them
with the same `t1text.build_ds_text`, so T1's partial words are not scored against
`transcript_a`-style text).

- **Fillers kept.** Matching uses the app's `filler_key` and `lang/en/fillers.yaml`
  (hesitations).
- **Partial words keep a trailing hyphen** (`th-`). `configs/evaluation.yaml: t1.partial_marker`.
- **A partial filler counts as the filler**: `um-` is one "um", not a partial word.
- **Backchannels** (`mm-hmm`, `uh-huh`, `mm-mm`, `uh-uh`; `t1.ami.backchannels`) are kept as
  written, count as **zero** fillers and are **never split at the hyphen** (`mm` and `hmm` are
  hesitations, so a split would add two).
- **DisfluencySpeech text comes from `transcript_annotated`**, with markup removed (`{F …}`,
  `{D …}`, `{C …}`, `[ … + … ]`, `--`, lone `-`, `<tags>`), fillers and partials kept.
  **This departs from S3 (ADR 0005)**, which trained on `transcript_a`; `transcript_a` writes
  partial words as the full word ("over th-" → "over the"), which hides false starts. The data run
  checks, row by row, that the built text equals `transcript_a` except at partial words, and
  reports every other difference.
- **AMI spelling** (`t1.ami.spelling`): AMI filler spellings map onto the E3 lexicon (`Um` → `um`,
  `Uhm` → `um`, …); a word marked truncated gets the hyphen. The map is a first version; the data
  run's format probe lists the filler-like spellings AMI actually uses, and any change to the map is
  recorded here before T1-r2.

### 2. AMI segments and amount (owner P1, P4)

- **10 h of speech kept in the manifest, counted after the crosstalk filter**
  (`t1.ami.train_speech_hours: 10.0`), not raw audio.
- Pool: **scenario meetings** (ES, IS, TS) outside the ten E3 sessions
  (`e3.ami_split.pool`), minus every meeting with a speaker from those sessions.
- Order: seeded shuffle within each site (seed `20261127`), then round-robin ES → IS → TS; whole
  meetings in that order, the last one cut segment by segment at the target.
- Segments, per speaker headset, cut at that speaker's own pauses:

  | Key (`t1.ami.segment`) | Value | Status |
  |---|---|---|
  | `split_gap_s` | 0.5 | PROVISIONAL |
  | `max_s` | 30.0 | Whisper's window |
  | `min_s` | 1.0 | PROVISIONAL |
  | `pad_s` | 0.1 | PROVISIONAL |
  | `crosstalk_max_share` | 0.20 | owner P4, PROVISIONAL |
  | `drop_if_gap_element` | true | PROVISIONAL |

  Crosstalk share = the part of the segment covered by other speakers' timed words.
- **Weak repetition signal:** these meetings have no disfluency layer, so AMI adds only what the
  orthographic transcript shows (repeated words as written, partial words). This is a T1
  limitation (ADR 0026 §3).

### 3. The data run (Colab, CPU runtime)

`training/t1_whisper_verbatim/colab_data.py build` writes the notebook; the run, in order:

1. downloads the AMI manual annotations v1.6.2 zip (official Edinburgh URL) and records its
   SHA-256;
2. **reads only its licence file**; if the text does not state CC BY 4.0 (or names a
   NonCommercial, NoDerivatives or ShareAlike term), **it stops before any other AMI file is
   opened** (DATA_SOURCES §4). The full wording is saved and goes into `data/README.md`;
3. extracts; runs the **format probe** (elements, `<w>` attributes, filler-like spellings,
   untimed words, unknown elements), the **speaker check** (ADR 0012) and the manifest;
4. downloads each selected headset, records SHA-256, sample rate and duration, checks that every
   segment ends inside its file, and deletes the audio;
5. loads DisfluencySpeech at revision `b7da294` and runs the text check of §1;
6. downloads LibriSpeech test-clean, records its SHA-256 and file counts, deletes it.

**Resumable, after the first attempt was lost (owner, 2026-11-27).** Run `20261127-d1` took about
27 minutes, then Colab disconnected at the zip download and everything was lost. Also, the old last
cell (delete the runtime) ran straight after the download pop-up under Run all, which can cut the
download off. Changes:
- the results folder is on the owner's Google Drive (`MyDrive/presenceai_p4/p4data_<id>/p4data`,
  results only; datasets stay on the Colab disk, ADR 0026 D2); every stage writes there when it
  finishes, and a rerun of the same build skips finished stages and headsets (a folder from
  another build is refused);
- the zip is also copied to that Drive folder, so a failed browser download can be taken from Drive;
- the delete-runtime cell does nothing under Run all (`DELETE_RUNTIME = False`; run by hand);
- faster: headsets 3 at a time (the AMI site asks for limited bandwidth), and DisfluencySpeech reads
  only its two text columns from the pinned parquet files instead of 1.4 GB with audio;
- **`HF_TOKEN`** (the owner's Hugging Face *read* token) may be added as a Colab Secret. The notebook
  passes it to the script as an environment variable; it is used only to read DisfluencySpeech from
  Hugging Face, never printed or written; without it, or if it is refused, the public dataset is
  read anonymously. It does not speed up the AMI or LibriSpeech downloads (other servers).

Results come back as `p4data_<id>.zip` in `data/colab_results/`; `ingest` checks every SHA-256,
puts the manifest in the git-ignored `data/t1/<id>/` and the rest in
`experiments/results/t1/data_<id>/`, then deletes the zip (ADR 0026 §1). The hashes are then
pinned in `public_data.*.sha256`, so later runs stop on any change.

### 4. Format assumptions

The AMI file layout and attribute names (`<w punc trunc>`, `meetings.xml` `<speaker nxt_agent
global_name channel>`, `Headset-<channel>.wav`) come from the corpus documentation and common
usage, not from the files, because nothing is downloaded to the laptop. The tests use
hand-written fixtures in that format. **The data run's probe is the check**: if it reports unknown
elements, no `trunc` attribute, or zero segments, the parser is fixed and the run repeated before
anything is used.

### 5. Run `20261127-d2` (Colab CPU, 2026-11-27): stopped at headset 51

Results: `experiments/results/t1/data_20261127-d2/` (ingest checked all 9 files). 184.5 s, status
`error`.

- **Licence: passed.** The zip holds one licence file, `LICENCE.txt`. It opens: "The AMI corpus
  and its annotations are released under the Creative Commons Attribution 4.0 International Public
  License agreement (CC BY 4.0)", followed by the CC BY 4.0 legal code. Full text in
  `licence/AMI_LICENCE.txt`. The zip (22,887,865 bytes) is now pinned:
  `public_data.ami_annotations.sha256 = b56e5bab…bc9f9d`.
- **Format probe: the assumptions hold**, with one addition. Elements: `w` 548,077,
  `vocalsound` 13,525, `disfmarker` 12,184, `gap` 2,574, and **`transformerror` 3** (unexpected).
  `<w>` attributes: `starttime`/`endtime` (1 word untimed), `punc` 79,585, **`trunc` 9,103**
  (partial words, as assumed), `mispronounced` 1,261 and `pron` 69 (the word is kept as written;
  these attributes are not used).
  - **`transformerror` is treated like `<gap>`**: a segment holding one is dropped (PROVISIONAL;
    3 occurrences).
- **Filler spellings found:** `uh` 13,209, `um` 6,485, `mm` 2,161, `hmm` 597, `ah` 238, `hm` 10,
  `er` 2. Backchannels: `mm-hmm` 1,502, `uh-huh` 112, `mm-mm` 27, `uh-uh` 6, `mm-hm` 1, `mmm-hmm` 1.
  The last two reduce to the same key as `mm-hmm`, so they also count zero. **No spelling-map change
  is needed.**
- **Bug found and fixed: one-letter cut-off words counted as fillers.** In the kept segments, `m-`
  (61 times, e.g. "you m- mute") was counted as the filler "mm", because letter-run collapsing gives
  `m-` and `mm` the same key. A partial now counts as a filler only if its stem has two or more
  letters. `um-` is still "um" (owner P3); `m-` and `u-` are partial words.
- **Speaker check (ADR 0012): passed.** 12 speakers in the 3 dev sessions, 28 in the 7 test
  sessions, none shared. The report listed the dev sessions as "missing". That was a bracket bug in
  the report line; the pass/fail value was computed correctly. Fixed, with a test.
- **Pool:** 98 scenario meetings. Excluded: 40 (the ten E3 sessions × 4 meetings), 33 non-scenario,
  **0** for a shared speaker.
- **Manifest (d2 code):** 10.0001 h kept, 4,445 segments from 35 meetings (140 headsets), 99,604
  words. Of 41,623 candidate segments, dropped: too short (< 1 s) 14,139; crosstalk 12,812; gap
  2,379. Run d3 recomputes it with the fixes above, so its hash and counts differ slightly.
- **Headsets:** 50 of 140 downloaded (mean 59.1 MB, 16 kHz mono, every segment inside its file).
  Then the AMI server answered **HTTP 503**. Fix: a busy server (429/5xx), a reset or a timeout is
  retried after 15, 30, 60, 120, 240 and 480 s (or the server's `Retry-After`), and headsets are
  fetched **2 at a time** instead of 3. A missing file (404) is not retried.
- DisfluencySpeech and LibriSpeech were not reached. The code changed, so the next run is a new
  build (`20261127-d3`, a new Drive folder; a resumed folder must come from the same build).

### 6. Run `20261127-d3` (Colab CPU, 2026-11-27): complete

Results: `experiments/results/t1/data_20261127-d3/` (ingest checked all 11 files). Status `ok`,
694.5 s, no resumed stage.

- **Licence** passed again; the annotation zip matched its pin from d2.
- **Speaker check** passed: 12 dev / 28 test speakers, none shared, **no missing session** (the
  report fix works).
- **Format probe:** same elements as d2; no unknown element now that `transformerror` is known.
- **Manifest (final, pinned):** **10.0001 h of speech kept**, 4,445 segments, 35 meetings, 140
  headsets, 99,604 words, **5,784 fillers** (d2: 5,845; the 61 one-letter partials are no longer
  counted) and **1,718 partial words** (d2: 1,657). Dropped of 41,623 candidates: too short 14,139,
  crosstalk 12,811, gap or transformerror 2,380 (one segment moved from crosstalk to gap). Same
  meetings as d2. `t1.ami.manifest_sha256 = fd51f661…a693a8`; T1-r2 checks it before training, and
  a data run that rebuilds a different manifest stops with `manifest_changed`.
- **Headsets:** all 140 downloaded and hashed (8.43 GB, 16 kHz mono, every segment inside its
  file), 5 retries in total, no failure. The 50 files hashed in d2 have **identical** hashes in d3.
  Pins: `configs/pins/ami_headsets.json` (`public_data.ami_headset.sha256_file`).
- **LibriSpeech test-clean:** 346,663,984 bytes, SHA-256 `39fde525…967e6c23` (pinned), 2,620 FLAC
  files and 87 transcript files. Its OpenSLR page states **CC BY 4.0** (checked 2026-11-27).
- **DisfluencySpeech text check** (read with the token; only the two text columns of the 5 parquet
  files): of 5,000 rows, **4,806 (96.1 %)** gave the same words as `transcript_a` except at partial
  words (train 4,332 of 4,500, validation 240 of 250, test 234 of 250). The old check compared exact
  tokens, so punctuation counted as a difference. Reading the 60 saved examples:
  - **most are punctuation only** (`transcript_a` ends a clause with "." where the annotation has
    ",", or detaches a comma). E3 strips punctuation, so they do not matter;
  - **a builder gap:** Switchboard's uncertain-word marks were kept (`((of … stuff)).`,
    `((Uh)),`), and a doubled comma (`know,,`). **Fixed**: the marks are removed and the words kept;
    `,,` becomes `,`. Tests added;
  - with the fix, **2 of the 60** still differ in words. Both are rare partial-word notations: `a
    -bout` (hyphen in front) and a `we-` that only `transcript_a` marks. Left as they are, as
    label noise.
  - The data run's check now sorts every row into `same`, `punctuation_only`, `partials_only` and
    `words_differ`. **The fixed builder has not yet been checked on all 5,000 rows.** That check runs
    as the first step of PR-3's Colab run, which loads DisfluencySpeech anyway, so no extra data
    run is needed. T1 training (PR-4) uses only the checked builder.
  - **Pass line, fixed before PR-3 runs it (owner, 2026-11-27):** at most **1 % of the 5,000
    rows (50)** may be `words_differ`; `same`, `punctuation_only` and `partials_only` all pass.
    If `words_differ` is above 50, the builder is fixed and the check repeated before PR-4; **the
    line is not moved.** The PR-3 run reports the four counts per split (train, validation, test).
    Key: `configs/evaluation.yaml: t1.disfluencyspeech.text_check_max_words_differ: 50`.

## Consequences

- E3 (PR-3) scores DisfluencySpeech against the same text style T1 learns, so partial words are not
  counted as errors against `transcript_a`.
- T1 can count false starts (partial words), which `speech/repetitions.py` left for T1.
- **Recorded:** licence wording, speaker check, probe findings (no spelling-map change) and the
  annotation-zip hash (d2, §5); final manifest, every headset hash, LibriSpeech hash and licence,
  and the DisfluencySpeech text check (d3, §6). **Still open:** the fixed builder's check on all
  5,000 DisfluencySpeech rows, run as the first step of PR-3's Colab run, **pass = at most 50 rows
  `words_differ`** (§6; fixed before the run, never moved). Above 50: fix the builder and repeat
  the check before PR-4.
- **This decision is wrong if** the text check shows many DisfluencySpeech rows that differ from
  `transcript_a` outside partial words (the builder misreads the markup), or if the probe shows AMI
  marks partial words in a way the parser does not read.
