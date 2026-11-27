# 0027 — T1 data preparation: verbatim text style, AMI segments, licence-first data run

- Status: **proposed** (owner's answers P1–P4 of 2026-11-27 written in; waits for the owner's
  acceptance). Values marked PROVISIONAL may change only with a new decision recorded here.
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

## Consequences

- E3 (PR-3) scores DisfluencySpeech against the same text style T1 learns, so partial words are not
  counted as errors against `transcript_a`.
- T1 can count false starts (partial words), which `speech/repetitions.py` left for T1.
- **Open, recorded after the data run:** licence wording, speaker-check result (ADR 0012), probe
  findings and any spelling-map change, hours and segments kept, pinned hashes.
- **This decision is wrong if** the text check shows many DisfluencySpeech rows that differ from
  `transcript_a` outside partial words (the builder misreads the markup), or if the probe shows AMI
  marks partial words in a way the parser does not read.
