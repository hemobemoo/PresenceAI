# Data sources

**Use statement (used unchanged in the docs, model cards and report drafts):**
All planned data sources, books and datasets are used for non-commercial academic research and
graduation-project purposes only, under each source's terms, with each source acknowledged.

Rules that apply to every source (ADR 0010, AGENTS.md):

- **Research-only or no-derivatives sources are used for evaluation only, never for training.**
- Where a source has privacy or usage terms: no redistribution of raw data or clips, no attempt to
  identify speakers or faces, and only derived features are kept where the terms require it.
- Model cards state any non-commercial training data. Only models trained on data that allows it
  may ever be published.
- Nothing listed as a candidate is downloaded until the owner approves that source.
- Everything downloaded lives under `data/` (git-ignored) and is recorded in `data/README.md`.

Marks: ✓ = licence and access read on the source page (date given) · ~ = confirm on the source page
before download and before citing (including the exact paper citation).

## 1. Sources in use or already planned

| Source | Licence | Use | Restrictions | Attribution text | Checked |
|---|---|---|---|---|---|
| AAC&U VALUE Oral Communication Rubric | CC BY-NC-SA 4.0 | Rubric dimensions and anchors (adapted) | Non-commercial; share-alike: the adapted rubric is released under CC BY-NC-SA 4.0; never training text. Copy used (2026-11-21, git-ignored `data/texts/raw/`): UT Tyler mirror, SHA-256 `ba41c136…bde`, structure only, no sentence copied; the Binghamton copy is gone (404). This mirror shows no licence line; the licence is the one checked on 2026-11-11 (owner's decision, ADR 0017) | "Adapted from the Oral Communication VALUE Rubric, Association of American Colleges and Universities (AAC&U), CC BY-NC-SA 4.0." | ✓ 2026-11-11 |
| *Fundamentals of Public Speaking* (Lumen Learning, FSCJ Pressbooks) | CC BY 4.0 | Knowledge base and T2 training text | Attribution | "*Fundamentals of Public Speaking*, Lumen Learning, CC BY 4.0, https://fscj.pressbooks.pub/publicspeaking." | ✓ 2026-11-11 |
| *Stand up, Speak out* (University of Minnesota Libraries Publishing) | CC BY-NC-SA 4.0 | Knowledge base only | Non-commercial; share-alike; flagged and excluded from training | "*Stand up, Speak out: The Practice and Ethics of Public Speaking*, University of Minnesota Libraries Publishing, CC BY-NC-SA 4.0." | ✓ 2026-11-11 |
| *Exploring Public Speaking*, 4th ed. (Tucker et al.) | CC BY-NC-SA 4.0 | Knowledge base only | Non-commercial; share-alike; excluded from training | "*Exploring Public Speaking*, 4th ed., Maricopa Open Digital Press, CC BY-NC-SA 4.0." | ✓ 2026-11-11 |
| *Speak Out, Call In* (Mapes) | CC BY-NC-SA 4.0 | Knowledge base only | Non-commercial; share-alike; excluded from training | "*Speak Out, Call In: Public Speaking as Advocacy*, Meggie Mapes, CC BY-NC-SA 4.0." | ✓ 2026-11-11 |
| DisfluencySpeech (amaai-lab) | Apache-2.0 | T1 training; E3 evaluation (test split) | Single speaker; scripts derive from Switchboard transcripts ~ | "DisfluencySpeech, AMAAI Lab, Apache-2.0, https://huggingface.co/datasets/amaai-lab/DisfluencySpeech." Paper ~ | ✓ 2026-11-17 |
| LibriSpeech test-clean | CC BY 4.0 | Evaluation only (E1, E3 WER regression) | Attribution | "LibriSpeech ASR corpus, OpenSLR 12, CC BY 4.0." Paper ~ | ~ |
| Common Voice (English accents; Arabic) | CC0-1.0 | Evaluation only (E1 accent and Arabic slices) | Privacy terms: downloaders agree not to determine the identity of speakers; no redistribution of clips. Download through Mozilla Data Collective (sign-in) | "Mozilla Common Voice, CC0-1.0." Cite the exact release used (e.g. "Common Voice Scripted Speech 23.0 – Arabic") | ✓ 2026-11-17 (Arabic: §3) |
| MASC (Massive Arabic Speech Corpus) | CC BY 4.0 (IEEE DataPort; one HF mirror says CC BY-NC 4.0, so the DataPort terms are used) | Evaluation only (E1 Arabic) | Attribution; no redistribution | "MASC: Massive Arabic Speech Corpus, IEEE DataPort, CC BY 4.0." Paper ~ | ✓ 2026-11-11 (conflict noted) |
| EdAcc (Edinburgh accent corpus) | ~ confirm on card | Evaluation only (E1 accents) until the licence is confirmed | ~ | ~ | ~ |
| Speak & Improve Corpus 2025 | Non-commercial, registration | Evaluation only | Text and audio never sent to OpenRouter or any hosted model; local models only | Exact wording from the licence on registration ~ | ✓ 2026-11-11 |
| PodcastFillers | Annotations and metadata: Adobe non-commercial research licence ("academic research and teaching only"); audio: per-episode CC BY 3.0 / CC BY-SA 3.0 / CC BY-ND 3.0 | Evaluation only (E3) | Research-only → never training; no redistribution of clips | "PodcastFillers dataset, Adobe Research, non-commercial research licence, https://podcastfillers.github.io." Plus the per-episode credit from the dataset CSV. Paper ~ | ✓ 2026-11-17 |
| TED Talks | CC BY-NC-ND 4.0 | Evaluation only (E12 reference ranges; E13 expert agreement); teaching links | No derivatives → never training; no edits; no scraping; obtain through TED's own download option only; keep derived features and talk URLs, delete video after extraction | "Talk title, speaker, TED, CC BY-NC-ND 4.0, talk URL." | ✓ 2026-11-11 |
| TED-LIUM 3 | CC BY-NC-ND 3.0 | Evaluation only (E1 presentation speech; E2 on real speech, see §2) | No derivatives → never training; no redistribution | "TED-LIUM release 3, OpenSLR 51, CC BY-NC-ND 3.0." Paper ~ | ✓ 2026-11-11 |
| L2-ARCTIC (non-native English, including Arabic-L1 speakers) — **candidate** | CC BY-NC 4.0 (non-commercial, acknowledged by the owner) ~ confirm on the download page | Evaluation only: an independent check of the language-check threshold on Arabic-L1 English (ADR 0016 open item 7); possibly E1 accent slices | **Approved for later** (owner, 2026-11-21). Downloaded only when the check is scheduled, **on Kaggle, never on the owner's laptop**; never training; no redistribution; access terms (request form) ~ confirm before download | ~ exact wording from the corpus page | ~ |
| Kokoro-82M (TTS weights and six voices, generates C1) | Apache-2.0 (weights and voices). Model card: trained on public-domain audio, permissively licensed audio (Apache, MIT and similar, including SIWIS CC BY 4.0 and Koniwa CC BY 3.0) and synthetic audio from closed commercial TTS models | Generates synthetic speech C1 (E2) | Runs only on Colab, in its own environment: the `kokoro` package pulls in `phonemizer-fork` (GPL-3.0+) and `espeakng-loader` (MIT, bundling espeak-ng, GPL-3.0), so it never enters the app's dependencies; espeak-ng is never loaded or called; every script word is checked against misaki's word lists first (ADR 0014 §1). Revision `f3ff3571…`, SHA-256 per file in `configs/evaluation.yaml: c1` | "Kokoro-82M, hexgrad, Apache-2.0." | ✓ 2026-11-17 |
| C1 scripts (`resources/c1/scripts.yaml`, 60 scripts) | Ours (released with the code) | C1 text | Written for PresenceAI in slice 5; no third-party text; owner reviews a sample of 10, automated checks cover all (ADR 0014 §2) | "Written for PresenceAI." | — |
| spaCy `en_core_web_sm` 3.8.0 | MIT | On Colab while generating C1 (misaki's tokenizer); **in the app** for the English text detectors (P2-b, ADR 0018) | Official wheel from spaCy's GitHub release; no published digest, so its SHA-256 was recorded on the first download and pinned in `configs/models.yaml` (`1932429d…fb85`, 2026-11-21); fetched only by `scripts/download_models.py --only spacy_en`, unpacked into `models/`, never pip-installed | "spaCy en_core_web_sm, Explosion, MIT." | ✓ 2026-11-21 |
| wordfreq 3.1.1 word-frequency data | Code Apache-2.0; data CC BY-SA 4.0, with SUBTLEX (credit Brysbaert et al.), OpenSubtitles (attribution) and Google Books Ngrams (acknowledgement) terms | Rare-term check of the language dimension (Zipf frequency) | Used at runtime only, installed as a package; not redistributed separately; credits in NOTICE | See NOTICE | ✓ 2026-11-21 (README) |
| Synthetic sets C1–C4 | Ours (released with the code) | C1 E2; C2 E4; C3 E9; C4 T2 | C2/C4 text generated by hosted models follows that model's terms (ADR 0003) | "Generated for PresenceAI." | — |
| Owner's own recordings (B1 pilot, B2 gaze protocol, E14 self-study) | Owner's own | T1 hand-corrected data, T3 training, detector development, E14 | Media the app records: encrypted at write time (Fernet, FINAL_11 §3). **B1** is recorded outside the app (Windows Sound Recorder) and stays as plain WAV, with its annotations and cached reports, in the git-ignored `data/b1/` on the owner's laptop only (ADR 0016, open item 3). Never committed, never synced to OneDrive, never uploaded to Kaggle, Colab, Hugging Face or any cloud; never released | — | — |

## 2. Candidates for C2 on public data (ADR 0010): licence, access, labels

Checked on 2026-11-17 from the source pages and search results. Nothing is downloaded yet.

| Candidate | Licence | Access | Labels useful to us | Speakers | Use if approved | Proposal |
|---|---|---|---|---|---|---|
| AMI Meeting Corpus | **CC BY 4.0** for all signals, the transcription, the *manual annotations package v1.6.2* and the *automatic annotations package v1.5.1* (official download page: both packages "license altered to CC BY 4.0", 10-Apr-2017). **Not covered:** the separately listed DOME dominance and social-role annotations (no licence stated; not used: they rate people, rule 3). Mirrors disagree (OpenSLR 16 lists CC BY-NC-SA 2.0 for v1.6.1; an HF card says non-commercial), so only the official Edinburgh files are used, and the `LICENCE` file inside the v1.6.2 zip is read and recorded before any other file is used (details in §4) | Official Edinburgh download only (not mirrors) | Manual v1.6 layers (official per-meeting table): word-level orthographic transcript with times (all meetings); **disfluency** (40 meetings: ES2002, ES2005, ES2006, ES2008, IS1001, IS1003, IS1009, TS3005, TS3006, TS3007, each a–d); **focus of attention** (head orientation and gaze target: table, whiteboard, slide screen, participants; 14 meetings); head, hand and leg movement; dialogue acts. AmiEmotion was never released and is never used | ~100 h of meetings, many speakers, many non-native | E1 (meeting speech WER); E3 (fillers vs the disfluency layer); E7 gaze-away sanity check on the 14 focus-of-attention meetings; T1 training allowed (CC BY 4.0) | **Shortlist** (E1, E3, E7 sanity check) |
| PodcastFillers | See §1 (research-only annotations) | Zenodo record 7121457 | 35k manually annotated filler words ("uh", "um") and other non-filler events, with timestamps; automatic transcripts | 199 episodes, many speakers | E3 only (evaluation) | **Shortlist** (E3) |
| DisfluencySpeech | Apache-2.0 | HF | Four transcript levels (with and without fillers, paralanguage) | **One** speaker | E3 and T1, but it adds no speaker variety for C2 | Keep as planned; not counted as multi-speaker |
| TED / TED-LIUM 3 | CC BY-NC-ND (see §1) | TED download option; OpenSLR 51 | TED-LIUM 3: aligned transcripts (segment times) for ~2,350 talks; TED.com: talk transcripts | ~2,000 speakers | E1; E2 on real speech (wpm and long-pause checks against segment and word times obtained by forced alignment, reported as approximate); E12; E13 talks | **Shortlist** (E1, E2-real, E12, E13) |
| TEDx (including Arabic TEDx) | Published under the TEDx licence on TEDx's YouTube channel; TED's usage policy does not allow re-uploading, and YouTube's terms do not allow downloading. No open download route found | YouTube only | Talk audio and video; captions vary | Many, including Arabic speakers | Arabic E12/E13 talks | **Not shortlisted** unless a permitted download route is confirmed (ask tedx@ted.com) ~. Arabic-language talks on TED.com itself are the fallback |
| Three Minute Thesis (3MT) videos | No open licence found; recordings belong to each university and speaker; UQ's site mentions institution and speaker permission even for indexing | YouTube and university pages | None (raw video) | Many | E12/E13 | **Not shortlisted** (no licence that covers analysis). Links may be used as teaching examples (FINAL_07 G) |
| Columbia Gaze Data Set | Free download for **non-commercial use only**; cite the paper | Columbia CAVE page | 5,880 images of 56 people; 5 head poses × 21 gaze directions (known angles) | 56 (21 Asian, 19 White, 8 South Asian, 7 Black, 4 Hispanic or Latino) | E7 cross-person gaze-zone check: zones from known gaze angles by a rule fixed before evaluation | **Shortlist** (E7), evaluation only; faces never shown or redistributed |
| MPIIFaceGaze | CC BY-NC-SA 4.0, non-commercial scientific use | perceptualui.org / DaRUS | Gaze targets on a laptop screen, face landmarks, 37,667 face images | 15 | Second E7 set (laptop viewing, close to our setting) | **Reserve** (evaluation only) |
| MIT Interview Dataset | "Available to other researchers" per the paper; no request form or licence text found | Contact the authors (Hoque group, now University of Rochester) ~ | 138 mock interviews, 69 students; Mechanical Turk ratings, which include trait and hireability judgements | 69 | At most: correlation of our observable measures with human ratings of observable behaviour | **Not shortlisted**: access unconfirmed, and its main labels are trait and hireability ratings that AGENTS.md rule 3 forbids us to model. Revisit only if you want it |

### Proposed shortlist (waiting for your approval, one source at a time)

| Experiment | Sources |
|---|---|
| E1 ASR accuracy | LibriSpeech test-clean, Common Voice (EN accents; **AR**), **MASC (AR)**, TED-LIUM 3 test, AMI (IHM) |
| E2 rate and pauses | C1 (EN, targets ADR 0009); TED-LIUM 3 as a real-speech check (approximate); **Arabic: owner's B1 only, see §3** |
| E3 fillers | DisfluencySpeech test, PodcastFillers test, AMI disfluency layer (40 meetings, CC BY 4.0) |
| E4–E6 content, rubric, calibration | C2 synthetic transcripts; criterion labels by the two raters on TED transcripts (evaluation only) |
| E7 vision | Owner's B2 protocol; Columbia Gaze (cross-person, posed still images: an approximation, see `docs/cards/LIMITATIONS.md`); AMI focus-of-attention (14 meetings) as a sanity check |
| E12 reference ranges | ~30 TED talks (derived features only) |
| E13 expert agreement | ~40 TED talks, including Arabic-language TED talks where available |

## 3. Arabic coverage (E1, E2)

TEDx is excluded (§2), so these are the public sources that could cover Arabic. Checked on 2026-11-17.

| Source | Licence | Speakers and size | Labels | Covers | Proposal |
|---|---|---|---|---|---|
| **Common Voice Arabic** (Scripted Speech 23.0) | CC0-1.0; downloaders agree not to determine the identity of speakers | 158 h recorded, 92 h validated, 1,654 speakers (Mozilla's automatically generated datasheet for release 23.0; the exact figures depend on the release, so the one used is cited) | One read sentence per clip with its transcript; optional self-reported demographics. **No word times and no pause labels** | E1 Arabic WER on read speech, many speakers | **Shortlist** (E1) |
| **MASC** (Massive Arabic Speech Corpus) | CC BY 4.0 (IEEE DataPort terms) | ~1,000 h, many speakers and dialects | Transcribed segments from online videos; segment-level times ~ (to confirm on DataPort); no word-level or pause ground truth found | E1 Arabic WER across dialects, spontaneous and broadcast speech | **Shortlist** (E1) |
| Arabic-language talks on TED.com | CC BY-NC-ND 4.0 | A small number of talks | TED transcripts | E12/E13 only (evaluation, derived features) | Keep for E12/E13 where available |
| QASR / MGB-2 (Al Jazeera broadcasts) | Research release by QCRI; terms need an agreement and are not confirmed as open | ~2,000 h / 1,200 h | Lightly supervised transcripts, segment-aligned | Would add broadcast Arabic to E1 | **Not shortlisted**: licence and access not confirmed; no word-level timing ground truth either |
| Kokoro (C1 generator) | Apache-2.0 | — | — | No Arabic voice in the MVP voice set | Cannot make Arabic C1 |

**Result:**
- **E1 (Arabic ASR) is covered** by Common Voice Arabic (read speech, 1,654 speakers) and MASC (multi-dialect).
- **E2 (Arabic rate and pauses) has no suitable public source.** None of the allowed Arabic sources has word-level timing or pause ground truth, and Kokoro cannot synthesise Arabic. Arabic E2 is therefore measured only on the owner's own annotated recordings (B1), one speaker, and is reported descriptively. This gap is stated in `docs/cards/LIMITATIONS.md`.
- Optionally, E2-style wpm on Common Voice Arabic can be reported against the reference word counts and VAD-measured speaking time. That is a consistency check, not ground truth.

## 4. Before any AMI file is downloaded

The licence is confirmed from the official download page (2026-11-17). Rules for using it:

1. Download only from `groups.inf.ed.ac.uk/ami/download/` (not OpenSLR or HF mirrors, whose licence text differs).
2. Use only the manual annotations package v1.6.2, the automatic annotations package v1.5.1 and the signals. Never use the DOME or social-role files.
3. When the v1.6.2 zip is downloaded, after the owner's approval, its `LICENCE` file is read first and its wording is recorded in `data/README.md`. If it does not say CC BY 4.0, stop and ask before using any file.
4. Before any tuning, the speaker list in the corpus metadata is checked: no speaker may appear in both an E3 development session and a test session (ADR 0012). The result is recorded in ADR 0012.
5. Attribution: "AMI Meeting Corpus, University of Edinburgh / AMI Consortium, CC BY 4.0, https://groups.inf.ed.ac.uk/ami/." Paper ~
