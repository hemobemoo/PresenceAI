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
| AAC&U VALUE Oral Communication Rubric | CC BY-NC-SA 4.0 | Rubric dimensions and anchors (adapted) | Non-commercial; share-alike: the adapted rubric is released under CC BY-NC-SA 4.0; never training text | "Adapted from the Oral Communication VALUE Rubric, Association of American Colleges and Universities (AAC&U), CC BY-NC-SA 4.0." | ✓ 2026-11-11 |
| *Fundamentals of Public Speaking* (Lumen Learning, FSCJ Pressbooks) | CC BY 4.0 | Knowledge base and T2 training text | Attribution | "*Fundamentals of Public Speaking*, Lumen Learning, CC BY 4.0, https://fscj.pressbooks.pub/publicspeaking." | ✓ 2026-11-11 |
| *Stand up, Speak out* (University of Minnesota Libraries Publishing) | CC BY-NC-SA 4.0 | Knowledge base only | Non-commercial; share-alike; flagged and excluded from training | "*Stand up, Speak out: The Practice and Ethics of Public Speaking*, University of Minnesota Libraries Publishing, CC BY-NC-SA 4.0." | ✓ 2026-11-11 |
| *Exploring Public Speaking*, 4th ed. (Tucker et al.) | CC BY-NC-SA 4.0 | Knowledge base only | Non-commercial; share-alike; excluded from training | "*Exploring Public Speaking*, 4th ed., Maricopa Open Digital Press, CC BY-NC-SA 4.0." | ✓ 2026-11-11 |
| *Speak Out, Call In* (Mapes) | CC BY-NC-SA 4.0 | Knowledge base only | Non-commercial; share-alike; excluded from training | "*Speak Out, Call In: Public Speaking as Advocacy*, Meggie Mapes, CC BY-NC-SA 4.0." | ✓ 2026-11-11 |
| DisfluencySpeech (amaai-lab) | Apache-2.0 | T1 training; E3 evaluation (test split) | Single speaker; scripts derive from Switchboard transcripts ~ | "DisfluencySpeech, AMAAI Lab, Apache-2.0, https://huggingface.co/datasets/amaai-lab/DisfluencySpeech." Paper ~ | ✓ 2026-11-17 |
| LibriSpeech test-clean | CC BY 4.0 | Evaluation only (E1, E3 WER regression) | Attribution | "LibriSpeech ASR corpus, OpenSLR 12, CC BY 4.0." Paper ~ | ~ |
| Common Voice (English accents; Arabic) | CC0-1.0 | Evaluation only (E1 accent and Arabic slices) | Privacy terms: no attempt to identify speakers; no redistribution of clips | "Mozilla Common Voice, CC0-1.0." | ✓ 2026-11-11 |
| MASC (Massive Arabic Speech Corpus) | CC BY 4.0 (IEEE DataPort; one HF mirror says CC BY-NC 4.0, so the DataPort terms are used) | Evaluation only (E1 Arabic) | Attribution; no redistribution | "MASC: Massive Arabic Speech Corpus, IEEE DataPort, CC BY 4.0." Paper ~ | ✓ 2026-11-11 (conflict noted) |
| EdAcc (Edinburgh accent corpus) | ~ confirm on card | Evaluation only (E1 accents) until the licence is confirmed | ~ | ~ | ~ |
| Speak & Improve Corpus 2025 | Non-commercial, registration | Evaluation only | Text and audio never sent to OpenRouter or any hosted model; local models only | Exact wording from the licence on registration ~ | ✓ 2026-11-11 |
| PodcastFillers | Annotations and metadata: Adobe non-commercial research licence ("academic research and teaching only"); audio: per-episode CC BY 3.0 / CC BY-SA 3.0 / CC BY-ND 3.0 | Evaluation only (E3) | Research-only → never training; no redistribution of clips | "PodcastFillers dataset, Adobe Research, non-commercial research licence, https://podcastfillers.github.io." Plus the per-episode credit from the dataset CSV. Paper ~ | ✓ 2026-11-17 |
| TED Talks | CC BY-NC-ND 4.0 | Evaluation only (E12 reference ranges; E13 expert agreement); teaching links | No derivatives → never training; no edits; no scraping; obtain through TED's own download option only; keep derived features and talk URLs, delete video after extraction | "Talk title, speaker, TED, CC BY-NC-ND 4.0, talk URL." | ✓ 2026-11-11 |
| TED-LIUM 3 | CC BY-NC-ND 3.0 | Evaluation only (E1 presentation speech; E2 on real speech, see §2) | No derivatives → never training; no redistribution | "TED-LIUM release 3, OpenSLR 51, CC BY-NC-ND 3.0." Paper ~ | ✓ 2026-11-11 |
| Kokoro-82M (TTS weights, generates C1) | Apache-2.0 | Generates synthetic speech C1 (E2) | C1 scripts use only words Kokoro already knows, so the GPL espeak-ng fallback is never called (owner decision 2026-11-17) | "Kokoro-82M, hexgrad, Apache-2.0." | ✓ 2026-11-14 |
| Synthetic sets C1–C4 | Ours (released with the code) | C1 E2; C2 E4; C3 E9; C4 T2 | C2/C4 text generated by hosted models follows that model's terms (ADR 0003) | "Generated for PresenceAI." | — |
| Owner's own recordings (B1 pilot, B2 gaze protocol, E14 self-study) | Owner's own | T1 hand-corrected data, T3 training, detector development, E14 | Encrypted at write time; never uploaded to training platforms; never released | — | — |

## 2. Candidates for C2 on public data (ADR 0010): licence, access, labels

Checked on 2026-11-17 from the source pages and search results. Nothing is downloaded yet.

| Candidate | Licence | Access | Labels useful to us | Speakers | Use if approved | Proposal |
|---|---|---|---|---|---|---|
| AMI Meeting Corpus | Signals, transcripts and *some* annotations CC BY 4.0 (current download page); 2006 release was CC BY-NC-SA 2.5. Which annotation layers are CC BY 4.0 is not stated, so check `LICENCE` in the annotation zip ~ | Free download (Edinburgh); HF mirror `edinburghcstr/ami` | Word-level orthographic transcripts with times; disfluency layer (subset ~); dialogue acts; focus-of-attention (head and gaze target) and head-gesture layers on a subset ~; close-up video | ~100 h of meetings, many speakers, many non-native | E1 (meeting speech WER), E3 (fillers vs disfluency layer), E7 gaze-away sanity check on focus-of-attention video ~; T1 training only on layers confirmed CC BY 4.0 | **Shortlist** (E1, E3); E7 only if focus-of-attention is CC BY 4.0 |
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
| E1 ASR accuracy | LibriSpeech test-clean, Common Voice (EN accents, AR), MASC, TED-LIUM 3 test, AMI (IHM) |
| E2 rate and pauses | C1 (targets, ADR 0009); TED-LIUM 3 as a real-speech check (approximate); owner's B1 for Arabic |
| E3 fillers | DisfluencySpeech test, PodcastFillers test, AMI disfluency layer ~ |
| E4–E6 content, rubric, calibration | C2 synthetic transcripts; criterion labels by the two raters on TED transcripts (evaluation only) |
| E7 vision | Owner's B2 protocol; Columbia Gaze (cross-person); AMI focus-of-attention ~ |
| E12 reference ranges | ~30 TED talks (derived features only) |
| E13 expert agreement | ~40 TED talks, including Arabic-language TED talks where available |
