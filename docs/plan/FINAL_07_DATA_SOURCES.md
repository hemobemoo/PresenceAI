# FINAL 07 — Data Sources (in your priority order: F > E > D > C > G > B > A)

✓ = licence/access read on the source page on 11 Nov 2026 · ~ = confirm on the page before download and before citing.

Store everything under `data/` (git-ignored) and record each item in `data/README.md`: name, version/date, URL, licence, citation, local path, used for (train / eval / knowledge / teaching).

## F. Books, articles and text (priority 1)

Roles you chose: knowledge base, training text, rubric definitions.

| Source | Licence | Role | Link | Checked |
|---|---|---|---|---|
| AAC&U **VALUE Oral Communication Rubric** | CC BY-NC-SA 4.0 | Rubric dimensions and anchor wording (adapted to observable anchors) | Copy used: `https://binghamton.edu/academics/general-education/instructorinformation/oralcommunicationaauc.pdf`; publisher page at aacu.org ~ | ✓ licence |
| *Fundamentals of Public Speaking* (Lumen Learning, via FSCJ) | **CC BY 4.0** | Knowledge base **and** T2 training text; exercise sources | `https://fscj.pressbooks.pub/publicspeaking` | ✓ |
| *Stand up, Speak out* (Univ. of Minnesota) | CC BY-NC-SA 4.0 | Knowledge base, exercise ideas | `https://open.lib.umn.edu/publicspeaking` | ✓ |
| *Exploring Public Speaking*, 4th ed. (Tucker) | CC BY-NC-SA 4.0 | Knowledge base | `https://open.maricopa.edu/exploringpublicspeaking` | ✓ |
| *Speak Out, Call In* (Mapes) | CC BY-NC-SA 4.0 | Knowledge base (audience, advocacy) | `https://open.umn.edu/opentextbooks/formats/1476` | ✓ |
| Research papers (Naim et al. on interviews, PodcastFillers, CrisperWhisper, Barrett et al. 2019, EU AI Act) | Citation only | Thesis Chapter 2, design justification | arXiv MCP server, publisher pages | ✓ (see FINAL_11) |
| Commercial books (e.g. popular presentation guides) | Copyright | Cited in the thesis only; never ingested | — | — |

Download (Pressbooks offers PDF/EPUB exports on each book's home page; use the site's export links):

```bash
mkdir -p data/texts/raw
# save each book's PDF/EPUB from its home page into data/texts/raw/, then:
uv run python scripts/build_knowledge_base.py --input data/texts/raw --out data/texts/kb.parquet
# chunk ~400 tokens, keep book/chapter/section/licence on every chunk, embed with EmbeddingGemma
```

Rules: every knowledge-base answer cites book and section; NC-SA chunks are flagged and excluded from T2 training data; the adapted rubric is published in `resources/rubrics/` under CC BY-NC-SA 4.0 with attribution.

## E. TED Talks and public presentation videos (priority 2)

Licence facts: TED Talks are CC BY-NC-ND 4.0; edits/derivatives are not allowed; embedding must use TED's player; scraping video from TED.com is not permitted. ✓ (`https://www.ted.com/about/programs-initiatives/ted-talks/ted-talks-usage-policy`)

| Use | How | Stored |
|---|---|---|
| **Teaching examples** | `resources/examples/ted_examples.yaml`: talk URL, skill it illustrates, start time, one-line note. Shown as a link or TED embed in the report | URLs only |
| **Research (E12)** | Curate ~30 talks across speakers and styles; obtain each through TED's own download option where offered (no scraping, no YouTube downloaders); analyse locally; keep only derived features; delete the video files after extraction | Parquet of derived features + talk URL |
| Audio + transcripts at scale | **TED-LIUM 3** (CC BY-NC-ND 3.0): `https://www.openslr.org/51/` | Eval only (ASR on presentation speech) |

```bash
# TED-LIUM 3 is ~50 GB; download only if you need it for E1
wget -P data/public/tedlium https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz   # ~
```

Ask your supervisor to confirm the TED research use before E12; it is the least certain licensing item in the plan.

## D. Public speech datasets (priority 3)

| Dataset | Role | Licence | Command / access | Checked |
|---|---|---|---|---|
| **DisfluencySpeech** | T1 training + filler eval | Apache-2.0 | `load_dataset("amaai-lab/DisfluencySpeech")` | ✓ |
| AMI Meeting Corpus (IHM) | T1 training (multi-speaker verbatim) | CC BY 4.0 ~ | `load_dataset("edinburghcstr/ami", "ihm")` ~ | ~ |
| **PodcastFillers** | Filler eval only | Non-commercial research (Adobe); audio CC-BY variants | `https://zenodo.org/records/7121457` | ✓ |
| LibriSpeech test-clean | ASR regression check | CC BY 4.0 | `wget https://www.openslr.org/resources/12/test-clean.tar.gz` | ~ |
| **Common Voice** (English with accents; Arabic) | Accent and Arabic WER slices | CC0-1.0 | Mozilla Data Collective, sign in and download per language: `https://datacollective.mozillafoundation.org` | ✓ |
| EdAcc | Accented conversational English | Confirm on card | `load_dataset("edinburghcstr/edacc")` | ✓ exists, ~ licence |
| **MASC** (Massive Arabic Speech Corpus, ~1,000 h, multi-dialect) | Arabic ASR eval | CC BY 4.0 per IEEE DataPort; one HF mirror says CC BY-NC 4.0 — use the DataPort terms | `https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus` | ✓ conflict noted |
| Speak & Improve Corpus 2025 | L2 English with holistic scores | Non-commercial, registration; local LLMs only | `https://researchdatasets.cambridge.org/datasets/speak-and-improve-corpus-2025` | ✓ |
| MIT Interview Dataset | Related observable measures vs human ratings | Research, on request | `https://roc-hci.com/past-projects/automated-prediction-of-job-interview-performances/` | ✓ |
| pyannote community-1 (model, optional) | Diarization | CC BY 4.0, gated | Accept terms on HF, then `Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=...)` | ✓ |

```bash
uv run python scripts/fetch_public_data.py --only disfluencyspeech,ami_ihm,librispeech_test_clean,edacc
# Common Voice, MASC, PodcastFillers, Speak & Improve, MIT Interview: manual download or request; the script verifies files you place in data/public/
```

Never send Speak & Improve text or audio to OpenRouter (its terms require local models or no-retention APIs). ✓

## C. Synthetic data (priority 4)

| Set | Size | Built by | Role |
|---|---|---|---|
| C1 Synthetic speech | ~300 clips | Kokoro TTS from scripts with known word counts, inserted silences, inserted filler tokens | Exact ground truth for rate/pause (E2) |
| C2 Synthetic presentation transcripts | ~400 | OpenRouter teacher writes presentations with controlled defects (no central message, no conclusion, no supporting material, undefined jargon); hand-checked | Content detector eval (E4); T2 inputs |
| C3 Synthetic learners | 2,000+ | `engine/simulator.py` | Policy comparison (E9) |
| C4 Teacher feedback pairs | ~3,000 | `FINAL_06` §5 | T2 training |

```bash
uv run presenceai data synth-speech --n 300 --seed 7
uv run presenceai data synth-text --n 400 --seed 7      # uses OpenRouter quota (~80 requests, 5 per call)
```

Kokoro has no Arabic voice in the MVP set ~, so Arabic rate/pause accuracy uses hand-annotated pilot recordings instead.

## G. Public instructional videos (priority 5)

Used as **links** in the teaching library only (YouTube/TED/university channels), with title, URL, skill and timestamp. Not downloaded, not analysed (YouTube's terms prohibit downloading; licences vary). TED-Ed videos are under YouTube's standard licence, not Creative Commons. ✓

## B. Your own pilot recordings (priority 6)

| Set | Size | Role |
|---|---|---|
| B1 Pilot presentations (you + 2–3 friends with consent) | 30–40 clips, EN and AR | Detector development, T1 hand-corrected verbatim data, fixtures |
| B2 Gaze/robustness protocol | You + 4–6 volunteers × ~16 conditions | T3 training and E7 |
| B3 Annotations | 40 clips in Label Studio | Fillers, pauses, structure, supporting material, gaze spot checks |

## A. Participant recordings (priority 7, exploratory study)

~20 participants × (baseline + practice + post + follow-up), blind human ratings by 2–3 raters. Pseudonymous; encrypted; deleted after rating per `FINAL_11`. Never uploaded to training platforms.

## What is released at the end

| Released | Not released |
|---|---|
| Code, rubrics (CC BY-NC-SA), exercise library, lexicons, C1–C3 synthetic sets, T1/T3 models with cards, T2 adapter if licences allow, derived metrics from consenting participants | Any raw voice/video, PodcastFillers/S&I/TED material, participant transcripts |
