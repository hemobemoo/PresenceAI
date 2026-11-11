# PresenceAI — Final Plan: Read Me and Decision Register

Version: final-1 · 11 Nov 2026 · supersedes `implementation-plan/00–08` (kept for history, not deleted).

Legend used in every FINAL file: ✓ = checked on the provider's own page on 11 Nov 2026 · ~ = from secondary sources or general knowledge; confirm before you rely on it or cite it.

## 1. Files

| File | Answers |
|---|---|
| `FINAL_00_README_DECISIONS.md` | This file: every decision, where it came from, what changed |
| `FINAL_01_PROJECT_SCOPE.md` | Users, scenarios, capabilities, MVP vs future, research questions, contributions |
| `FINAL_02_ARCHITECTURE_AND_DEPLOYMENT.md` | Hybrid local/cloud split, live loop, deployment stages, zero-cost service map, storage |
| `FINAL_03_TECH_STACK.md` | Every library, model and service, with licence and role |
| `FINAL_04_MULTIMODAL_MEASUREMENT.md` | Exact definition of every audio, text and visual measure, live and post-recording |
| `FINAL_05_ADAPTIVE_ENGINE.md` | Skill model, rubric, exercise selection, adaptive live prompts, baselines, simulator |
| `FINAL_06_MODELS_AND_TRAINING.md` | Pretrained models, the three training tracks, Unsloth/Kaggle workflow, serving |
| `FINAL_07_DATA_SOURCES.md` | Every data source in your priority order, with links and download commands |
| `FINAL_08_FOLDER_STRUCTURE.md` | Repository tree, dependency groups, commands |
| `FINAL_09_REPOS_MCP_AGENT_SETUP.md` | Starred repos verdict, MCP servers, Claude Code and Cline configuration |
| `FINAL_10_EVALUATION.md` | Experiments E1–E12, exploratory user study, statistics |
| `FINAL_11_ETHICS_PRIVACY.md` | Consent, retention, the expression/emotion line, legal references, tests |
| `FINAL_12_ROADMAP_AND_THESIS.md` | 18-week plan, spikes, risks, thesis chapter map, demo |

## 2. Your answers → decisions

### Round 1

| # | Your answer | Decision in this plan |
|---|---|---|
| 1 | Users: all three groups equally | Product language and scenarios suit students, job seekers and general learners. Study recruits a mixed sample (`FINAL_01` §2) |
| 2 | Primary: general presentation / public speaking. Secondary: project defense | Primary scenario **"Prepared short presentation (2–3 min)"**. Secondary **"Project defense with skeptical follow-up questions"**. Interview is a YAML stub |
| 3 | Hybrid; cloud where it suits, local where it suits; must cost nothing | See round 2 Q1/Q2 and `FINAL_02` |
| 4 | Process in memory and delete, unless that hurts the engine | Superseded by round 2 Q3 (encrypted temporary storage). The engine only needs derived features, so retention is driven by study rating and re-analysis needs, not by the engine |
| 5 | Framing/visibility, head orientation, posture, gestures, eye gaze, facial expressions | All six are in the MVP vision module (`FINAL_04` §4). Slide alignment and background/lighting are not product features; lighting is kept only as a pre-recording quality check |
| 6 | Both real-time and post-recording | Live meters and live prompts while speaking; full analysis after (`FINAL_02` §3) |
| 7 | Text coach, optional static image, no avatar | Confirmed. Animated coach is future work |
| 8 | English and Arabic | See round 2 Q6 |
| 9 | Data priority F > E > D > C > G > B > A | `FINAL_07` is ordered exactly this way, with each source's role |
| 10 | Fine-tune a language/speech/vision model | See round 2 Q4 |
| 11 | Contributions: adaptive exercise recommendation; multimodal communication measurement | These are the two named contributions (`FINAL_01` §5). Calibration and fairness become supporting evidence |
| 12 | About 20 participants, mixed | Two-arm exploratory study, n ≈ 20 (`FINAL_10` §3) |
| 13 | Private research prototype now; public later | No public launch in scope. Online access is by invitation only |
| 14 | New finalized file set | This set |

### Round 2

| # | Your answer | Decision |
|---|---|---|
| 1 | You choose the best hybrid split under zero cost (free student cloud offers noted) | **Local laptop is the main runtime** (capture, live loop, analysis, storage). **Cloud does what the laptop cannot do for free**: language model (OpenRouter), model training (Kaggle/Colab GPUs via Unsloth), model and dataset hosting (Hugging Face Hub), and an **optional invited online pilot** (Hugging Face ZeroGPU Space + Supabase free). Azure for Students and Oracle Always Free are documented fallbacks, not defaults. Rationale in `FINAL_02` §1 |
| 2 | No paid APIs or cloud bills; free tiers for development and research; public deployment later | Every service in the default path has a $0 tier that covers the study. Anything requiring a card on file is marked and kept out of the default path |
| 3 | Raw media stored encrypted temporarily, user-controlled deletion, explicit consent | Encrypted at rest from the moment it is written; default retention 30 days (study: until blind rating ends, max 60 days); delete button always available (`FINAL_11` §3) |
| 4 | Fine-tune a language model and a speech or vision model | Three training tracks: **T1 Verbatim-Whisper** (speech, primary), **T2 Coach-SLM** (small language model for feedback wording), **T3 gaze-zone classifier** (small vision model on landmarks). OpenRouter stays the default LLM; T2 is evaluated against it (`FINAL_06`) |
| 5 | Yes: exploratory study; main evidence from system experiments and simulation | Built into `FINAL_10` |
| 6 | English full; Arabic ASR + basic measures + Arabic feedback templates; full Arabic is future work | Implemented as language packs (`FINAL_04` §5) |
| 7 | Books/articles: all roles (knowledge base, training text, rubric source) | Only openly licensed texts are used: rubric adapted from the AAC&U VALUE Oral Communication rubric, knowledge base and training text from open public-speaking textbooks (`FINAL_07` section F) |
| 8 | TED/public videos: both research data and teaching examples | Research: a small, manually curated TED set analysed locally, only derived features stored. Teaching: links and TED's own embed player. Licence limits in `FINAL_07` section E |
| 9 | No to geometry-only; labelling how the user is looking, or emotion, may be good | **Partly adopted.** The app labels gaze direction and readily apparent facial expressions (smile, raised brows, lowered brows, eyes closed, mouth movement). It does **not** label emotions (happy, nervous, confident, etc.). See §3 below |
| 10 | Both live meters and live prompts | Live meters always; live prompts rule-based, rate-limited, chosen by the engine to match the exercise's target skill |
| 11 | You choose; should be a web UI | Web UI in **Gradio + FastRTC** (Python only). Order: local web app → optional online pilot → public deployment as future work |
| 12 | Exact links and commands | Given in `FINAL_07` and `FINAL_06`; restricted datasets get the request link instead of a command |

## 3. The one place this plan does not follow your answer literally

Round 2 Q9 suggested emotion labels "may be a good addition". The plan labels expressions but not emotions, for three reasons:

1. **Your own concept forbids it.** The concept's final statement says PresenceAI "does not infer personality, emotion, confidence, honesty, or professional worth". Reversing that removes the main ethical differentiator from your thesis and competition pitch.
2. **The EU AI Act draws exactly this line.** Recital 18 says detecting readily apparent expressions such as a smile or a frown is not emotion recognition, unless they are used to infer emotions; Article 5(1)(f) prohibits emotion inference in education and workplace settings, which is where students and job seekers would use this. Jordan is not bound by the Act, but your concept cites it.
3. **Scientific validity.** The 2019 review your concept cites found facial movements are not reliable indicators of specific emotions across people and cultures.

So the app says "visible smile during 8% of the opening" and never "you looked happy" or "you seemed nervous". If you still want emotion labels, tell me and I will add them as a clearly separated, opt-in research module with its own consent and its own limitations section, but I recommend against it.

## 4. What changed from the earlier plan

| Area | Earlier (implementation-plan/) | Final |
|---|---|---|
| UI | NiceGUI, Python-side capture | Gradio + FastRTC web UI, browser capture, still Python only |
| Primary scenario | Project defense | General prepared presentation; project defense secondary |
| Runtime | Local-first, single machine | Local web app first, optional invited online pilot (ZeroGPU Space) |
| Raw media | Optional, user choice | Encrypted temporary storage with consent, 30-day default |
| Vision | Geometry only, blendshapes off | Adds gaze zones and expression labels from blendshapes; emotion labels still excluded |
| Live feedback | Level meter and preview only | Live meters + engine-chosen live prompts |
| Training | None | T1 Whisper verbatim fine-tune, T2 Coach-SLM, T3 gaze classifier |
| Language | English; Arabic later | English full; Arabic ASR, basic measures, Arabic templates |
| Data | Own recordings first | Your priority order: texts, TED, public speech data, synthetic, instructional video, pilot, participants |
| Rubric source | Concept document | AAC&U VALUE Oral Communication rubric, adapted to observable anchors |
| Starred repos used | NiceGUI, DuckDB, Laya | DuckDB, Laya, Unsloth, Supabase (online pilot), plus agent tooling |

## 5. Facts that most affect the plan (verified 11 Nov 2026)

- **Hugging Face Spaces changed in 2026**: free personal accounts can no longer create Gradio/Docker Spaces on CPU Basic, but can host up to 2 Gradio Spaces on ZeroGPU (account older than 30 days, verified email). GPU time is charged to each visitor's own daily quota (a few minutes per day for a free account). ✓
- **OpenRouter free models**: 20 requests/minute and 1,000/day once 10 credits have been bought. ✓
- **Kaggle** gives roughly 30 GPU-hours per week; the exact quota floats and is shown in your account. ~
- **Unsloth** supports Whisper (speech-to-text) fine-tuning and has a ready Colab notebook. ✓
- **Your Unsloth Gemma embedding GGUF** is EmbeddingGemma under the Gemma Terms of Use, which is free but not an OSI open-source licence. It is fine for this private prototype; an Apache-2.0 alternative is kept behind the same interface. ✓
- **Oracle Always Free** Arm allowance was halved to 2 OCPU / 12 GB in June 2026 and needs card verification. ✓

## 6. Assumptions still open (change them and tell your agent)

| Assumption | Effect | Where |
|---|---|---|
| Your laptop has 8–16 GB RAM and no NVIDIA GPU | Training on Kaggle/Colab, Whisper `small`/`turbo` int8 on CPU | `FINAL_06`, `FINAL_03` §9 |
| You are an enrolled student with a university email | Azure for Students fallback is available | `FINAL_02` §5 |
| About 18 working weeks remain | Phase lengths | `FINAL_12` |
| Your Hugging Face account is older than 30 days by week 12 | ZeroGPU online pilot is possible | `FINAL_02` §4 |
