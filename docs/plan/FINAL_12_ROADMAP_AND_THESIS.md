# FINAL 12 — Roadmap, Risks, Thesis Map, Demo

Assumes ~18 working weeks. Two tracks from day one: **Build** and **Study/Writing**. Feature freeze at the end of week 12.

## 1. Week-1–2 spikes (de-risk before building)

| Spike | Question | Pass condition | If it fails |
|---|---|---|---|
| S1 Live capture | Can FastRTC inside `gr.Blocks` stream mic + webcam to Python handlers locally, with meters updating? | ≥ 8 fps landmarks, < 300 ms meter delay on your laptop | Two Gradio streaming components (`gr.Audio` + `gr.Image` webcam) |
| S2 ZeroGPU | Can a private/unlisted Gradio ZeroGPU Space run faster-whisper on a 3-min clip within a free visitor's quota? | One analysis uses < 60 GPU-s | Space runs Whisper `small` on CPU, or Stage B dropped |
| S3 Unsloth Whisper | Does the Unsloth Whisper notebook train on 200 DisfluencySpeech clips on Kaggle? | Loss falls; checkpoint converts to CTranslate2 and loads in faster-whisper | Use plain HF `Seq2SeqTrainer` with PEFT |
| S4 OpenRouter | Which free model gives valid JSON, verified quotes and allows training on outputs? | ≥ 95% JSON validity, ≥ 90% quote found on 30 prompts | Next model; templates as floor |
| S5 Gaze | Can iris + head-pose features separate camera / notes / away for you? | ≥ 0.8 macro-F1 on your own protocol | Merge `at_screen` into `at_camera` |

**Status (2026-11-13):** S1 passed (ADR 0002). S4 JSON/quote conditions met by Nemotron 3 Ultra, Nemotron 3.5 Super VL and Flint (ADR 0003); T2 teacher open (no provider confirmed to allow training on outputs). S5, S3 and S2 to run, in that order.

## 2. Phases

| Phase | Weeks | Build | Study / writing | Exit criteria |
|---|---|---|---|---|
| P0 Setup + spikes | 1–2 | Repo, uv, CI, AGENTS.md, MCP, models download, S1–S5 | Supervisor: ethics route, TED use, NVIDIA free-tier reading (trial/evaluation only; ADR 0003), hosted-LLM data policy (free endpoints may log or train on prompts; consent switch (d)); consent draft (EN/AR); start Ch. 2 reading | CI green; spike results in ADRs |
| P1 Speech | 3–4 | `speech/*`, CLI analyze, C1 synthetic speech, Arabic language pack basics, including the code-switching rule (`lang/ar/coach_prompt.yaml`, ADR 0003) | Record pilot set B1 | E2 passes on C1 |
| P2 Text + rubric + KB | 5–6 | `text/*`, `rubric/*`, VALUE-adapted rubric, knowledge base from open textbooks, C2 synthetic transcripts | Rubric anchors reviewed by supervisor | Assessment with verified evidence for every finding |
| P3 App | 7 | Gradio pages, storage, encryption, retention, export/delete, EN/AR UI | Ch. 3 skeleton | Record → report end to end locally; safety tests pass |
| P4 Training | 6–9 (parallel) | T1 on Kaggle (2–3 runs), T3 on protocol data, T2 pairs (open-weight Apache-2.0/MIT teacher run on Kaggle, so only the model licence applies; ADR 0003) + first Coach-SLM | — | T1 in faster-whisper; E3 first numbers; T3 F1 recorded |
| P5 Engine | 8–9 | Policies, library (~140 items), adaptive live prompts, simulator, progress page | Engine section of Ch. 3 | Full journey works; E9 plots |
| P6 Vision + live loop | 10–11 | Live meters and prompts, posture/gesture/expression events, calibration, low-light / low-camera-fps notice, battery run | Robustness matrix B2 recorded | E7 run; unstable measures disabled |
| P7 Coach + pilot prep | 12 | Guardrails, provider chain (prompt builder with `coach_prompt.yaml` rules), consent-(d) and payload tests (text only, names redacted; (d) off → no OpenRouter call), follow-ups, rater page, randomisation; optional Stage B deploy | Protocol and analysis plan committed; raters trained | Dry-run session ≤ 45 min; **feature freeze** `v0.9-study` |
| P8 Study + experiments | 13–15 | Bug fixes only; run E1–E9, E11, E12 on the frozen tag | Participants (13–14), blind rating (14–15), annotation | All experiments reproduce from clean clone |
| P9 Write + ship | 16–18 | Cards, README, demo fixtures, `v1.0` | E10 analysis; Ch. 4–6; abstract; slides | Draft to supervisor; demo rehearsed twice |

## 3. Cut order if time runs short

1. Stage B online pilot (keep in-person study).
2. T2 Coach-SLM (keep OpenRouter + templates; report as future work).
3. Kokoro voice.
4. Gesture and posture measures (keep framing, head pose, gaze).
5. Expression events.
6. E12 TED analysis.

Never cut: evidence linking, adaptive + fixed policies, simulator, T1 (main training result), gaze zones, export/delete, guardrail tests, blind rating.

## 4. Risks

| Risk | L | I | Mitigation |
|---|---|---|---|
| HF free-tier rules change again | M | M | Stage B optional; Stage A complete on its own; Azure for Students fallback |
| ZeroGPU quota too small for visitors | M | M | S2 measures it; Whisper `small` on CPU fallback |
| FastRTC/Gradio streaming unstable on some browsers | M | H | S1; fallback components; upload-a-recording path always available |
| Kaggle quota or session limits | M | M | Small models first (whisper-small); Colab as second source; checkpoints every epoch |
| T1 does not improve fillers | M | M | Still publishable as a negative result; fusion detector remains |
| Webcam gaze too inaccurate | M | M | Coarse zones only; per-user calibration; report accuracy |
| OpenRouter free model withdrawn mid-study | H | M | Pinned and first fallback are both NVIDIA (single-provider risk, ADR 0003); Flint (Recursive) as emergency fallback, `coach_slm`, cache, templates floor; freeze model for study weeks |
| NVIDIA trial term judged to exclude the study | M | M | Supervisor confirms the reading (ADR 0003); if excluded, Flint becomes pinned |
| No hosted teacher allowed for T2 | H | L | Open-weight teacher on Kaggle; T2 is second in the cut order |
| Ethics approval or recruitment slow | H | H | Start week 1; crossover design at n ≈ 12; simulation carries C1 |
| Licence surprise (TED, NC-SA texts) | M | M | Supervisor check in week 1; NC-SA text excluded from training |
| Expression labels perceived as emotion detection | M | H | Fixed label set, UI copy review, explanation on About page |
| Agent-written code you cannot defend | M | H | You review `engine/`, `rubric/`, `guardrails`, `expressions` line by line; ADRs |

## 5. Thesis template mapping (BAU)

| Chapter | Content | Source files |
|---|---|---|
| 1 Introduction | Problem, aims, objectives, stakeholders (students, job seekers, learners, instructors, researchers), challenges (verbatim fillers, webcam gaze, small n, zero-cost hosting, ethics line), outline | Concept, FINAL_01 |
| 2 Related Works | AI speaking/interview coaches (MACH, Naim et al., commercial tools); disfluency detection (PodcastFillers, CrisperWhisper); spoken assessment (Speak & Improve); gaze estimation with webcams; adaptive learning (knowledge tracing, bandits); emotion-inference critique and EU AI Act; comparison table | FINAL_07 F, arXiv MCP |
| 3 Methodology | Measurement definitions, rubric adaptation from VALUE, engine mathematics, training methods (T1–T3), study design, analysis plan | FINAL_04, 05, 06, 10 |
| 4 Design & Implementation | Architecture, live loop, deployment stages, data model, UI screens, privacy mechanisms, tooling and agents; feasibility spikes | FINAL_02, 03, 08, 09, 11; `docs/report-drafts/S*_summary.md` |
| 5 Results & Discussion | E1–E12 tables and figures, fairness slices, limitations | `experiments/results/` |
| 6 Conclusion & Future Work | Answer RQs within what data supports; future: full Arabic, avatar coach, public deployment, interview scenario, longer study | — |

Keep a weekly half-page log in `docs/thesis/log.md`; it becomes most of Chapter 4.

## 6. Competition demo (5 minutes)

| Time | Shown |
|---|---|
| 0:00 | Problem in one sentence; "your voice and video stay on this laptop; only text goes to the AI wording model, and that switch is here" |
| 0:30 | Short presentation live, with meters and one live prompt appearing |
| 1:45 | Report: transcript evidence, timeline with fillers (T1), gaze zones, expression events labelled descriptively, rubric cards with confidence |
| 2:45 | Engine decision and its reason; 60-second targeted exercise with its live prompt |
| 3:45 | Second attempt; side-by-side change |
| 4:30 | Progress bands; Delete; folder gone |

Fallback: pre-recorded consented before/after fixtures with cached LLM responses, run offline.

## 7. First three days

1. Create the repo from FINAL_08; copy FINAL files to `docs/plan/`; add AGENTS.md, `.mcp.json`, CI.
2. `uv add` core dependencies; run `scripts/download_models.py`; put the OpenRouter key in `.env`; email your supervisor about ethics approval and TED research use.
3. Start S1 (live capture) and S3 (Unsloth Whisper on Kaggle) in parallel; record three 2-minute pilot presentations (one in Arabic).
