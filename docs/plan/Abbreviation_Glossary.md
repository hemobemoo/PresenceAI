# PresenceAI — Abbreviation Glossary

Source: the final plan documents (FINAL_00–FINAL_13), the ADRs in `docs/adr/` and related project
files. Checked against them on 2026-11-17, after ADR 0010 (no volunteer study) and ADR 0011.

Watch for labels that mean different things in different places: **S1/S2** (spike or scenario),
**T1/T2** (training track, or a former study time point), **C1/C2** (contribution, or synthetic
data set) and **CI** (Continuous Integration or confidence interval). The context decides.

---

## Project / Planning / Process

| Abbreviation | Full form / meaning |
|---|---|
| **ADR** | Architecture Decision Record: one page per design or threshold decision, stored in `docs/adr/` (AGENTS.md rule 7) |
| **P0–P9** | Project phases (FINAL_12): P0 Setup + spikes, P1 Speech, P2 Text + rubric + KB, P3 App, P4 Training, P5 Engine, P6 Vision + live loop, P7 Coach + evaluation prep, P8 Experiments, P9 Write + ship |
| **S1–S5** (spikes) | Week 1–2 de-risking spikes: S1 Live capture, S2 ZeroGPU, S3 Unsloth Whisper, S4 OpenRouter, S5 Gaze |
| **S1–S10** (scenarios) | Practice scenarios: **S1** Prepared short presentation (primary), **S2** Project defense with skeptical follow-up (secondary), S3 STAR interview answer (stub), S4–S10 other communication scenarios (stubs) |
| **T1, T2, T3** | Training tracks: **T1** Verbatim-Whisper (speech fine-tune for fillers), **T2** Coach-SLM (small language model for feedback wording), **T3** Gaze-zone classifier |
| **T0 / T1 / T2** (removed) | Former study time points (baseline, post-practice, follow-up) of the removed user study E10; not used since ADR 0010 |
| **E1–E14** | Experiments (FINAL_10): E1–E9, E11, E12 system experiments and simulation; **E13** engine vs expert raters on public talks; **E14** single-case (n = 1) ABAB self-study. **E10** (exploratory user study) was removed by ADR 0010 |
| **C1, C2** (contributions) | C1 Adaptive exercise recommendation, C2 Multimodal communication measurement (FINAL_01 §5) |
| **C1–C4** (data sets) | Synthetic data sets (FINAL_07 C): C1 synthetic speech (Kokoro), C2 synthetic presentation transcripts, C3 synthetic learners, C4 teacher feedback pairs |
| **B1, B2, B3** | The owner's own data sets: B1 pilot presentations, B2 gaze/robustness protocol, B3 annotations |
| **RQ1–RQ6** | Research questions (FINAL_01 §6) |
| **ABAB** | Single-case design that alternates two conditions (A adaptive, B fixed) in blocks (E14) |
| **Stage A / Stage B** | Deployment stages: A the local app on the laptop (always available), B an optional hosted Space |
| **MVP** | Minimum Viable Product |
| **KB** | Knowledge Base (from open textbooks) |
| **CLI** | Command-Line Interface (`presenceai analyze`) |
| **CI** | Continuous Integration (GitHub Actions); in statistics also confidence interval |
| **YAML** | "YAML Ain't Markup Language": the format of the files in `configs/`, the language packs, scenarios and the exercise library |
| **i18n** | Internationalisation (English/Arabic language packs) |
| **RTL** | Right-to-left (Arabic text) |

---

## Technical / ML / Speech / Vision

| Abbreviation | Full form / meaning |
|---|---|
| **ASR** | Automatic Speech Recognition (faster-whisper) |
| **VAD** | Voice Activity Detection (Silero VAD, bundled with faster-whisper) |
| **WER** | Word Error Rate |
| **wpm** | Words per minute (speaking rate) |
| **F0 / pYIN** | Fundamental frequency (pitch) / probabilistic YIN pitch tracker (librosa) |
| **RMS** | Root mean square (loudness) |
| **SNR** | Signal-to-noise ratio: median speech level minus the noise floor (pre-recording check; loudness and pitch confidence, ADR 0013) |
| **dBFS** | Decibels relative to full scale (0 dBFS = the loudest level a file can store) |
| **AGC** | Automatic gain control (browser `autoGainControl`; turned off for live capture, ADR 0013) |
| **TTS** | Text-to-speech (Kokoro, used to generate C1) |
| **LLM** | Large Language Model (hosted through OpenRouter, only when the user enables AI wording) |
| **SLM** | Small Language Model (Coach-SLM = T2) |
| **GGUF** | Model file format run with `llama-cpp-python` (EmbeddingGemma, Coach-SLM) |
| **LoRA** | Low-Rank Adaptation (fine-tuning method used with Unsloth) |
| **PEFT** | Parameter-Efficient Fine-Tuning |
| **CTranslate2** | Inference format and runtime that faster-whisper uses for every Whisper model (base and fine-tuned) |
| **NLP** | Natural Language Processing |
| **fps** | Frames per second (landmarks / live vision) |
| **F1** | F1 score (harmonic mean of precision and recall) |
| **MAE** | Mean absolute error (E2: wpm and duration) |
| **p90** | 90th percentile (pause length) |
| **κ / Cohen's κ** | Cohen's kappa (agreement between two raters, or a rater and a detector) |
| **α / Krippendorff's α** | Krippendorff's alpha (agreement across more than two raters, E13) |
| **ANCOVA** (removed) | Analysis of covariance, planned for E10; removed with E10 (ADR 0010) |
| **HF** | Hugging Face |
| **ZeroGPU** | Hugging Face Spaces free GPU tier, charged to each visitor's daily quota |
| **WebRTC** | Web Real-Time Communication (through FastRTC) |
| **TURN** | Traversal Using Relays around NAT. Cloudflare TURN helper, only for hosted use (Stage B); the local app uses none (ADR 0002) |
| **MCP** | Model Context Protocol (servers used with Claude Code / Cline agents) |
| **UI** | User Interface |
| **EN / AR** | English / Arabic |
| **MSA** | Modern Standard Arabic |
| **PREP** | Point–Reason–Example–Point structure cue |
| **STAR** | Situation–Task–Action–Result interview structure |
| **AAC&U VALUE** | Association of American Colleges & Universities, VALUE Oral Communication rubric (adapted) |

---

## Licences and data terms

| Abbreviation | Meaning |
|---|---|
| **CC BY / BY-SA / BY-NC / BY-NC-SA / BY-NC-ND** | Creative Commons: attribution / share-alike / non-commercial / no derivatives combinations |
| **CC0** | Creative Commons public-domain dedication (Common Voice) |
| **GPL / AGPL / LGPL** | GNU General / Affero / Lesser General Public License (GPL and AGPL are never runtime dependencies, AGENTS.md rule 6) |
| **AMI** | AMI Meeting Corpus (multi-speaker meetings, CC BY 4.0) |
| **MASC** | Massive Arabic Speech Corpus |
| **TED-LIUM** | Corpus of TED talk audio with aligned transcripts (CC BY-NC-ND 3.0) |

---

## Other frequent terms

| Term | Meaning |
|---|---|
| **OpenRouter** | Hosted LLM API with free models, used to word feedback only when the user turns on AI wording (consent switch (d)); templates otherwise |
| **Unsloth** | Fast fine-tuning library (T1 Whisper, T2) |
| **MediaPipe** | Google's face/pose landmark and blendshape library |
| **faster-whisper** | Whisper runtime based on CTranslate2 |
| **Gradio + FastRTC** | Web UI + live audio/video streaming stack |
| **Supabase** | Free backend for encrypted media storage, Stage B only |
| **DuckDB** | Local analytics database; optional vector search for the KB |
| **Laya** | Library for typed rubric decisions |
| **Fernet** | Symmetric encryption used for raw media at write time (key `MEDIA_KEY`) |

---

*Checked against the PresenceAI plan and ADRs on 2026-11-17.*
