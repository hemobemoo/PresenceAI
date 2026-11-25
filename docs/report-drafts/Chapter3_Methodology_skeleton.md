# Chapter 3 — Methodology (skeleton)

> **Reference draft, not the report.** This file is source material for Chapter 3 of
> `Main_Project_Report.docx`, which stays on the owner's laptop and outside git. It follows the
> register rules in `docs/report-drafts/README.md` (third person, formal, no inference about
> people, every figure traceable). Editor notes in this style mark what each section still needs;
> they are not report text. Numbers are not copied here: each table points to the configuration
> file or ADR that holds the value, so the report uses the value that is current when it is
> written.

---

## 3.1 Overview of the method

PresenceAI is a locally run web application for rehearsing short spoken presentations. A
practice session passes through five stages: consent and setup, recording, measurement,
rubric assessment and feedback, and the choice of the next exercise. Every stage runs on the
user's own computer; the only optional outside service is a hosted language model that receives
text alone, and only when the user turns that switch on.

The method rests on three principles that recur in every section of this chapter:

1. **Observable behaviour only.** The system measures what can be seen or heard in a recording
   (speech rate, pauses, fillers, the presence of a stated central message, gaze direction
   zones) and never infers emotion, personality, confidence or any other inner state.
2. **Evidence for every statement.** Each feedback statement is tied to a transcript span, a
   time span, or both, and carries a confidence value. A statement whose evidence cannot be
   verified is not shown.
3. **Missing is not low.** A measure that cannot be computed reliably in a recording is reported
   as "not measured", with its reason, and never counts as a poor result.

*Figure 3.1 — Pipeline from recording to report and next exercise.* Source:
`docs/plan/FINAL_02_ARCHITECTURE_AND_DEPLOYMENT.md`.

> Editor note: one paragraph on the two contributions (C1 adaptive exercise selection,
> C2 multimodal measurement) and how the sections below map to them. Source: FINAL_01 §5.

## 3.2 Practice scenarios

The system supports two fully implemented scenarios.

- **S1 Prepared short presentation.** A talk of two to three minutes on a topic chosen by the
  user, assessed against an adaptation of the AAC&U VALUE Oral Communication rubric.
- **S2 Project defence with follow-up questions.** A 90-second explanation of a project,
  followed by two follow-up questions shown on screen. The questions are drawn at random from a
  fixed bank of skeptical questions in the session language, or typed by the user. Each answer
  has a fixed answer window; if no speech is detected, the question is offered once more, and a
  second silence is recorded as no answer.

*Table 3.1 — Scenario formats, time limits and answer windows.* Source: `scenarios/*.yaml`,
`configs/default.yaml` (`app`), ADR 0022, ADR 0024.

> Editor note: list the stub scenarios (S3–S10) in one sentence as library entries without
> assessment.

## 3.3 Recording and data handling

Audio is captured in the browser and streamed to the local application. Recorded media are
encrypted at the moment they are written, with a key read from a git-ignored environment file on the laptop; the
assessment report is stored encrypted as well, and only a text-free summary of levels and
measure values is kept in the local database. A completeness check compares the seconds of
audio received with the wall-clock length of the recording and warns the user when audio
appears to be missing. Retention, export and deletion are under the user's control.

*Table 3.2 — What is stored, in what form, and for how long.* Source: ADR 0023, ADR 0024,
`docs/plan/FINAL_11_ETHICS_PRIVACY.md`.

> Editor note: state what encryption protects against and what it does not (ADR 0023), and the
> no-outside-connection guarantee during recording and analysis with the test that enforces it
> (ADR 0024, "Correction" section).

## 3.4 Speech measurement

Speech is transcribed with Whisper (faster-whisper, int8 on the CPU) and segmented with a voice
activity detector. The measures below are computed from the transcript, the voice activity
regions and the waveform.

| Measure | Definition (summary) | Source |
|---|---|---|
| Speech rate | Words divided by speaking time, from the first speech onset to the last offset; words transcribed after the last detected speech are removed first | ADR 0011, ADR 0015 |
| Articulation rate | Words divided by time inside detected speech | ADR 0011 |
| Pauses | Silent gaps above fixed durations, with their position (mid-sentence or at a sentence boundary); boundary pauses are not penalised | ADR 0011 |
| Fillers and discourse markers | Hesitation words per 100 words, reported as a lower bound with the base model; discourse markers counted separately, for information only | ADR 0012 |
| Repetitions | Repeated short word sequences within a short interval | ADR 0012 |
| Loudness variation | Relative level in one-second speech windows; never an absolute level | ADR 0013 |
| Pitch variation | Fundamental frequency in semitones from the speaker's own median | ADR 0013 |
| Timing | Elapsed time against the scenario limit, and overrun | ADR 0024 |
| Language check | Share of the transcript in the expected language | ADR 0016 |

*Table 3.3 — Thresholds and their status (provisional or decided).* Source: `configs/default.yaml` (`speech`, `asr`).

> Editor note: explain the confidence attached to each measure (FINAL_04 §6) and why rate
> targets are relative to the user's own band rather than one norm.

## 3.5 Content measurement

Content measures are computed from the transcript with rule-based detectors: cue lexicons,
position rules (opening and closing shares of the talk) and part-of-speech patterns. They cover
the central message, organisation (opening, body, conclusion and transitions), the opening hook,
supporting material by type (example, statistic, analogy, quotation) and language suited to the
audience (low-frequency terms without a definition, sentence length).

*Table 3.4 — Content detectors, cue sources and limits.* Source: ADR 0018,
`src/presenceai/lang/en/text_cues.yaml` (English only).

> Editor note: describe the knowledge base built from open textbooks and the embedding model
> used for retrieval and for follow-up relevance (ADR 0020). State that the cue lexicons are
> tuned only on the development part of the synthetic transcript set (3.9).

## 3.6 Follow-up handling (S2)

Each S2 answer is measured on three properties:

- **Latency:** seconds from the end of the question to the first speech onset, less a reading
  allowance proportional to the length of the question.
- **Relevance:** cosine similarity between the question and the answer transcript, from a
  multilingual sentence-embedding model. For questions drawn from the fixed bank, relevance is
  reported for information and does not enter the level; for questions the user types, it does.
- **Answered:** whether every answer window contains speech.

A further cue, an explicit statement of a limitation of the work, is required for the highest
level. English only for relevance and the limitation cue; Arabic follow-up handling therefore
reaches at most level 3, and the report says so.

*Table 3.5 — Follow-up conditions per level.* Source: ADR 0022, `configs/rubric.yaml`.

## 3.7 Rubric adaptation and scoring

The rubric adapts the five dimensions of the AAC&U VALUE Oral Communication rubric
(organisation, language, delivery, supporting material, central message) into observable
anchors at four levels, reviewed by the project supervisor. S2 reuses the same anchors with
count thresholds scaled to 90 seconds and adds follow-up handling as a sixth dimension.

For each dimension the level is the highest one whose computable conditions all hold. The
dimension's confidence is the lowest confidence among its computed measures. When half or fewer
of a dimension's measures could be computed, the level is capped, and the report states how many
measures it rests on. Measures that are absent by the user's choice (for example, camera off)
are left out of that count; measures absent for system reasons (for example, a measure not yet
available in Arabic) are included.

*Table 3.6 — Dimensions, measures and level conditions.* Source:
`resources/rubrics/s1_presentation.yaml`, `resources/rubrics/s2_project_defense.yaml`,
ADR 0017, ADR 0019, ADR 0022.

> Editor note: add the evidence-verification rule (a quote must match the transcript at its
> offsets; a time span must lie inside the recording) and how language-model claims are checked
> (ADR 0019 item 5, AGENTS.md rule 4).

## 3.8 Feedback wording

Feedback statements follow one pattern: an observable behaviour, a number and a location in the
recording. Templates exist in English and Arabic. Statements never describe the speaker's
feelings or traits.

> Editor note: one example statement in each language from `lang/{en,ar}/report.yaml`; describe
> the optional hosted wording model and its consent switch (ADR 0003).

## 3.9 Data used for development and evaluation

No multi-participant study was conducted. Development and evaluation rely on:

- **C1 synthetic speech** with known word and pause timings, split by script into development
  and test parts (ADR 0014);
- **C2 synthetic presentation transcripts** generated by an open-weight language model with
  controlled defects, split into development and test parts, with hand labels for the central
  message on the development part (ADR 0021);
- **B1 pilot recordings** of the project owner in English and Arabic (ADR 0016);
- **public datasets** used under their terms, evaluation-only where the terms require it
  (`docs/data/DATA_SOURCES.md`).

> Editor note: state the rule that thresholds are tuned only on development parts and that each
> test part is scored once (ADR 0014 ledger).

## 3.10 Adaptive exercise selection

> Editor note: complete after the engine is built (FINAL_05). Content: the skill taxonomy, the
> Gaussian belief per skill with its update rule, the "weakest reliable skill" policy with
> Thompson sampling over exercises, the difficulty ladder, the handling of floor effects, and
> the simulator with synthetic learners. Equations from FINAL_05 §2–§3.

## 3.11 Model training

> Editor note: complete after training (FINAL_06). Content: T1 verbatim Whisper fine-tuning with
> LoRA (Unsloth) for filler transcription; T3 gaze-zone classifier; T2 small feedback model if
> kept. Training runs on Google Colab free GPUs with fixed seeds and logged data manifests;
> trained models are converted for CPU inference on the laptop.

## 3.12 Vision measurement

> Editor note: complete after the vision stage (FINAL_04 §4). Content: framing and head
> orientation, three gaze zones (`at_camera`, `down_notes`, `away`; ADR 0004), and the fixed set
> of descriptive facial-movement events, which never feed scores. Camera-off sessions are fully
> supported.

## 3.13 Evaluation design

The evaluation combines system experiments on public and synthetic data, a simulation of the
adaptive engine, an agreement check between the engine and two blind expert raters on public
talks, and a single-case self-study with the project owner.

*Table 3.7 — Experiments, data and metrics (E1–E14).* Source:
`docs/plan/FINAL_10_EVALUATION.md` §1.

> Editor note: state what each kind of evidence can and cannot support (FINAL_10 §4), and the
> fairness and robustness slices (FINAL_10 §2).

## 3.14 Ethical and privacy considerations in the method

> Editor note: summarise the consent switches, local processing, encryption, retention and
> deletion, the exclusion of emotion inference and face recognition, and the data-use statement
> for public sources. Source: FINAL_11, ADR 0010, ADR 0023.

---

## Sources

- `docs/plan/FINAL_01_PROJECT_SCOPE.md`, `FINAL_02`, `FINAL_04`, `FINAL_05`, `FINAL_06`,
  `FINAL_10`, `FINAL_11`
- ADRs 0003, 0004, 0010–0024 in `docs/adr/`
- `configs/`, `scenarios/`, `resources/rubrics/`, `src/presenceai/lang/`
