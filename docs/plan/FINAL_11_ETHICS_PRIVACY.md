# FINAL 11 — Ethics, Privacy and Safety

## 1. Position

PresenceAI is a practice tool chosen by the user. It measures observable behaviour and content, shows evidence and confidence, and never judges the person. It is not a hiring, grading or surveillance tool.

## 2. The expression / emotion line

| Allowed (implemented) | Not allowed (blocked in code) |
|---|---|
| Gaze zone: at camera, at screen, down at notes, away | Any emotion: happy, sad, angry, surprised, nervous, anxious, bored, excited, embarrassed, ashamed |
| Head orientation, framing, posture, gestures | Mood, attitude, engagement level, interest |
| Expression events: smile visible, brows raised, brows lowered, eyes closed, mouth moving while silent | Personality, honesty, intelligence, "confidence", charisma, hireability |
| Voice: rate, pauses, fillers, loudness, pitch variation | "Sounds nervous", "sounds unsure" |

Basis:

- **EU AI Act Recital 18**: the mere detection of readily apparent expressions (e.g. a frown or a smile), gestures or voice characteristics is not emotion recognition unless used to identify or infer emotions. ✓ (`https://artificialintelligenceact.eu/recital/18/`)
- **EU AI Act Article 5(1)(f)**: prohibits AI systems that infer emotions in workplace and education settings, except for medical or safety reasons. ✓ (Commission guidance summarised by FPF)
- **Barrett et al. (2019)**: facial movements are not reliable, specific indicators of emotion across people and contexts (cited in your concept).

Enforcement:

1. `vision/expressions.py` returns only the five labels; a test fails on any other label.
2. `coach/guardrails.py` blocks ~150 emotion/trait terms in EN and AR in every generated string; one regeneration, then template.
3. Expression events never feed the skill profile, rubric or engine (test asserts no dependency).
4. LLM prompts never receive blendshape values or expression events.
5. UI copy review checklist for every new string.

## 3. Consent, storage and retention

| Item | Rule |
|---|---|
| Consent switches | The app's own user consent: (a) use the app, (b) camera, (c) keep encrypted raw recordings temporarily, (d) **AI wording and content checks**: send transcript text to a hosted AI model (OpenRouter) for feedback wording, follow-up questions and rubric content checks (ADR 0017); the text is stored by the model provider and may be used to improve its models. The former study-only switches (e) blind rating of recordings and (f) release of de-identified derived metrics are **not applicable: no human-participant study** (ADR 0010) |
| Encryption | Fernet key generated per installation (`MEDIA_KEY`); all media the app records is encrypted at write time; in Stage B encrypted before upload to Supabase. One exception, ADR 0016: the owner's B1 pilot clips, recorded outside the app, stay as plain WAV in the git-ignored `data/b1/` on the owner's laptop only, with their annotations; never committed, synced to OneDrive or uploaded to any cloud |
| Retention | Raw media 30 days by default (the former study-media rule is not applicable: no human-participant study); user can delete any session at once; retention job runs at start-up and daily (Stage B) |
| Derived metrics | Kept until the user deletes their profile |
| Export | One zip: JSON, Parquet, decrypted media if still retained |
| Minimisation | No name in the database (display name stays in the browser session); pseudonymous ids; no age, gender, employer or nationality; optional self-described first language for robustness slices only |
| Hosted LLM | OpenRouter free endpoints that may log or train on prompts are allowed (decision 2026-11-13, ADR 0003; reason: more capable free models at no cost). Users are told this in consent switch (d) and can decline it; declining (d) turns the feature off and the app uses templates. Protections stay: text only, personal names redacted, no media, no ids, no `user` field |
| Training platforms | Raw recordings, including the owner's own, never go to Kaggle, Colab or Hugging Face |
| AI disclosure | Every generated paragraph marked "AI-generated"; "dispute this" button stores disputes for human review |

## 4. Requirements → implementation → test

| Requirement | Implementation | Automated test |
|---|---|---|
| Consent before capture | Practice tab disabled until consent saved | UI test |
| Camera-off parity | Vision optional; skills omitted, not penalised | Journey test with camera off |
| No identity tracking | No face embedding or recognition model loaded | Import/dependency scan |
| Evidence + confidence | `Measurement` schema requires both | Schema test |
| Only allowed network hosts | Socket allow-list (OpenRouter, Supabase in Stage B) | Socket-blocking test |
| Delete really deletes | `rights.delete(session_id)` | Test: no file/row remains, including Supabase in Stage B |
| Retention | `retention.run()` | Test with clock override |
| No emotion labels | §2 | Safety tests |
| Hosted LLM payload is text-only, names redacted | Request builder | Test inspects payload (P7) |
| Hosted LLM only with consent (d) | Provider chain checks consent; templates otherwise | Test: (d) off → no OpenRouter call (P7) |

## 5. Study ethics: not applicable

There is no human-participant study (ADR 0010, 2026-11-17). Not applicable, and removed: the study information and consent sheet (EN/AR), the study-only consent switches (e) and (f), the debrief sheet, the participant checklist and the supervisor email about the ethics approval route. The app's own user consent switches (a)–(d) in §3 stay.

What remains, as recorded positions:

- **Data use:** all planned data sources, books and datasets are used for non-commercial academic research and graduation-project purposes only, under each source's terms, with each source acknowledged (`docs/data/DATA_SOURCES.md`). Research-only or no-derivatives sources are used for evaluation only; no attempt is made to identify any speaker or face; raw public data and clips are not redistributed.
- **NVIDIA free tier:** working position (owner, 2026-11-17, ADR 0010): a non-commercial academic research prototype counts as evaluation use under the NVIDIA API Trial terms (ADR 0003). If that changes, Flint becomes pinned (ADR 0003).
- **Hosted LLM privacy:** the decision of 2026-11-13 (ADR 0003) is disclosed to every user in consent switch (d).
- **E13 raters** rate published talks; nobody is recorded. **E14** uses only the owner's own recordings.

## 6. Documents to deliver

`docs/cards/PRIVACY_ETHICS.md` (data-flow diagram, §2–§4), `DATA_CARD.md` (every source in FINAL_07 with licence and use), model cards for T1/T2/T3 and pretrained models, `LIMITATIONS.md` (no multi-participant user study, n = 1 self-study, Arabic depth, simulated learners, webcam gaze accuracy, filler recall, cultural variation in delivery norms).
