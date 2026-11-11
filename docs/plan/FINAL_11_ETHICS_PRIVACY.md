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
| Consent switches | (a) take part / use the app, (b) camera, (c) keep encrypted raw recordings temporarily, (d) send transcript text to a hosted AI model (OpenRouter) for wording and follow-ups, (e) study only: blind rating of recordings, (f) study only: release of de-identified derived metrics |
| Encryption | Fernet key generated per installation (`MEDIA_KEY`); media encrypted at write time; in Stage B encrypted before upload to Supabase |
| Retention | Raw media 30 days by default; study media until blind rating ends, max 60 days; user can delete any session at once; retention job runs at start-up and daily (Stage B) |
| Derived metrics | Kept until the user deletes their profile |
| Export | One zip: JSON, Parquet, decrypted media if still retained |
| Minimisation | No name in the database (display name stays in the browser session); pseudonymous ids; no age, gender, employer or nationality; optional self-described first language for robustness slices only |
| Hosted LLM | Text only, personal names redacted, no media, no ids; free endpoints that may train on prompts are blocked in OpenRouter privacy settings during the study; feature works off |
| Training platforms | Raw study recordings never go to Kaggle, Colab or Hugging Face |
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
| Hosted LLM payload is text-only | Request builder | Test inspects payload |

## 5. Study ethics checklist

- [ ] Ask your supervisor about Al-Balqa Applied University's approval route in week 1.
- [ ] Information sheet and consent in English and Arabic, naming OpenRouter and the model, the retention period and the deletion process.
- [ ] Voluntary; no grades; no one with authority over participants recruits or sees results.
- [ ] Participants told: practice software, not an assessment of them, not used for hiring or grading.
- [ ] OpenRouter privacy setting checked before participant 1; all arms use the same pinned model.
- [ ] Debrief sheet with the system's limitations.
- [ ] TED research use confirmed with supervisor (`FINAL_07` E).

## 6. Documents to deliver

`docs/cards/PRIVACY_ETHICS.md` (data-flow diagram, §2–§4), `DATA_CARD.md` (every source in FINAL_07 with licence and use), model cards for T1/T2/T3 and pretrained models, `LIMITATIONS.md` (Arabic depth, small study, simulated learners, webcam gaze accuracy, filler recall, cultural variation in delivery norms).
