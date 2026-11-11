# FINAL 01 — Project Scope

## 1. One-sentence definition

PresenceAI is a Python web application that lets students, job seekers and other learners rehearse presentations, measures observable speaking behaviour and content with evidence and stated confidence, gives live and post-practice feedback, and picks the next exercise from each user's weakest reliably measured skill.

## 2. Users

| Group | Typical goal | What the MVP gives them |
|---|---|---|
| University students | Class presentations, graduation project defense | Primary scenario + project-defense scenario |
| Job seekers / early-career | Presenting themselves and their work clearly | Primary scenario with self-introduction and work-sample prompts; interview scenario as stub |
| General public-speaking learners | Pace, structure, fluency, eye contact | Primary scenario with open topics |

All three are served by the same product and scenario library; the user picks a goal, not a group. No user is assessed for any third party.

## 3. Scenarios

| Scenario | Status | Format | Rubric source |
|---|---|---|---|
| **S1 Prepared short presentation** | Primary, fully implemented | 2–3 min talk on the user's own topic or a prompt from the bank; optional outline upload | Adapted AAC&U VALUE Oral Communication rubric (organization, language, delivery, supporting material, central message) |
| **S2 Project defense with skeptical follow-up** | Secondary, fully implemented | 90 s project explanation, then 2 generated follow-up questions | Concept document's defense rubric + follow-up handling |
| S3 Structured interview answer (STAR) | Stub (YAML only) | 60–90 s answer | Concept document |
| S4–S10 Criticism, disagreement, bad news, feedback, delay, accountability, clarification | Stubs | — | Concept document |

## 4. Capabilities in the MVP

| Concept capability | MVP features |
|---|---|
| 1. Think clearly and express ideas | Central message detection, organization (intro/body/conclusion, transitions), PREP/STAR cue detection, relevance of follow-up answers |
| 2. Stories and audience engagement | Opening hook check, supporting material types (example, statistic, analogy, quotation), conclusion check, audience-fit language (jargon, sentence length) |
| 3. Confidence through practice | 7-level difficulty ladder, live meters, live prompts, timed tasks, follow-up questions, pre/post self-rated comfort (self-report only) |
| 4. Professional communication | S2 follow-up handling; S3–S10 as library stubs |
| 5. Measurable development | Skill profile with uncertainty, progress dashboard, attempt comparison, exports |

Delivery measures: speech rate, pauses, fillers, repetitions, loudness, pitch variation, response latency, timing; framing/visibility, head orientation, gaze zone, posture, gestures, and descriptive facial-expression events.

## 5. Contributions (the two you chose)

**C1 — Adaptive exercise recommendation.** A per-skill belief model with uncertainty, a "weakest reliable skill" policy, a difficulty ladder, engine-chosen live prompts, and a fair comparison against fixed-sequence and practice-only baselines, in simulation and in an exploratory user study.

**C2 — Multimodal communication measurement.** A transparent, evidence-linked measurement layer across speech, text and video, including a fine-tuned verbatim speech recogniser for fillers (T1), a calibrated gaze-zone classifier (T3), detector fusion with calibrated confidence, and robustness testing across accents, Arabic, lighting, glasses and camera position.

Supporting evidence (not separate contributions): confidence calibration, fairness/robustness analysis, Coach-SLM vs hosted LLM comparison.

## 6. Research questions

**Main:** Does adaptive, skill-progressive coaching improve observable presentation performance more than a fixed sequence of the same exercises?

| Sub-question | Answered by |
|---|---|
| RQ1 How accurately can the system measure the delivery and content behaviours it coaches? | E1–E7 |
| RQ2 Does fine-tuning Whisper for verbatim transcription improve filler measurement enough to change feedback? | E3 |
| RQ3 Under what measurement noise does adaptive selection beat a fixed sequence? | E9 simulation |
| RQ4 In a small exploratory study, what is the size and direction of the adaptive effect on human-rated performance? | E10 |
| RQ5 Can a fine-tuned small model write feedback as faithfully as a hosted model? | E11 |
| RQ6 Where does measurement degrade (accent, Arabic, lighting, glasses, camera position, camera-off)? | E1, E7, E10 slices |

## 7. MVP user journey

1. Open the web app, choose language (English / Arabic), read and give consent (separate switches for camera, raw-media retention, hosted AI wording).
2. Pick a goal and scenario; optional 10-second camera calibration (look at camera, look at notes).
3. Baseline: 2–3 min presentation with live meters.
4. Report: transcript with highlighted evidence, timeline (rate, pauses, fillers, gaze zones, expression events), rubric cards with confidence, one main finding, the engine's next-exercise decision and reason.
5. Targeted exercise (60–90 s) with the engine's chosen live prompt active.
6. Repeat task or follow-up question at the next difficulty level.
7. Progress dashboard; export or delete.

## 8. Explicitly out of scope

- Emotion, personality, honesty, intelligence, "confidence" or hireability inference.
- Face recognition or identity tracking; cross-user ranking; any report to an employer or instructor.
- Animated or video avatar coach (future work).
- Full Arabic semantic scoring and Arabic LLM feedback quality evaluation (future work).
- Public deployment, payments, accounts beyond the invited pilot (future work).
- Slide-to-speech alignment (dropped from MVP per your Q5 selection).
