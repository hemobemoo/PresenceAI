# FINAL 04 — Multimodal Measurement Specification

Every measure is a `Measurement(value, unit, confidence, method, evidence[], language, live: bool)`. Evidence is a transcript span, a time span, or both. A measure that cannot be computed reliably is reported as "not measurable in this recording", never as a low score.

## 1. Pre-recording quality check (not scored)

| Check | Method | Action |
|---|---|---|
| Microphone level and noise floor | 3 s RMS sample, SNR estimate | Ask to move closer or reduce noise |
| Lighting | Mean and spread of face-region luminance | Suggest facing a light source; mark vision confidence lower |
| Framing | Face box position and size | Show framing guide |
| Camera calibration (optional, 10 s) | User looks at camera dot, then at notes area, then neutral face | Feeds gaze-zone model (T3) and per-user expression baselines |

## 2. Speech and audio

| Measure | Live? | Definition | Notes |
|---|---|---|---|
| Speech rate (wpm) | Post | Words ÷ speaking time; 10 s sliding windows | Targets are relative to the user's own baseline band, never one absolute norm |
| Pace estimate (syll/s) | **Live** | Syllable nuclei from intensity peaks in voiced regions, 5 s window | No ASR needed; works for Arabic |
| Articulation rate | Post | Words ÷ voiced time | — |
| Pauses | Live (timer) + Post | Gaps ≥ 0.25 / 1.0 / 2.0 s from VAD; count, mean, p90, longest, position (mid-sentence vs boundary) | Boundary pauses are not penalised |
| Fillers per 100 words | Post | Fusion of (a) lexicon on T1 verbatim transcript, (b) acoustic candidates (voiced, no aligned word, low pitch variance); confidence high if both agree | Reported as a range when confidence is low |
| Repetitions / false starts | Post | Repeated 1–3-grams within 2 s; T1 transcript marks | — |
| Loudness | **Live** + Post | RMS in 1 s windows; coefficient of variation; clipping | — |
| Pitch variation | Post | pYIN F0 → semitone SD over voiced frames | Shown as information; no "monotone" verdict without user goal |
| Response latency (S2) | Post | First voice onset after the follow-up question ends | From event log |
| Timing | **Live** + Post | Elapsed vs limit, overrun seconds | — |
| ASR confidence | Post | Mean word probability, no-speech probability | Lowers confidence of everything that depends on the transcript |

## 3. Content (from the transcript)

| Measure | Rule part | Model part |
|---|---|---|
| Central message stated | Thesis-cue lexicon in first 20% ("today I will", "my main point") + repeated key noun phrases | Laya `noul` + LLM judge with quote |
| Organization | Intro / body / conclusion position cues; transition lexicon ("first", "next", "in conclusion") | Laya `choice` on sentence role |
| Opening hook | First 2 sentences: question, statistic, story, quotation, surprising claim patterns | LLM judge with quote |
| Supporting material | spaCy matchers: numbers/percentages (statistic), "for example" (example), "like a"/"imagine" (analogy), quotation marks/"said" (quotation) | LLM judge confirms type |
| Conclusion | Closing cues in last 15%; restatement similarity to central message | — |
| Language / audience fit | Low-frequency terms without definitions (wordfreq Zipf), mean sentence length, readability | — |
| Hedging | Hedge lexicon rate | Count only, never a trait |
| Relevance (S2 follow-up) | Embedding similarity question↔answer | Laya `noul` |
| Limitations acknowledged (S2) | Limitation cue lexicon | Laya `noul` |

LLM judge claims are kept only if the quoted span is found in the transcript (rapidfuzz ≥ 85).

## 4. Vision (all optional; camera-off is fully supported)

### 4.1 Geometry

| Measure | Live? | Definition |
|---|---|---|
| Visibility | **Live** + Post | Share of frames with face and shoulders detected |
| Framing | **Live** + Post | Face centre offset and face height as fraction of frame; head-cut-off flag |
| Head orientation | **Live** + Post | Yaw, pitch, roll from the facial transformation matrix; share of time within ±20° |
| Posture | Post | Shoulder-line tilt; torso lean; vertical nose–shoulder distance relative to calibration |
| Gestures | Post | Wrist-speed peaks per minute while hands visible; share of time hands visible; gesture–speech overlap |
| Movement stability | Post | Shoulder-midpoint velocity variance, normalised by shoulder width |

### 4.2 Gaze zone (labels how the user is looking)

| Label | Meaning |
|---|---|
| `at_camera` | Looking toward the audience: the lens or the screen area near it (slides, self-view) |
| `down_notes` | Looking down at notes or the keyboard |
| `away` | Looking elsewhere |
| `unknown` | No face found, or eyes closed / not reliably visible (glare, occlusion) |

MVP label set after spike S5 (ADR 0004, planned fallback): the lens and the screen area could not be separated reliably from webcam features (4-zone macro-F1 0.70; 3-zone 0.93), so they form one `at_camera` label. A separate `at_screen` label is **future work**.

Features: iris centre relative to eye corners (both eyes), eyelid aperture, head yaw/pitch. Classifier: T3 (`FINAL_06` §4), personalised by the optional 10 s calibration. Reported: share of time per zone, longest `away`/`down_notes` stretch, gaze at camera during the opening and conclusion. Accuracy per zone is measured in E7 and shown on the About page.

### 4.3 Facial expression events (descriptive, never emotions)

From MediaPipe blendshapes, relative to the user's own neutral baseline from calibration.

| Event label shown to the user | Blendshapes used | Threshold idea |
|---|---|---|
| Smile visible | `mouthSmileLeft`, `mouthSmileRight` | Mean above baseline + δ for ≥ 0.5 s |
| Brows raised | `browInnerUp`, `browOuterUpLeft/Right` | Same pattern |
| Brows lowered | `browDownLeft/Right` | Same pattern |
| Eyes closed (longer than a blink) | `eyeBlinkLeft/Right` | Both high ≥ 0.4 s |
| Mouth moving while silent | `jawOpen` variance while VAD = silence | — |

Rules enforced in code and tests:

1. Labels are the five above. No emotion, mood, attitude or intention words exist in the label set, the UI strings or the LLM prompts (`FINAL_11` §2).
2. Expression events are **informational**: shown on the timeline with counts and durations; they do not feed the skill profile or any score.
3. A user may opt in to one expression-related practice goal (e.g. "smile visible during my opening") as a self-chosen target; the system then reports whether the event occurred, not whether it was "good".

## 5. Language packs

| Item | English (full) | Arabic (MVP) |
|---|---|---|
| ASR | faster-whisper (base or T1) | faster-whisper `language="ar"`; T1 Arabic is future work |
| Rate, pauses, loudness, pitch, live pace, timing | ✓ | ✓ (rate in words/min and syllables/s) |
| Fillers | Lexicon + acoustic + T1 | Lexicon (e.g. "يعني", "امم", "اه", "يعني يعني", "طيب") + acoustic; lower confidence |
| Repetitions | ✓ | ✓ |
| Content measures (§3) | ✓ | Not scored; transcript shown; future work |
| Vision (§4) | ✓ | ✓ (language-independent) |
| Feedback text | Templates + LLM wording | Arabic templates; LLM wording off by default |
| Arabic code-switching (ADR 0003) | — | In the coach's own wording (templates, LLM, Coach-SLM), common English technical/academic terms that Arabic-speaking students use (e.g. *slides*, *feedback*, *pace*) stay in English script; never transliterated into Arabic letters; the Arabic equivalent may follow on first use; everything else is Modern Standard Arabic. **Quotes are excluded**: quoted spans match the transcript exactly, and `coach.guardrails.latin_outside_transcript` flags Latin-script text in an Arabic quote that is not in the transcript. Rule text: `lang/ar/coach_prompt.yaml` |
| UI | LTR | RTL strings and text components |

Language detection on the first 30 s; mismatch with the chosen language triggers a notice, not an error.

## 6. Confidence for every measure

`confidence = f(input quality, detector agreement, ASR word probability, landmark visibility, duration)` mapped to {high, medium, low}. The engine weights observations by confidence (`FINAL_05` §2); the report shows the confidence next to every number; E6 checks whether "high" findings are right at the stated rate.

## 7. Reference ranges from TED talks

E12 computes distributions of rate, pauses, gaze-at-camera share and gesture rate on a small curated TED set. They are shown only as "typical range in a sample of published talks", never as a target, so one presentation culture is not treated as the standard.
