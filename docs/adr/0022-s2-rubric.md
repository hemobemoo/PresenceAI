# 0022 — S2 project-defense rubric: S1 scaled to 90 s, plus follow-up handling (P2, S2)

- Status: **accepted** (2026-11-23), **reviewed by the supervisor**: approved subject to five
  conditions, recorded under "Supervisor review" below. Rubric v0.1 accepted; thresholds
  provisional, as for S1.
- Date: 2026-11-23
- Decider: project owner; anchors reviewed by the supervisor
- Plan reference: FINAL_01 §3 (S2: 90 s explanation, then 2 follow-up questions), FINAL_05 §1
  (`followup_handling`), FINAL_04 §2–§3 (response latency, relevance, limitations), ADR 0017
  (S1 rubric), ADR 0019 (scoring), ADR 0020 (embedding model)
- Config keys affected: `configs/rubric.yaml: s2_project_defense` (new block, all provisional);
  `scenarios/s2_project_defense.yaml` (new)

## Decision

1. **Rubric `s2_project_defense` v0.1** (`resources/rubrics/s2_project_defense.yaml`, CC BY-NC-SA
   4.0, VALUE-adapted like S1). It reuses S1's five dimensions and the anchor wording the
   supervisor reviewed (ADR 0017), and adds a sixth dimension, `followup_handling`, which is
   not part of the VALUE rubric (FINAL_05 §1). Same level rule as S1 (ADR 0019): the highest
   level whose computable conditions all hold; a missing measure is never low.
2. **Rubric source** (owner, 2026-11-23): FINAL_01 named "the concept document's defense
   rubric" for S2; that document is not in the repository. S2 uses S1's five dimensions plus
   follow-up handling instead, and the FINAL_01 S2 row is updated.
3. **Scaling to 90 s** (owner, 2026-11-23). **The 90 s counts are S1's counts scaled by ratio
   (about 90/150 = 0.6, rounded, at least 1); they are not derived from data.** Shares (opening
   20 %, closing 15 %), the hook window (2 sentences), per-sentence values and rates (fillers per
   100 words, mid-sentence pauses per minute, gaze share) are S1's values unchanged.

   | Threshold | S1 | S2 |
   |---|---|---|
   | `sup_items_high` / `sup_items_mid` / `sup_items_min` | 5 / 3 / 1 | 3 / 2 / 1 |
   | `sup_types_high` / `sup_types_mid` | 3 / 2 | 2 / 2 |
   | `cm_recurrences_min` | 2 | 1 |
   | `org_transitions_high` / `org_transitions_mid` | 3 / 2 | 2 / 1 |
   | `lang_undefined_max_2` | 3 | 2 |
   | `del_overrun_4` / `_3` / `_2` (s over the limit) | 0 / 15 / 30 | 0 / 10 / 20 |

   `sup_types_mid` stays 2 (owner's table) so level 3 still needs two kinds of support.
   All values are provisional: re-estimate after E5/E6.
4. **Follow-up handling** (owner, 2026-11-23). Measures, all planned in S2-b:
   - `followup.relevance_min`: cosine similarity between each question and its answer
     transcript, from the knowledge-base embedding model (multilingual-e5-small, ADR 0020).
   - `followup.latency_max_s`: seconds from the end of the question to the first speech onset.
   - `followup.limitation_acknowledged`: an explicit limitation phrase in at least one answer.

   **The weaker answer decides**: the lowest relevance and the longest latency over the 2
   answers. Levels:

   | Level | Conditions |
   |---|---|
   | 4 | relevance_min ≥ `fu_relevance_high`, latency_max ≤ 2 s, a limitation acknowledged |
   | 3 | relevance_min ≥ `fu_relevance_mid`, latency_max ≤ 4 s |
   | 2 | relevance_min ≥ `fu_relevance_low` (both answers, since the weaker decides) |
   | 1 | default |

   The latency values (2 s, 4 s) are the owner's, with no data. **The ≤ 2 s Capstone latency
   includes reading time (item 8) and is likely to be relaxed after the first real S2 pilots.**
5. **Relevance calibration** (owner, 2026-11-23). The owner writes 20 question–answer pairs:
   10 on-topic and 10 off-topic, about 5 of the off-topic ones **evasive** (they reuse the
   question's words without answering). `fu_relevance_high/mid/low` are set between the two
   score distributions in S2-b and recorded here as provisional. **These pairs are calibration
   data only and are kept out of any test set.** Until then the three values in
   `configs/rubric.yaml` were **placeholders, not used for scoring** (0.85 / 0.80 / 0.75);
   set to one cut of 0.82 on 2026-11-24 (see "Relevance calibration").
   A similarity of words is not an understanding of whether the question was answered; the
   evasive pairs show how far an evasive answer can score.
6. **Arabic.** Relevance is **not measured for Arabic** until there are Arabic calibration
   pairs (the model is multilingual, but its thresholds would rest on English pairs only). The
   limitation lexicon is English only. Latency is language-independent.
7. **Limitation cues** (owner, 2026-11-23): explicit limitation phrases only ("one limitation",
   "we did not test", "future work", …). The list **must not overlap hedging**: phrases such as
   "I'm not sure", "maybe" or "I think" are not limitation cues. Hedging is not measured
   (ADR 0018), so there is no hedge lexicon yet. **The test that no limitation cue is a hedging
   phrase is written in S2-b, before the follow-up measures go live** (supervisor's condition).
8. **Response latency replaces the FINAL_04 event-log method** (owner, 2026-11-23). Each answer
   recording **starts automatically when the question ends** (no manual start), so latency is
   the first speech onset in that answer recording (voice activity detection), with no event
   log. **If the question is shown as text, the reading time is included** in the latency. A
   latency that cannot be computed (no speech onset found, no answer recorded) drops its
   condition; it is never scored low. FINAL_04 §2 is updated.
9. **Scenario file** `scenarios/s2_project_defense.yaml`: 90 s, 2 follow-ups, answer recording
   `auto_start`. Questions are generated in P7 (consent switch (d)); until then they are given
   as input to `presenceai assess` (S2-c).

## Consequences

- The anchors document is generated from the rubric (`scripts/render_rubric_review.py --rubric
  s2_project_defense`), so the supervisor reviews exactly what the app uses.
- `MeasureSpec.requires` gains `followups`: no follow-up answers recorded means the
  follow-up measures are missing, never low, and the dimension is "not measurable".
- S2 inherits S1's limitations (lexicon matches, lower-bound fillers, Whisper punctuation) and
  ADR 0017 open item 2 (length-aware thresholds): the ratio scaling is a stopgap for that item.

## S2-b: follow-up measures (2026-11-23)

Built and tested, not yet scored (S2-c wires them into the assessment, so the rubric still
lists them as planned in S2-c):

- **Limitation cues:** `lang/en/text_cues.yaml: limitation`, 29 explicit phrases ("one
  limitation", "we did not test", "out of scope", "future work", …; contractions listed in both
  forms). **The no-hedge test is written** (`tests/unit/test_followup.py`): no cue is or contains
  a common hedging phrase, and hedged answers ("I'm not sure, maybe…") are not matched.
- **Latency** (`speech/latency.py`): the first speech onset in the answer recording; no speech
  region gives no latency (missing, never low).
- **Relevance** (`text/followup.py`): cosine similarity, question embedded as an e5 "query" and
  the answer as a "passage". **Empty text gives no relevance (missing)**: the text alone
  cannot tell silence from a failed transcription (owner, 2026-11-23; see "Empty answers").
- **Weaker answer** (`text/followup.py: summarise`): lowest relevance, longest latency; an
  answer whose value is missing is left out; a limitation counts if any answer has one.
- **Relevance does not score until calibrated:** S2-c emits `followup.relevance_min` only after
  `fu_relevance_*` are set from the owner's 20 pairs; until then it is reported as information
  and its conditions are dropped ("based on N of M measures", item 2 below).

## Empty answers: "no answer given" is decided from the audio (owner, 2026-11-23)

`pipeline/followup.py: measure_answer` gives each answer a status (`schemas/followup.py`):

1. **No answer given** (`no_answer`): the recording passed its checks, **the full answer window
   elapsed and the voice detector found no speech**. This is a measured non-answer: relevance
   0.0, and the latency counts as the whole window, which is longer than every latency limit,
   so it fails every latency condition and is not missing. The finding uses the answer window
   as its time evidence (S2-c). The dimension scores **level 1**.
2. **Speech found but the transcript is empty** (`speech_untranscribed`): relevance is missing;
   the latency is known (the speech onset).
3. **The recording failed** (`recording_failed`): relevance and latency are missing.

   In cases 2 and 3 the partial-evidence wording applies (ADR 0019: "based on N of M
   measures"). A recording that stopped before the window ended with no speech found
   (`incomplete`) is also missing, never a non-answer.
4. **In the app (P3), before a non-answer is scored, the user is offered a retry** ("We didn't
   hear an answer").

Tests: `tests/unit/test_followup.py` covers all three cases, the early stop and a full answer.

## Evasive answers (owner, 2026-11-23)

If the owner's calibration pairs show evasive answers (reusing the question's words without
answering) overlapping the real answers in similarity, **relevance only separates off-topic
answers**, and this ADR records that limit. No word lists are added to catch evasion: it is a
matter of meaning, and goes to the language-model judge slice (P7, consent switch (d)).

## S2-c: follow-ups in the assessment (2026-11-24)

`presenceai assess --scenario s2_project_defense --followups followups.json` scores S2. The
file lists each follow-up's `question`, answer recording `file` (relative to the JSON file) and
answer `window_s`; until P7 generates questions they are given as input (item 9). Each answer
recording gets its own speech report; a file that cannot be read is a failed recording.

- **Evidence names its answer:** `Evidence.answer` is the follow-up index; its quote and times
  are into that answer's own recording, and verification checks them against that answer's
  transcript (`rubric/evidence.py`). Latency evidence is the time from the end of the question
  to the first speech; relevance evidence is the whole answer of the weaker answer; limitation
  evidence quotes the cue's words (or, with no cue, every measured answer).
- **Non-answer finding:** an answer with no speech over its whole window adds "Answer N: no
  speech in the W s answer window." to the finding, with the window (0–W s of that answer) as
  time evidence.
- **Relevance is reported, not scored, until calibrated** (until 2026-11-24): `configs/rubric.yaml`
  listed the `fu_relevance_*` keys under `uncalibrated`; a measure with a condition on one of them keeps its
  value in the not-measurable reason ("measured 0.87, not scored until its thresholds are
  calibrated") and its conditions are dropped (partial evidence, supervisor's condition 2).
- **No follow-ups recorded:** the follow-up measures are missing ("no follow-up answer
  recorded") and the dimension is not measurable, never level 1.
- **Confidence:** latency medium (voice-activity onset, not yet checked on real answers),
  limitation medium (lexicon match), relevance low (similarity of words; 20 calibration pairs).
- Arabic answers: latency only (item 6); relevance and limitation give the reason "not
  measured for ar answers".

## Relevance calibration: the owner's 20 pairs (2026-11-24)

The owner wrote 20 question-answer pairs about PresenceAI (`data/s2_relevance/pairs.txt`,
git-ignored, SHA-256 `80b97b79…a733b`): 10 good answers, 5 evasive and 5 off-topic, each bad
answer paired with a good answer's question. Scored with the app's relevance function
(`experiments/s2_relevance/score_pairs.py`; ids, labels and scores only in
`experiments/results/s2_relevance/scores.json`):

| Label | n | min | median | max |
|---|---|---|---|---|
| good | 10 | 0.841 | 0.874 | 0.933 |
| evasive | 5 | 0.908 | 0.934 | 0.943 |
| off-topic | 5 | 0.776 | 0.790 | 0.842 |

- **Evasive answers score above every good answer** (lowest evasive 0.908; only one good
  answer, 0.933, is higher). The fallback above applies: **relevance only separates off-topic
  answers**; evasion goes to P7.
- **Off-topic and good touch:** the highest off-topic answer (0.842) is above the lowest good
  answer (0.841). No value separates all 15 of them; the gap is narrow and rests on 15 answers
  by one writer.

**Decision (owner, 2026-11-24): `fu_relevance_high = mid = low = 0.82`, one on-topic cut,
provisional.** `uncalibrated` is now empty, so relevance is scored.

- **Relevance only flags clearly off-topic answers.** At 0.82 the calibration pairs give 0 of
  10 good answers flagged (lowest good 0.841, margin 0.02), **3 of 5 off-topic caught**
  (0.776, 0.785, 0.790; the two borderline ones, 0.835 and 0.842, pass) and **0 of 5 evasive
  caught**.
- **Why not 0.84:** it clears good answer #10 by only 0.0005, and real answers are spoken and
  transcribed, which likely lowers similarity, while the weaker answer decides.
- **All three thresholds are equal, so relevance only separates level 1 from level 2.
  Levels 3 and 4 are decided by latency and the limitation cue.**
- **The finding describes what was measured:** "Answer N shares little wording with the
  question (similarity 0.78, below 0.82)". It never says the speaker did not answer or was
  off-topic. Relevance keeps low confidence.
- **The pairs were written, not spoken.** The cut is re-checked on the first real spoken S2
  answers (open item 2, together with the 2 s latency caveat).

## P3-e: S2 in the app (owner, 2026-11-25)

Config keys: `configs/default.yaml: app.answer_window_s` (60, **provisional**),
`app.answer_preroll_s` (1.0).

1. **One microphone stream for the whole session** (`live/s2_session.py`, `live/s2_audio.py`,
   `ui/pages/practice_s2.py`). The user presses Record once: explanation, then question 1,
   then question 2. "Next: question 1" ends the explanation and shows question 1; "Done
   answering" ends an answer early; otherwise a window ends after `app.answer_window_s` (60 s)
   of audio. The next question appears as soon as a window ends, and its window opens then.
2. **Marks are received-audio sample positions, not clock times** (owner). A button press
   places its mark at the number of samples the server has received; window lengths and the
   explanation hard stop are counted in samples. **Expected timing error:** the audio still in
   flight when the press reaches the server (browser capture, WebRTC jitter buffer, network on
   loopback; spike S1 measured under 300 ms from event to meter, ADR 0002) is not yet counted,
   so a mark sits **about 0.1–0.3 s before** the moment the question appears on screen. The
   measured latency is therefore **too long by about 0.1–0.3 s** (and by the page's render time,
   a few ms), a bias against the user that matters next to the 2 s level-4 limit. Window ends
   and the hard stop, counted in samples, have no such error. Check it on the owner's first
   real S2 runs (item 2 below).
3. **The explanation is not cut at 90 s** (owner): "Next: question 1" ends it, with a hard stop
   at 90 s + `app.recording_extra_s` (150 s), so overrun stays measurable (ADR 0019); at the
   hard stop question 1 is shown and the report says so. The stream's time limit covers the
   whole session, retries included: 90 + 60 (explanation) + 2 questions × 2 attempts × 60 s +
   60 s for reading prompts and pressing buttons = **450 s**.
4. **Speech already going when a window opens is not an onset** (owner). The voice detector
   looks back `app.answer_preroll_s` (1.0 s) before the mark; a speech region that began before
   the mark and continues after it is running speech, and the latency is the first speech start
   at or after its end (`speech/latency.py`). Tested: answer 1 running 2 s into question 2's
   window gives question 2 the onset of the new speech, not 0 s. The running words stay in that
   answer's transcript (they are part of its audio).
5. **Retry before a non-answer is scored** (item 4 of "Empty answers", decided 2026-11-24): a
   window that ran its full length is checked by voice activity detection only; with no speech
   the page says "We didn't hear an answer to question N" with **Try again** (one retry per
   question; the silent take is discarded and never stored) or **Keep (no answer)**. A second
   silent window is kept as no answer. "Done answering" before the window ends is never a
   non-answer (incomplete: missing, never low).
   - **Running speech is not an answer either (2026-11-25, found by the browser journey test,
     ADR 0024 "Correction").** The silence check uses the same rule as the latency (item 4):
     the window is checked with the 1 s look-back, and speech that was already running at the
     mark does not count. Without this, the 0.1–0.3 s of audio still in flight when the question
     appears made a silent answer look answered, so the retry was never offered. Scoring matches:
     a full window whose only speech is running speech is a measured non-answer (relevance 0,
     latency = window, `pipeline/followup.py: measure_answer`). Tested in
     `tests/unit/test_latency_running.py`.
6. **Questions:** a bank of 6 per language (`lang/{en,ar}/followup_questions.yaml`, Arabic a
   draft), 2 picked at random per session, or the user's own (either box). **No bank question
   asks for limitations directly** (tested against the limitation cues and a list of
   limitation words), since that would prompt the cue the rubric measures. The 0.82 relevance
   cut was calibrated on project-specific questions (item 5 of the open items); the bank is
   generic.
   - **Owner review (2026-11-25): bank questions must not invite scope or future-work answers
     either**, not only avoid limitation words. "Out of scope" and "future work" are limitation
     cues, so a question such as "What did you decide not to build, and why?" (the drafted Q6)
     would hand whoever draws it the level-4 limitation condition almost for free. Q6 is now
     "Which result surprised you the most, and why?". The test also rejects scope, future-work
     and "not build" wording in both languages. Arabic Q2, Q3, Q5 and Q6 were reworded by the
     owner; Q1 and Q4 kept.
   - **Relevance with the generic bank:** in the browser check an unrelated talk scored 0.77
     against the 0.82 cut. If the owner's real answers to bank questions come out near the cut,
     relevance becomes information-only for bank questions (reported, not scored, as before
     calibration) until it is recalibrated on bank questions.
7. **Analysis:** the explanation and each kept answer are cut from the stream, encrypted
   (`explanation.wav`, `answer_1.wav`, `answer_2.wav`) and analysed in five steps (models,
   explanation, answers, content, save). e5 is loaded only for English S2 analyses and released
   after it (ADR 0024). An answer that was not recorded (the user stopped early) is missing,
   never low. **S2 uploads** carry no answers: an uploaded file is analysed as the explanation,
   and follow-up handling says "no follow-up answer recorded".

## After the owner's first real S2 run (owner, 2026-11-25)

**The run (numbers only):** 197.8 s of audio, analysed in 90.2 s; explanation 92.5 s.
Answer 1: latency 6.53 s, relevance 0.851; answer 2: latency 4.93 s, relevance 0.876; no
limitation stated; follow-up handling level 2 under the earlier rules. This is the first
evidence, **not the basis** for the values below.

1. **Relevance from bank questions is information-only** until relevance is recalibrated on
   generic questions; **questions the user types keep scoring.** Reason: the gap is narrow
   (the owner's answers 0.851 and 0.876 against unrelated talks at 0.77–0.81 in the browser
   checks), and on generic questions evasive answers are likely to clear 0.82, as they did on
   project-specific ones ("Relevance calibration": evasive 0.908–0.943). A bank answer's
   similarity stays in the report with its value ("measured 0.851, 0.876 on questions from the
   bank; not scored until relevance is recalibrated on generic questions"); it is left out of
   `followup.relevance_min` and of the low-relevance note. A measured non-answer still scores 0
   (it is decided from the audio, not the similarity). Calibration pairs for the bank are a
   later task for the owner.
2. **A reading allowance replaces raw latency** (`configs/rubric.yaml: s2_project_defense.
   fu_reading_wps: 4`, provisional, the same for Arabic until there is data). Each answer's
   allowance is its question's words ÷ 4 words per second; `followup.latency_max_s` is now the
   longest **thinking time**, latency minus allowance (never below 0). **Level 4: latency ≤
   allowance + 2 s; level 3: latency ≤ allowance + 4 s**: the 2 s and 4 s (`fu_latency_4`,
   `fu_latency_3`) stay as thinking time. Reason: the window opens when the question appears,
   so the raw latency includes reading it, and a 2 s limit cannot be met once reading time is
   included (a 15-word question takes about 3.75 s to read). The level-4 and level-3 anchor
   wording now names the allowance (English and Arabic; the review document is regenerated).
   The evidence time span stays the raw latency in that answer's recording. Tested: a short
   and a long question with the same latency give different results.
3. **Consequence for the owner to decide:** level 2's only condition is relevance, so when
   both questions come from the bank (relevance not scored) **level 2 cannot be reached**:
   follow-up handling is then 1, 3 or 4. A test pins this behaviour; whether level 2 should get
   a condition that is still scored (for example a latency or answered-both condition) is an
   open question (open item 6).

## Level 2: every window answered (owner, 2026-11-25; open item 6)

**Decision.** When relevance is information-only, level 2 means both answers contain speech:
the status is "answered" in both windows. This is measurable, and it keeps ADR 0019's rule that a
missing measure is never low. A real non-answer is still level 1.

1. **New measure `followup.answered_all`** (`pipeline/followup.py`), decided from the audio and
   measured in both languages:
   - **1** when every answer's status is "answered", with each answer as evidence;
   - **0** when a window had no new speech (`no_answer`, a measured non-answer), with that
     window as evidence;
   - **missing** for an early stop, untranscribed speech or a failed recording. These cannot
     tell, so the measure is never read as low.
2. **Level 2's conditions** are now `followup.relevance_min >= fu_relevance_low` **and**
   `followup.answered_all` present. The level engine drops a missing measure, so:
   - **Bank questions** (relevance missing, information-only): level 2 = every window
     answered.
   - **Questions the user types** (relevance scored): **no change in level**. A non-answer
     already scores relevance 0, so the new condition holds whenever relevance holds.
   - **Arabic** (relevance not measured, item 6 of the Decision): **the same rule applies
     (owner's decision, 2026-11-25).** The principle is the same as for bank questions: relevance
     is not measured, so level 2 must not depend on it alone (ADR 0019: missing is never low).
3. **Consequence for "N of M"**: follow-up handling now has 4 measures, not 3. A typed-question
   English run with everything measured shows "4 of 4"; a bank run shows "3 of 4" (relevance
   reported, not scored). Arabic shows "2 of 4" and is still capped at 3 (ADR 0019 cap, share
   ≤ 0.5), as it was at "1 of 3".
4. **Anchor wording** (English and Arabic; Arabic is a draft for the owner) for levels 2 and 1
   names the answered condition; `docs/rubric/S2_ANCHORS.md` is regenerated.
5. **Tests** (`tests/unit/test_followup_thresholds.py`):
   - bank questions, both answered and slow → level 2;
   - a silent bank answer → level 1;
   - an early stop → measure missing, not low;
   - typed questions answered off-topic → still level 1.

6. **Level 4 needs a measured limitation cue (owner, 2026-11-25).** The limitation lexicon is
   English-only, so in Arabic level 4 reduced to the tighter latency. Rubric conditions now
   have a `required` flag (`schemas/rubric.py: Condition.required`, `rubric/levels.py`): when a
   required measure is missing, its level is not awarded (unavailable, not failed), and lower
   levels are still reachable. `followup.limitation_acknowledged` is required for level 4.
   - The report says "Level 4 not available in Arabic yet (limitation cues not measured)"; the
     Arabic wording is a draft.
   - **Arabic follow-up handling tops out at level 3** until Arabic limitation cues exist (open
     item 8). It is now reached as level 3, not "capped from 4" by the ADR 0019 cap.
   - The same holds in English when no answer could be checked for a cue (early stops,
     untranscribed speech): "Level 4 not available (limitation cues not measured)".
   - Tested: the same fast Arabic answers give level 3, not 4 (`tests/unit/
     test_followup_assess.py`). The anchors document marks the condition "(required)" and its
     Arabic column says "not available in Arabic yet".

**The owner's first real run, re-scored** (latencies 6.53 s and 4.93 s, both answered, no
limitation stated, bank relevance not scored). The first question was not recorded, so every
ordered pair of the 6 bank questions was scored with the real rubric. Answer 1's thinking time
decides in every case:

| First question | Thinking time (answer 1) | Before (ADR 0022 "After the first run") | Now |
|---|---|---|---|
| Q1 (10 words) | 4.03 s | 1 | **2** |
| Q2 (9 words) | 4.28 s | 1 | **2** |
| Q6 (8 words) | 4.53 s | 1 | **2** |
| Q3 (14 words) | 3.03 s | 3 | 3 |
| Q4 (15 words) | 2.78–2.93 s | 3 | 3 |
| Q5 (12 words) | 3.53 s | 3 | 3 |

With Q1 first, the run misses level 3's 4 s by 0.03 s. That is inside the 0.1–0.3 s timing bias
against the user ("P3-e" item 2), so open item 2's check of the mark timing matters for this
case. **Owner, 2026-11-25: no change; the bias stays an open item (item 2) until there are more
real runs.**

## Supervisor review (2026-11-23)

Approved, subject to five conditions, all met in this ADR's PR or bound to S2-b:

1. The S2 row is in `resources/rubrics/README.md`, whose render note names `S2_ANCHORS.md`.
2. **Partial evidence applies to `followup_handling` unchanged** (ADR 0019 item 3, no new
   policy): its dimension result and finding state how many of its measures were computed
   ("based on N of M measures"). This includes **Arabic, where follow-up handling rests on
   latency alone** ("based on 1 of 3 measures"), and **early English runs before the relevance
   thresholds are calibrated**, when relevance is not used for scoring.
3. The ≤ 2 s Capstone latency includes reading time and is likely to be relaxed after the
   first real S2 pilots (item 4).
4. The no-hedge test for limitation cues is written in S2-b, before the measures go live
   (item 7).
5. **The Arabic wording stays a draft** pending a native-speaker review (ADR 0016 open item 4),
   with the same placeholder discipline as S1: `text_ar` uses exactly the placeholders of
   `text`, and the loader rejects any difference.

The owner's two additions were approved: `lang_undefined_max_2` scaled from 3 to 2, and open
item 1 below.

### Open items

1. ~~**Arabic follow-up handling rests on latency alone.**~~ **Closed 2026-11-24 (P3-a, ADR 0019
   amendment):** a level based on at most half of its measures is capped at 3, so Arabic
   follow-up handling on latency alone (1 of 3) reaches at most level 3, and its finding says
   "based on 1 of 3 measures" and why it is limited.
2. **Re-check on the first real spoken S2 answers:** the ≤ 2 s latency, which includes
   reading time (supervisor's condition 3, item 4), and the 0.82 relevance cut, which rests on
   20 written pairs by one writer ("Relevance calibration"). Also check the mark timing error (about
   0.1–0.3 s, "P3-e" item 2) and the 60 s answer window.
3. ~~Supervisor review~~: done 2026-11-23 (above).
4. ~~**P3: retry before a non-answer is scored**~~ **Built in P3-e (2026-11-25, "P3-e"
   above):** one retry per question; the silent take is discarded before anything is stored; a
   second silence is scored as no answer.
5. **The 0.82 relevance cut was calibrated on project-specific questions** (owner, 2026-11-24):
   the 20 pairs ask about PresenceAI itself. In the app (P3-e, until P7 generates questions) the
   questions come from a bank of about 6 generic skeptical questions per language, 2 picked at
   random, or the user's own; generic questions may share fewer words with a good answer, so
   the cut is re-checked on the first real spoken S2 answers together with item 2. No question
   in the bank may ask for limitations directly, since that would prompt the limitation cue.
6. ~~**Level 2 with bank questions**~~ **Decided (owner, 2026-11-25), "Level 2: every window
   answered" below.**
7. **Calibration pairs for the bank questions** (owner, later): until then bank relevance is
   information-only.
8. **Arabic limitation cues** (owner, 2026-11-25): Arabic follow-up handling tops out at level 3
   until an Arabic limitation lexicon exists (with the same rule as the English one: explicit
   phrases only, no hedging; "Evasive answers"), because level 4 requires a measured limitation
   cue ("Level 2: every window answered", item 6).
