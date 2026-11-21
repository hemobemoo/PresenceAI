# 0017 — S1 rubric: VALUE-adapted observable anchors

- Status: **accepted** by the owner on 2026-11-21, after the anchors were reviewed by the
  supervisor (rubric v0.2). **Every threshold is provisional: re-estimate after E5/E6.**
- Date: 2026-11-21 (draft v0.1 and decision on the source copy); 2026-11-21 (v0.2, accepted)
- Decider: project owner
- Plan reference: FINAL_12 §2 (P2: "VALUE-adapted rubric", study "Rubric anchors reviewed by
  supervisor"); FINAL_05 §1–§3 (skills, scores 0–1, promotion rules); FINAL_04 §3 (content
  measures, opening 20 %, closing 15 %, first 2 sentences); FINAL_07 §F (VALUE rubric, licence);
  FINAL_10 E4/E5 (content-detector F1, rubric κ)
- Config keys affected: `configs/rubric.yaml: s1_presentation.values.*` (new, all provisional)

## Context

P2 builds the text measures, the rubric and the knowledge base. The rubric came first, so the
supervisor could review its anchors before the detectors are built. The source is the AAC&U
Oral Communication VALUE rubric (CC BY-NC-SA 4.0): five dimensions (Organization, Language,
Delivery, Supporting Material, Central Message) at four levels (Capstone 4, Milestones 3 and 2,
Benchmark 1). Its wording is holistic ("compelling", "memorable", "speaker appears polished and
confident"). PresenceAI may only state observable behaviour with numbers (AGENTS.md "Wording")
and never infers confidence or other traits (AGENTS.md rule 3).

## Decision

1. **Source copy** (owner, 2026-11-21). The Binghamton copy named in FINAL_07 returned 404, as
   did the UF and Brooklyn copies. With the owner's approval the UT Tyler copy was downloaded
   into the git-ignored `data/texts/raw/`:
   `https://uttyler.edu/offices/continuous-improvement-and-accreditation/files/value-oral-communication.pdf`,
   SHA-256 `ba41c136a1d5ddd1fd0a3399ded4d8e3a10d38f9064305f27862024bcd667bde`. It shows no
   licence line; the owner decided to proceed with CC BY-NC-SA 4.0, the licence recorded in
   `DATA_SOURCES.md` (checked 2026-11-11). The copy is used for structure only: **no VALUE
   sentence is copied**; every anchor is our own wording.
2. **Files and licences.**
   - `resources/rubrics/s1_presentation.yaml`: dimensions, levels, anchor wording, operational
     definitions, measures. A derivative of the VALUE rubric, so **CC BY-NC-SA 4.0** with
     attribution (NOTICE).
   - `configs/rubric.yaml`: the threshold values, **per scenario** (S1 now, S2 later), not a
     formula of talk length (AGENTS.md rule 7).
   - The anchor wording stays in `resources/` rather than `src/presenceai/lang/`, because
     `lang/` is Apache-2.0 code and the wording is a CC BY-NC-SA derivative. This is the one
     exception to "user-facing text lives in `lang/`".
   - `schemas/rubric.py`; `rubric/loader.py` (the two files must agree: no missing and no unused
     threshold); `rubric/levels.py`; `rubric/review.py` and `scripts/render_rubric_review.py`
     write `docs/rubric/S1_ANCHORS.md` (a test fails when it is stale).
3. **Levels and scores.** VALUE's four levels are kept. A level is reached when all its
   computable conditions hold; the dimension gets the highest level reached; level 1 is the
   default. The engine's observation is **y = (level − 1) / 3**: 0, 0.33, 0.67, 1. Discrete
   levels also suit Cohen's κ in E5.
4. **A missing measure is missing, never low.** A condition whose measure cannot be computed
   (camera off, no session limit, Arabic content, unreliable punctuation) is dropped. A
   dimension with no computable measure is "not measurable", never level 1
   (`rubric/levels.py`, tested). In particular, when punctuation is unreliable only the
   sentence-length condition of Language is dropped; the dimension rests on undefined terms.
   The anchors document has an **Arabic** column: content dimensions are not measured in Arabic
   (FINAL_04 §5); Delivery uses only fillers (low confidence) and, later, gaze in Arabic.
5. **Operational definitions** (in the rubric file and the anchors document):
   - *opening* the first 20 % of the words; *closing* the last 15 %; *body* the words between.
   - *thesis sentence*: the first sentence in the opening that matches the thesis-cue lexicon.
   - *key phrase*: the **lemmatised** head nouns of the noun phrases of the thesis sentence
     (spaCy `en_core_web_sm`, stop words left out); matching is by lemma, not exact string and
     not meaning. A *key-phrase recurrence* is a body sentence with at least one key phrase.
   - *introduction cue*, *conclusion cue*, *transition*: **lexicon matches, not semantic
     understanding** of the structure.
   - *mid-sentence pause*: a pause of at least 1 s between two words of the same sentence, by
     the transcript's punctuation. English sentence positions rest on Whisper's punctuation
     (ADR 0014); not measurable for Arabic (ADR 0016).
   - *supporting-material types*: statistic, example, analogy, quotation (pattern rules; list
     numbers, years as dates and times of day are not statistics). *Counting*: one item per
     sentence and type; items = (sentence, type) pairs; types = distinct types.
   - *hook*: a question, statistic, story opening, quotation or contrast claim in the first 2
     sentences (pattern rules).
6. **Changes after the review** (v0.2, reviewed by the supervisor):
   - Organization: Capstone at ≥ 3 transitions (was 4), Milestone 3 at ≥ 2.
   - Central message: Capstone at ≥ 2 key-phrase recurrences in the body (was 3).
   - Delivery: Capstone at ≤ 4 fillers per 100 words (was 2); lower levels proposed to match:
     ≤ 6 (Milestone 3) and ≤ 10 (Milestone 2). Gaze kept at 70 / 50 / 30 %, camera on only.
   - Supporting material kept at 5 items of 3 types (Capstone) and 3 items of 2 types
     (Milestone 3), as **stretch goals** for a 2–3 minute talk.
7. **Left out.** Each dimension lists the parts of the VALUE wording that no rule can observe.
   The VALUE Delivery wording about how the speaker *appears* (confident, comfortable) is
   dropped for good: it is an inference about the person (AGENTS.md rule 3). Posture and
   gesture, vocal variety (opt-in, ADR 0013) and speech rate (own band only, P5) are not in
   the levels.
8. **LLM-judge policy (one line):** a language-model content check runs only with consent (d),
   only with a quoted transcript span, never overrides a rule-based level, and is shown as a
   coach observation, never as a numeric level. Its first targets: the central message's
   consistency with the supporting material, and whether each supporting item supports its
   claim. "Unsupported claims" (FINAL_05 §1) moves to the judge slice. Consent switch (d) is
   "AI wording and content checks" (AGENTS.md rule 2, FINAL_11 §3): with (d) off the judge does
   not run, and those measures use rules (and Laya, later) with lower confidence. The judge's
   own ADR is written in the judge slice.
9. **Laya is deferred**; its checkpoint download (about 2.3 GB) moves into
   `scripts/download_models.py` when it is added (AGENTS.md rule 2).

## Consequences

- P2-b builds the planned `text.*` measures; P2-c adds `pauses.mid_sentence_per_min` and the
  scoring. Gaze (P6) and overrun (P3) come later; until then Delivery rests on fillers and
  mid-sentence pauses.
- With base Whisper, fillers per 100 words is a lower bound (ADR 0012), so Delivery levels can
  be too high for speakers whose fillers Whisper drops.
- **Plan additions:** C2 (P2-e) plants hook and no-hook openings, so the hook detector's
  false-positive rate is known (FINAL_07 §C). P3 adds a short UI note explaining why a
  dimension shows "not measurable" (camera off, no time limit, Arabic, punctuation failure)
  (FINAL_12 §2).
- P2-b checks the wordfreq data licence and adds attribution to NOTICE if needed.
- The Arabic version of the anchor wording follows this ADR (owner, 2026-11-21).
- S2 (project defense) reuses these five dimensions scaled to a 90-second explanation, plus a
  `followup_handling` criterion (relevance, response latency, acknowledged limitations;
  FINAL_05 §1).

### Open items

1. **Engine promotion rules and these scores (for P5; the engine is not changed now).** The
   engine promotes after two observations with y_adj ≥ 0.75 and demotes after two with
   y_adj ≤ 0.40 (`configs/engine.yaml: difficulty`, FINAL_05 §3). The rubric gives only 0,
   0.33, 0.67 and 1. With y_adj = y + β·(d − d_ref), a level-3 score (0.67) reaches 0.75 only
   when β·(d − d_ref) ≥ 0.083, that is, never at or below the reference difficulty. So a
   speaker at level 3 can never move up there, and level 2 (0.33) always counts toward
   demotion. The promotion and demotion values, or the score mapping, must be revisited in P5.
2. **Length-aware thresholds** (e.g. transitions or supporting items per minute instead of per
   talk): test on C2 whether fixed per-scenario counts favour longer talks.
3. **Re-estimate every threshold after E5/E6** (κ against human labels, calibration).
