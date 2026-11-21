# 0017 — S1 rubric: VALUE-adapted observable anchors (draft)

- Status: **proposed** (P2-a, 2026-11-21). The structure and the scoring rule below were decided
  by the owner on 2026-11-21. **Every threshold value is proposed**, waiting for the supervisor's
  review of `docs/rubric/S1_ANCHORS_FOR_REVIEW.md` and the owner's decision.
- Date: 2026-11-21
- Decider: project owner, after supervisor review
- Plan reference: FINAL_12 §2 (P2: "VALUE-adapted rubric", study "Rubric anchors reviewed by
  supervisor"); FINAL_05 §1–§3 (skills, scores 0–1, promotion rules); FINAL_04 §3 (content
  measures, opening 20 %, closing 15 %, first 2 sentences); FINAL_07 §F (VALUE rubric, licence);
  FINAL_10 E4/E5 (content-detector F1, rubric κ)
- Config keys affected: `configs/rubric.yaml: s1_presentation.values.*` (new, all proposed)

## Context

P2 builds the text measures, the rubric and the knowledge base. The rubric comes first, so the
supervisor has time to review its anchors before the detectors are built (owner, 2026-11-21).
The source is the AAC&U Oral Communication VALUE rubric (CC BY-NC-SA 4.0): five dimensions
(Organization, Language, Delivery, Supporting Material, Central Message) at four levels
(Capstone 4, Milestones 3 and 2, Benchmark 1). Its wording is holistic ("compelling",
"memorable", "speaker appears polished and confident"). PresenceAI may only state observable
behaviour with numbers (AGENTS.md "Wording") and never infers confidence or other traits
(AGENTS.md rule 3).

## Decision

1. **Source copy.** The Binghamton copy named in FINAL_07 returned 404 on 2026-11-21, as did the
   UF and Brooklyn copies. With the owner's approval the UT Tyler copy was downloaded into the
   git-ignored `data/texts/raw/` (SHA-256 `ba41c136…bde`, recorded in the rubric file). It shows
   no licence line. The owner decided to proceed with CC BY-NC-SA 4.0, the licence recorded in
   `DATA_SOURCES.md` (checked 2026-11-11), and to use the copy for structure only: **no VALUE
   sentence is copied**; every anchor is our own wording.
2. **Files and licences.**
   - `resources/rubrics/s1_presentation.yaml`: dimensions, levels, anchor wording, measures.
     It is a derivative of the VALUE rubric, so it is **CC BY-NC-SA 4.0** with attribution (as
     NOTICE requires for `resources/`).
   - `configs/rubric.yaml`: the threshold values (AGENTS.md rule 7).
   - The anchor wording stays in `resources/` rather than `src/presenceai/lang/`, because
     `lang/` is Apache-2.0 code and the wording is a CC BY-NC-SA derivative. This is the one
     exception to "user-facing text lives in `lang/`". An Arabic version follows after the
     supervisor's review, so it is translated once, from the reviewed wording.
   - `schemas/rubric.py`, `rubric/loader.py` (checks the two files agree: no missing and no
     unused threshold), `rubric/review.py` and `scripts/render_rubric_review.py` (the review
     document, generated; a test fails when it is stale).
3. **Levels and scores** (owner, 2026-11-21). VALUE's four levels are kept. A level is reached
   when all its conditions hold; the dimension gets the highest level reached; level 1 is the
   default. The engine's observation is **y = (level − 1) / 3**: 0, 0.33, 0.67, 1. Discrete
   levels also suit Cohen's κ in E5.
4. **Missing measures are never penalised.** A condition whose measure cannot be computed
   (camera off, no session limit, Arabic content, unreliable punctuation) is dropped. A
   dimension with no computable condition is "not measurable" (FINAL_05 §1: camera-off users
   are not penalised).
5. **Left out.** Each dimension lists the parts of the VALUE wording that no rule can observe
   (e.g. "compelling", "memorable", credibility of sources, appropriateness to the audience).
   The VALUE Delivery wording about how the speaker *appears* (confident, comfortable) is
   dropped for good: it is an inference about the person (AGENTS.md rule 3). Posture and
   gesture, vocal variety (opt-in, ADR 0013) and speech rate (own band only, P5) are not in
   the levels.
6. **Proposed thresholds.** From the plan: opening 20 %, closing 15 %, hook in the first 2
   sentences (FINAL_04 §3). All other values are first proposals for a 2–3 minute talk with no
   source yet; the full table is in `configs/rubric.yaml` and the review document. Candidates
   for a data-based check later: E12 reference ranges from public talks.
7. **Content checks by a language model come under consent switch (d)** (owner, 2026-11-21).
   The rubric judge and AI wording send the same transcript text to the same provider, so one
   switch covers both: "AI wording and content checks". With (d) off the judge does not run,
   and those measures use rules (and Laya, later) with lower confidence. The judge itself is
   deferred; its ADR is written in the judge slice. AGENTS.md rule 2 and FINAL_11 §3 switch (d)
   are reworded accordingly.
8. **Laya is deferred**, and its checkpoint download (about 2.3 GB) moves into
   `scripts/download_models.py` when it is added (AGENTS.md rule 2: model downloads happen only
   there).

## Consequences

- P2-b builds the planned `text.*` measures; P2-c adds `pauses.mid_sentence_per_min` and the
  scoring. Gaze (P6) and overrun (P3) come later; until then Delivery rests on fillers and
  mid-sentence pauses only.
- With base Whisper, fillers per 100 words is a lower bound (ADR 0012), so Delivery levels can
  be too high for speakers whose fillers Whisper drops.
- English only for the content dimensions (FINAL_04 §5: Arabic content measures are not scored).
- P2-b checks the wordfreq data licence (the code is Apache-2.0; the word lists may carry
  CC BY-SA 4.0 data) and adds attribution to NOTICE if needed (owner, 2026-11-21).
- S2 (project defense) reuses these five dimensions scaled to a 90-second explanation, plus a
  `followup_handling` criterion (relevance, response latency, acknowledged limitations;
  FINAL_05 §1), **after** this draft is reviewed (owner, 2026-11-21).

### Open items

1. **Engine promotion rules and these scores (for P5; the engine is not changed now).** The
   engine promotes after two observations with y_adj ≥ 0.75 and demotes after two with
   y_adj ≤ 0.40 (`configs/engine.yaml: difficulty`, FINAL_05 §3). The rubric gives only 0,
   0.33, 0.67 and 1. With y_adj = y + β·(d − d_ref), a level-3 score (0.67) reaches 0.75 only
   when β·(d − d_ref) ≥ 0.083, that is, never at or below the reference difficulty. So a
   speaker at level 3 can never move up there, and level 2 (0.33) always counts toward
   demotion. The promotion and demotion values, or the score mapping, must be revisited in P5.
2. Supervisor review of the anchors and the numbers, then the owner decides the values and
   this ADR becomes accepted.
3. The Arabic version of the anchor wording, after the review.
