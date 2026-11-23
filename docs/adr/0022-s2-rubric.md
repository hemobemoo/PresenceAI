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
   `configs/rubric.yaml` are **placeholders, not used for scoring** (0.85 / 0.80 / 0.75).
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

1. **Arabic follow-up handling rests on latency alone.** With relevance and limitation cues not
   measured for Arabic, levels 4 and 3 could be reached on latency only, the same issue as
   ADR 0019 open item 1 (partial levels); the owner's P3 decision there covers this dimension.
2. **Relevance thresholds** are set in S2-b from the owner's 20 pairs (item 5).
3. ~~Supervisor review~~: done 2026-11-23 (above).
