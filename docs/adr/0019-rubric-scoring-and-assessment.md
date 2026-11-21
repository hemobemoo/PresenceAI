# 0019 — Rubric scoring and the assessment with verified evidence (P2-c)

- Status: **accepted** by the owner on 2026-11-21 (P2-c plan and answers 1–5).
- Date: 2026-11-21
- Decider: project owner
- Plan reference: FINAL_12 §2 (P2 exit: "Assessment with verified evidence for every finding");
  FINAL_05 §1–§2 (skills, observations with confidence); FINAL_10 E5 ("share of feedback
  statements with verified evidence"); AGENTS.md rule 4; ADR 0011, 0014, 0017, 0018
- Config keys affected: `configs/rubric.yaml: s1_presentation.values.mid_pause_min_s` (new,
  1.0 s, provisional)

## Decision

1. **Mid-sentence pauses per minute** (`rubric/derived.py`, key `pauses.mid_sentence_per_min`):
   pauses of at least `mid_pause_min_s` (1.0 s, provisional; the anchor wording in English and
   Arabic uses the placeholder) labelled `mid_sentence`, per minute of speaking time. Speaking
   time is the first speech onset to the last offset, as for `rate.wpm`, with the same minimum.
   Missing for Arabic and when the punctuation is unreliable (positions cannot be told, ADR
   0011, 0016); English positions rest on Whisper's punctuation (ADR 0014).
2. **Scoring** (`rubric/levels.py`, `rubric/score.py`): for each dimension the condition
   results (value, limit, holds; a missing measure stays missing), the level (ADR 0017), the
   score y = (level − 1) / 3, and the **confidence: the lowest among the measures that were
   computed**. When all of them are low confidence, the level is still given, at low
   confidence; the engine down-weights it (FINAL_05 §2).
3. **Partial evidence is visible.** Every dimension result and finding states how many of its
   measures were computed ("based on 1 of 4 measures (fillers per 100 words (at least))").
4. **Findings** (`schemas/rubric.py: Finding`): one per dimension, in observable wording from
   templates in `lang/{en,ar}/report.yaml` (level, level name, measures and their values), with
   the evidence of its computed measures and a confidence. A not-measurable dimension says why.
5. **Verification** (`rubric/evidence.py`): a transcript quote must equal the transcript text at
   its offsets; a time span must lie within the recording. **For rule-based findings a failure is
   a bug**: `EvidenceError` is raised, the tests fail, and `assess` stops with it.
   - **In the app (P3), one failed finding must not crash the report:** that dimension shows "not
     measurable", and the failure is logged as counts only, never transcript text.
   - Language-model claims whose quote is not found are dropped silently (AGENTS.md rule 4).
6. **Next-level gaps** are structured data only (`DimensionResult.next_level`: the unmet,
   computed conditions of the next level). **The P7 coach slice must word them in the templates
   too, not only through the language model**, so users with consent (d) off still get a next
   step.
7. **Output:** a new command, `presenceai assess FILE --lang en|ar [--scenario s1_presentation]`,
   writes one JSON (`schemas/report.py: AssessmentReport`: speech report, text report,
   assessment). `analyze` is unchanged. For Arabic no text parser is loaded.

## Consequences

- The P2 exit criterion is met in tests: every finding with a level has evidence and it
  verifies (`tests/unit/test_assessment.py`), including the CLI path with fakes.
- **B1 observation (owner's laptop, counts only, nothing tuned on B1):** on the 20 cached B1
  reports (Whisper's transcripts), every finding verified. Central message was level 1 on all
  10 English clips and organization level 1 on 9: the thesis and introduction lexicons rarely
  match these talks. The lexicons are adjusted on the C2 development part only (P2-e, ADR 0018);
  this is a reason to look there, not a change. Arabic delivery was level 4 on all 10 clips,
  from fillers alone (1 of 4 measures, low confidence; base-Whisper fillers are lower bounds).
- Gaze (P6) and overrun (P3) stay missing until built; Delivery rests on fillers and
  mid-sentence pauses.
