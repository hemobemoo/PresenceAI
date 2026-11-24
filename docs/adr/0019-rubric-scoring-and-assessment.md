# 0019 — Rubric scoring and the assessment with verified evidence (P2-c)

- Status: **accepted** by the owner on 2026-11-21 (P2-c plan and answers 1–5).
- Date: 2026-11-21
- Decider: project owner
- Plan reference: FINAL_12 §2 (P2 exit: "Assessment with verified evidence for every finding");
  FINAL_05 §1–§2 (skills, observations with confidence); FINAL_10 E5 ("share of feedback
  statements with verified evidence"); AGENTS.md rule 4; ADR 0011, 0014, 0017, 0018
- Config keys affected: `configs/rubric.yaml: s1_presentation.values.mid_pause_min_s` (new,
  1.0 s, provisional); `configs/rubric.yaml: partial_cap` (new 2026-11-24, decided: 0.5, 3)

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
   - **M leaves out measures absent by the user's choice** (owner, 2026-11-24; P3-a amendment
     item 2a): camera measures when the camera is off, opted-out measures and excluded skills.
     **Accepted consequence: Arabic camera-off Delivery is "based on 2 of 3 measures"** (fillers
     and overrun) **and is not capped**; it keeps low confidence (base-Whisper fillers).
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

### Open items

1. ~~**Partial levels in the report (owner, 2026-11-22, for P3).**~~ **Closed 2026-11-24 (P3-a,
   below):** both. Every finding shows "based on N of M measures", and a level based on at most
   half of its measures is capped at level 3.
2. **Central message floor effect (owner, 2026-11-25).** Central message is at **level 1 on all
   12 real English talks so far**: the 10 B1 clips and the owner's 2 real S1 runs in the app.
   Thesis sentences found: 0 on both S1 runs, although each opens with a topic announcement
   ("Today I will talk about…"), which the cue lists count as an introduction cue, not a thesis
   (ADR 0018, review). **No cue list is changed for this**: central-message lexicons may only be
   tuned on C2 development items the owner labels by hand (ADR 0021 §4c), now planned before P5
   (FINAL_12). Meanwhile the report card at level 1 says, in rules-only wording, what was
   checked and how to state a main point (P3-d), and the engine must not let a floor-effect
   skill keep winning "weakest skill" (FINAL_05 §3).

## P3-a amendment: partial levels, overrun, one failed finding (2026-11-24)

Owner's decisions in the P3 plan (2026-11-24); config keys `configs/rubric.yaml: partial_cap`
(new, decided) and the measure `timing.overrun_s` (now available in both rubrics).

1. **"Based on N of M measures" is always shown,** also when every measure was computed ("4 of
   4"). The finding's `measured` and `total` fields carry it, so the app shows it on each card;
   the statement already says it (item 3 above). A not-measurable finding says "none of its M
   measures".
2. **Partial-level cap.** When **at most half** of a dimension's measures are computed
   (computed share ≤ `partial_cap.max_share`, 0.5), the level is capped at
   `partial_cap.max_level` (3). The owner chose "at most half" over "fewer than half", so
   **2 of 4 is capped** (for example, camera on and gaze missing); with the camera off, M is 3
   (item 2a).
   - `DimensionResult.capped_from` / `Finding.capped_from` keep the level the conditions reached;
     `level` and the score y = (level − 1) / 3 follow the capped level; the confidence is
     unchanged (the lowest computed measure).
   - The finding adds: "Reached level 4, limited to level 3 because only 1 of 4 measures could
     be computed."
   - Levels 1–3 are never changed by the cap. The cap is the one deliberate exception to "a
     missing measure never lowers a level". It applies to every rubric, so it also covers
     ADR 0022 open item 1 (Arabic follow-up handling on latency alone: 1 of 3, capped).
   - `next_level` is taken from the capped level, so a capped level 3 lists the unmet computed
     level-4 conditions; when none are unmet, the missing measures are the gap.
2a. **What M counts** (owner, 2026-11-24, before merging P3-a). FINAL_05: camera off is never
   penalised, so **a measure absent because of the user's choice is removed from M**, both for
   the cap and for "based on N of M":
   - measures that need the camera when the camera is off, for any reason, including no camera
     before P6 (`score.assess(camera_on=False)`, the default until P6);
   - opt-in measures not turned on and excluded skills (`opted_out`, for later phases).

   **Absences caused by system limits still count in M:** Arabic, punctuation failure, no time
   limit on an upload, a failed stage, and gaze missing while the camera is on.
   `DimensionResult.excluded` lists the measures left out. Examples: an English upload with
   the camera off has Delivery "based on 2 of 3" (fillers, mid-sentence pauses; overrun missing)
   and is not capped; Arabic with the camera on and gaze missing for a system reason is 2 of 4
   and capped. **Accepted consequence (owner): Arabic camera-off Delivery from fillers plus
   overrun is 2 of 3 and not capped**; it still says "based on 2 of 3 measures", with low
   confidence. A dimension made only of excluded measures (none today) would keep them, as
   missing.
3. **Overrun** (`rubric/derived.py: overrun`, planned in P3 since ADR 0017): seconds from the
   session's time limit to the end of the last speech region, 0 when within it; time evidence is
   the limit to the end of speech (or the whole talk when within it). The recording's start is
   time 0, so it is measured **only for recordings made in the app**, which pass the scenario's
   limit (`limit_s`); for an uploaded file it is missing with the reason **"uploaded file: no
   time limit"** (owner). Medium confidence: the end comes from the voice detector, and a remark
   after the talk counts as speech. The `assess` command analyses files, so it never measures
   overrun.
4. **One failed finding must not crash the report** (item 5 above, now built). `score.assess`
   and `pipeline.assess.assess_report` take `strict`: the CLI and the tests keep `strict=True`
   (a failure in a rule-based finding is a bug and stops); the app uses `strict=False`.
   - Any error while scoring, wording or verifying one dimension makes **that dimension** "not
     measurable" ("could not be checked"); `Assessment.failed_dimensions` lists it; the other
     dimensions stand.
   - A failed text analysis makes every text measure missing ("text analysis failed"), so the
     content dimensions are not measurable and Delivery is still scored.
   - The log holds the dimension id (or stage) and the error class only, never transcript text
     (tested).

