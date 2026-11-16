# 0010 — No volunteer study: evaluate on public data, simulation, expert agreement and n = 1

- Status: **proposed** (the owner decided the direction on 2026-11-17; this ADR records the
  replacement design and waits for the owner's "accepted")
- Date: 2026-11-17
- Decider: project owner
- Plan reference: docs/plan/FINAL_01 §5–§6; FINAL_05 §7–§8; FINAL_07 A, B; FINAL_10 §2–§4;
  FINAL_11 §3, §5, §6; FINAL_12 §2–§4; ADR 0003 (NVIDIA trial terms), ADR 0004 (gaze zones)
- Config keys affected: `configs/default.yaml: retention.study_media_max_days` (marked not
  applicable; no value changed)

## Context

The plan had an exploratory two-arm user study (E10, n ≈ 20 volunteers) as supporting evidence
for C1, volunteer recordings for C2 (pilot set B1 with 2–3 friends, gaze protocol B2 with 4–6
volunteers), a study consent sheet and a supervisor email about the ethics route. Recruiting about
20 volunteers is not feasible for this project. The owner removed E10 and all participant
recruitment on 2026-11-17.

## Options considered

1. **Keep a smaller study (n ≈ 12 crossover).** It still needs recruitment, ethics approval and
   consent paperwork, the same blocker at a smaller scale. Rejected.
2. **Drop human evidence for C1 entirely (simulation only).** This is cheap, but C1 would rest only
   on the simulator's assumptions, with no check against human judgement. Rejected.
3. **Replace the study with public data, simulation, expert agreement and a single-case
   self-study.** No recruitment. Every human element is either public material or the owner. Chosen.

## Decision

### C2: measurement validity on public data

- E1–E7 and E12 run on public recordings with many speakers instead of volunteer recordings. The
  candidates, their licence, access, labels and a proposed shortlist are in
  `docs/data/DATA_SOURCES.md`. **Nothing is downloaded until the owner approves each source.**
- The gaze protocol (T3), pilot set B1 and robustness set B2 use **only the owner's own
  recordings**. Cross-person gaze robustness comes from a public multi-person gaze dataset
  (shortlist: Columbia Gaze), evaluation only.
- Consequence for T3: "leave-one-person-out" (FINAL_10 E7) is no longer possible with one
  recorded person. T3 is trained on the owner's protocol data. Its cross-person accuracy is
  measured on the public gaze set, with the gaze angles mapped to the `at_camera` / `down_notes` /
  `away` zones (ADR 0004) by a rule fixed before evaluation.

### C1: the adaptive engine, three parts

- **(a) Simulator study, main evaluation (E9, FINAL_05 §8).** Policies: adaptive, fixed, random
  and oracle. The measurement noise per skill is set from the E1–E7 error rates on public data,
  with sweeps around those values and runs with deliberately wrong noise assumptions. Outcomes:
  sessions to target, final skill, regret vs oracle.
- **(b) Expert-agreement check (new E13).** On about 40 public talks, the engine chooses the
  weakest skill and the exercise from the system's measurements. Two raters (the owner and one
  classmate) independently pick the weakest skill and the most useful exercise for the same talks,
  blind to the engine's choice and to each other. Agreement is reported as Cohen's κ
  (rater–rater, each rater–engine) and Krippendorff's α (all three), with bootstrap 95 % CIs.
  The talks come from shortlisted sources whose terms allow this research use. Only derived
  features and ratings are stored. Raters do not record themselves; they rate published talks, so
  this is not a human-participant study.
- **(c) Single-case self-study (new E14).** The owner uses the app for about 10 sessions in an
  ABAB design: blocks alternate between the adaptive and the fixed-sequence policy. It is reported
  as n = 1 with visual analysis of the per-session measures, level and trend per block, and no
  generalisation claims. The owner's recordings follow the app's normal rules: encrypted at write
  time, retention 30 days.

### What is removed

E10 and its power analysis; recruitment, randomisation and participant sessions (FINAL_12
P7/P8); the study consent and information sheet, the study-only consent switches (e) and (f), the
debrief sheet and the participant checklist (FINAL_11), all marked "not applicable: no
human-participant study". The app's own user consent switches (a)–(d) stay. The supervisor
email about the ethics route is dropped.

### NVIDIA free tier (working position)

The owner's working position, recorded here in place of a supervisor confirmation, is that a
non-commercial academic research prototype counts as **evaluation use** under the NVIDIA API
Trial terms (ADR 0003, "NVIDIA trial terms"). If NVIDIA's terms or endpoints change, ADR 0003's
fallback plan applies (Recursive Flint pinned).

### Data-use wording

All planned data sources, books and datasets are used for non-commercial academic research and
graduation-project purposes only, under each source's terms, with each source acknowledged. This
sentence is used unchanged in the docs, model cards and report drafts.

## Consequences

- RQ4 changes from "effect in a small study" to "does the engine's choice agree with expert raters,
  and what does a single-case ABAB show". The thesis claims no causal effect on learners.
- `LIMITATIONS.md` states that there was no multi-participant user study and that validation with
  real learners is future work.
- The owner's own recordings are the only non-public speech and video in the project; B1 and B2
  become smaller and single-speaker, so Arabic E2 and E7 condition effects on B2 describe one
  person.
- Re-check: if κ between the two raters is low (< 0.4), the expert-agreement result says more about
  rater disagreement than about the engine. Report it as such and do not tune the engine to the
  raters.
