# Limitations

Started 2026-11-17. Kept up to date as experiments report (FINAL_11 §6).

## Evaluation design

- **No multi-participant user study.** PresenceAI was not evaluated with a group of real learners
  (ADR 0010). Evidence for the adaptive engine (C1) comes from a simulator whose measurement noise
  is set from public-data results (E9), from agreement with two expert raters on public talks (E13),
  and from a single-case ABAB self-study by the developer (E14). None of these shows that the app
  improves real learners' presentations. **Validation with real learners is future work.**
- **The single-case study is n = 1,** and the participant is the developer, who knows how the
  engine works. Its results are descriptive and do not generalise.
- **The expert raters are two:** the developer and one classmate. Neither is a professional
  speaking coach, and agreement between them limits what agreement with the engine can mean.
- **Simulated learners** follow the simulator's assumptions about learning, transfer and
  forgetting (FINAL_05 §8). Conclusions hold only within those assumptions; sweeps over wrong noise
  assumptions show how sensitive they are.

## Data

- Measurement validity (C2) is checked on public recordings (`docs/data/DATA_SOURCES.md`). These are
  meetings, podcasts, read speech and published talks, not novice practice presentations recorded
  on a laptop webcam, so error rates in the app may differ.
- Non-public speech and video come only from the developer's own recordings (B1, B2). The
  per-condition vision robustness results describe one person.
- **Gaze (T3) is trained on one person** (the developer's own protocol recordings) and **tested
  across people only on Columbia Gaze**. Columbia Gaze contains posed still images taken under
  controlled conditions, not webcam video of people presenting. Its known gaze angles are mapped to
  the app's zones by a fixed rule. **The cross-person accuracy reported in E7 is therefore an
  approximation** of how T3 would do on other users' webcams. It may be optimistic (posed, sharp,
  frontal images) or pessimistic (no per-user calibration, a camera setup unlike a laptop). The AMI
  focus-of-attention check is meeting video, not presentations, and is only a sanity check.
- **Arabic rate and pause accuracy (E2) has no public ground truth.** None of the allowed Arabic
  sources (Common Voice Arabic, MASC) has word-level timing or pause labels, and the synthetic
  generator (Kokoro) has no Arabic voice. Arabic E2 is measured only on the developer's own
  annotated recordings, one speaker, and is reported descriptively. Arabic speech recognition (E1)
  is covered by Common Voice Arabic and MASC (`docs/data/DATA_SOURCES.md` §3).
- Research-only and no-derivatives sources (TED, TED-LIUM 3, PodcastFillers, Columbia Gaze) are used
  for evaluation only, which limits training data.
