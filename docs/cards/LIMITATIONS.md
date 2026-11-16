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
- Non-public speech and video come only from the developer's own recordings (B1, B2). Arabic
  rate and pause accuracy and the per-condition vision robustness results describe one person.
- The gaze-zone classifier (T3) is trained on one person. Its cross-person accuracy is measured on
  a public gaze dataset of still images, not on webcam video of presentations.
- Research-only and no-derivatives sources (TED, TED-LIUM 3, PodcastFillers, Columbia Gaze) are used
  for evaluation only, which limits training data.
