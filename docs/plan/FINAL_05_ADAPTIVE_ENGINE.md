# FINAL 05 — Adaptive Professional Development Engine (Contribution C1)

## 1. Skill taxonomy

| Skill | Rubric dimension (VALUE-adapted) | Fed by |
|---|---|---|
| `central_message` | Central message | Central message measures |
| `organization` | Organization | Intro/body/conclusion, transitions |
| `opening_conclusion` | Organization | Hook and conclusion checks |
| `supporting_material` | Supporting material | Material types and count, unsupported claims |
| `language_clarity` | Language | Jargon without definition, sentence length, readability |
| `concision_timing` | Delivery | Overrun, repetition |
| `fluency` | Delivery | Fillers, false starts |
| `pacing` | Delivery | Rate stability vs own band, pause placement |
| `vocal_variety` | Delivery | Loudness and pitch variation (opt-in) |
| `audience_gaze` | Delivery | Gaze at camera share, longest away stretch (camera on) |
| `posture_gesture` | Delivery | Framing, posture, gesture rate (camera on) |
| `followup_handling` | — (S2) | Relevance, latency, limitation acknowledgement |

Camera-off users simply have no `audience_gaze` and `posture_gesture`; nothing is penalised. Users can exclude any skill (accessibility, personal choice).

## 2. Skill state

Per user and skill: Gaussian belief (μ, σ²) on 0–1, observation count, difficulty level 1–7.

```
new session:  σ² ← σ² + q                          # skills drift between sessions
observation y with confidence c at difficulty d:
              y_adj = y + β·(d − d_ref)            # same score at harder level counts more
              r     = r0 / c                       # low-confidence evidence moves the belief less
              k     = σ² / (σ² + r)
              μ     ← μ + k·(y_adj − μ)
              σ²    ← (1 − k)·σ²
```

Parameters (`configs/engine.yaml`): q, r0, β, d_ref, σ_max, targets per skill. Initial values set by you in an ADR; r0 re-estimated from E5/E6 detector reliability.

## 3. Policy: weakest reliable skill

```
need[s]     = goal_weight[s] · max(0, target[s] − μ[s])
reliable[s] = σ[s] ≤ σ_max  and  n_obs[s] ≥ 2

if no skill reliable:   diagnostic exercise for argmax σ          (explore)
else:                   s* = argmax need over reliable skills       (exploit)
                        candidates = exercises(target=s*, level=level[s*]) not used in last 3
                        choose by Thompson sampling over each exercise's Beta(improved, not_improved)

promote level[s] after 2 consecutive y_adj ≥ 0.75; demote after 2 ≤ 0.40
```

Every call returns a `Decision`: chosen exercise, skill, alternatives with their numbers, live-prompt set, plain-language reason ("This targets supporting material: your lowest reliably measured skill, 0.38 ± 0.09 over 3 attempts").

## 4. Difficulty ladder (from the concept)

| Level | Condition |
|---|---|
| 1 | Private text outline exercise |
| 2 | Audio-only, untimed |
| 3 | Audio, timed |
| 4 | Audio + camera, timed |
| 5 | Timed with one follow-up question |
| 6 | Timed with interruption or changed constraint ("cut to 60 s now") |
| 7 | Full performance with two skeptical follow-ups |

## 5. Exercise library

YAML, one file per skill, ~6 exercises per skill × 2 parallel forms ≈ 140 items, generated from templates and filled with the user's topic. Each item: `id, target_skill, level, scenario, prompt_template (en, ar), constraints, success_criteria, live_prompts, parallel_forms, source` (citing the open textbook section it draws on, `FINAL_07` section F).

Example:

```yaml
id: supporting_material.L3.a
target_skill: supporting_material
level: 3
scenario: s1_presentation
prompt_template:
  en: "Give your main point in 60 seconds. Support it with one statistic and one concrete example, and say where the statistic comes from."
  ar: "قدّم فكرتك الرئيسية في 60 ثانية، وادعمها بإحصائية واحدة ومثال واقعي واحد، واذكر مصدر الإحصائية."
constraints: {time_limit_s: 60}
success_criteria: [supporting_material.statistic, supporting_material.example, timing.within_limit]
live_prompts: [time_remaining]
source: "Fundamentals of Public Speaking (Lumen, CC BY 4.0), ch. Supporting Materials"
```

## 6. Adaptive live prompts

The engine also decides which live-prompt categories are active in the next exercise, so live coaching follows the same personalisation:

| Target skill | Live prompts enabled |
|---|---|
| `pacing` | Pace above/below own band |
| `fluency` | Pause-instead-of-filler reminder after long voiced hesitations (acoustic only) |
| `audience_gaze` | Away/down for > 6 s |
| `posture_gesture` | Out of frame, head turned > 30° |
| `concision_timing` | Time remaining at 30 s and 10 s |
| others | Time remaining only |

Rate limit: one prompt per 20 s. Users can turn prompts off; that choice is logged.

## 7. Policies compared

| Policy | Report | Exercise choice | Live prompts | Used in |
|---|---|---|---|---|
| `AdaptivePolicy` | Full | Weakest reliable skill | Engine-chosen | Study arm A, simulation |
| `FixedSequencePolicy` | Full | Same fixed order for all | Same fixed set | Study arm F, simulation |
| `PracticeOnlyPolicy` | None | Repeat task | None | Simulation; study only if n ≥ 30 |
| `RandomPolicy` | Full | Random | Random | Simulation sanity check |
| `OraclePolicy` | — | Uses true latent skill | — | Simulation upper bound |

All implement `Policy.next(profile, history, library) -> Decision`.

## 8. Simulator

Synthetic learners: latent skill vector, per-skill learning rate (gain when the exercise targets that skill, small transfer to related skills), forgetting, measurement noise per skill (set from E1–E7 error rates), camera-off share, dropout. Outputs per policy over 2,000+ learners × 10 sessions: sessions to target, final mean skill, regret vs oracle, robustness to wrong noise assumptions. Seeded, config-driven, results in `experiments/results/`.

## 9. Tests that guard the engine

Property tests (Hypothesis): σ never increases after an observation; low-confidence observations move μ less than high-confidence ones; promotion never skips a level; excluded skills are never selected; camera-off profiles never get vision exercises; every Decision has a non-empty reason.
