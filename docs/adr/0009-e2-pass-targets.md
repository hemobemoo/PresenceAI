# 0009 — E2 pass targets (speech rate, pauses, duration on C1)

- Status: **accepted**. The owner accepted the proposed targets on 2026-11-17 (P1 decision 1).
- Date: 2026-11-17
- Decider: project owner
- Plan reference: docs/plan/FINAL_10 §1 (E2); FINAL_12 P1 (exit: "E2 passes on C1");
  FINAL_04 §2 (rate, pauses)
- Config keys affected: `configs/evaluation.yaml: e2`

## Context

FINAL_10 lists E2's metrics (wpm error %, pause F1 at 0.5 and 1.0 s, duration error) but no
numeric targets. P1 cannot exit without them, and AGENTS.md rule 7 requires them in `configs/`
with an ADR before E2 runs.

## Decision

Values as in `configs/evaluation.yaml` (`e2:`). E2 passes only if all of these hold on C1:

| Metric | Aggregation across clips | Target |
|---|---|---|
| Words-per-minute error | mean absolute percentage error, \|wpm_measured − wpm_true\| / wpm_true | ≤ 5 % |
| Pause detection F1 at 0.5 s | F1 over all clips (pauses ≥ 0.5 s) | ≥ 0.90 |
| Pause detection F1 at 1.0 s | F1 over all clips (pauses ≥ 1.0 s) | ≥ 0.90 |
| Duration error | mean absolute error, seconds | ≤ 0.10 s |

E2 also writes every per-clip value (wpm error, pause precision/recall/F1 at each threshold,
duration error) next to the aggregate, so a mean that hides a few bad clips is visible.

How a detected pause is matched to a true pause (the onset tolerance) is defined in the E2
experiment config before E2 runs. It is reported with the results and is not tuned after seeing
them.

## Consequences

- If E2 misses a target, the result is recorded as a fail. The targets are not relaxed without the
  owner's decision and a new ADR (AGENTS.md rule 7).
- E2 on Arabic pilot clips (FINAL_10 §1) reports the same metrics descriptively. These targets
  apply to C1.
