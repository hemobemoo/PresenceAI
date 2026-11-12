# 0001 — Record architecture decisions

- Status: accepted
- Date: 2026-11-12
- Decider: project owner
- Plan reference: AGENTS.md rule 7; docs/plan/FINAL_08_FOLDER_STRUCTURE.md
- Config keys affected: all of `configs/`

## Context

Thresholds, model ids and targets live in `configs/`. Coding agents may propose values, but a human decides them, and the reasoning must survive for the thesis and for later sessions.

## Options considered

1. Comments in the YAML files only — no history of why, easy to lose.
2. One short ADR per decision in `docs/adr/` — small cost, traceable.

## Decision

Use ADRs in `docs/adr/`, numbered `NNNN-title.md`, copied from `0000-template.md`. Every change to a threshold, model id or target in `configs/` references an accepted ADR. Values in `configs/` marked "ADR pending" or `null # TODO(ADR)` are not yet decided.

## Consequences

- No threshold is tuned to make a test pass (AGENTS.md rule 7).
- Spike outcomes (S1–S5) are recorded as ADRs.
