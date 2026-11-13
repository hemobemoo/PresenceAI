# 0003 — OpenRouter free models and prompt privacy (spike S4)

- Status: **proposed** (agent, 2026-11-13). The privacy rule in part A was decided by the owner on
  2026-11-13; the model choice in part B waits for the owner.
- Date: 2026-11-13
- Decider: project owner
- Plan reference: docs/plan/FINAL_12_ROADMAP_AND_THESIS.md §1 (S4); FINAL_11 §3–§5; FINAL_13 §3.3;
  FINAL_07 line 69
- Config keys affected: `configs/models.yaml: llm.openrouter.pinned`, `llm.openrouter.fallbacks`
- Evidence: `experiments/results/s4/20261113-143329_requests.csv`, `_replies.jsonl`, `_summary.csv`,
  `_summary_rescored.csv` (same run, secondary numbers check corrected, see Consequences);
  `experiments/results/s4/20261113-143154_policies.csv`; `experiments/s4_openrouter/terms_review.md`.
  All produced by `experiments/s4_openrouter/` (run.py, policies.py)

## Part A — Prompt privacy (decided by the owner, 2026-11-13)

### Context

The earlier rule allowed only OpenRouter free endpoints that neither log nor train on prompts. Very
few capable free models meet it.

### Decision

OpenRouter free endpoints that may log or train on prompts are allowed. Participants are told this
in consent switch (d), and can decline it; the app then uses templates. Protections stay: text
only, personal names redacted, no media, no ids.

- Reason: more capable free models at no cost.
- Requests send `provider.data_collection: "allow"` and never a `user` field. The OpenRouter
  account privacy setting must allow these endpoints (FINAL_13 step 3.3).
- Consent switch (d) now states that the model provider stores the text and may use it to improve
  its models (`src/presenceai/lang/en/ui.yaml`, `lang/ar/ui.yaml`: `hosted_ai`).
- **Unchanged:** FINAL_07 line 69. Speak & Improve text or audio is never sent to OpenRouter; its
  terms require local models or no-retention APIs, and this decision does not relax that.

### Enforcement now and later

- Spike only: `experiments/s4_openrouter/core.py: check_payload` refuses any payload that is not
  text only, contains a `user` field or a listed name, or lacks `data_collection: "allow"`
  (tests in `tests/unit/test_s4_core.py`).
- **Not yet in the product (P7 items, FINAL_11 §4):** a test that inspects the product request
  payload (text only, names redacted), and a test that consent (d) off means no OpenRouter call.
  The product provider (`src/presenceai/coach/providers/openrouter.py`) does not exist yet.

## Part B — Model choice (proposed)

### Context

S4 asks whether OpenRouter free models return usable feedback JSON. Pass condition (FINAL_12 §1):
**≥ 95 % JSON validity and ≥ 90 % of quotes found verbatim on 30 English prompts**. The owner's
rules (2026-11-13): 1 pinned model and 2 fallbacks from **three different providers**; at least
one usable as the T2 teacher; when two models are close, prefer a provider that does not train on
prompts as the pinned model.

Run 20261113-143329: 7 models × (30 English + 5 Arabic synthetic prompts) = 245 requests,
temperature 0.2, `max_tokens` 4096, reasoning effort low (excluded from the reply), one retry.

### Results (English, n = 30 each)

| Model | Provider | JSON valid | Quotes found | Main failure | Trains on prompts* |
|---|---|---|---|---|---|
| nvidia/nemotron-3-ultra-550b-a55b:free | Nvidia | **1.000** | **1.000** | none | yes |
| nvidia/nemotron-3.5-super-vl:free | Nvidia | **0.967** | **1.000** | 1 cut off at 4096 tokens | yes |
| recursive/flint-v1.0:free | Recursive | 0.833 | 1.000 | 5 cut off at 4096 tokens | no (flag); terms say yes |
| dots-studio/dots-3-note-preview:free | AtlasCloud | 0.400 | 1.000 | 18 cut off at 4096 tokens | no |
| google/gemma-4-31b-it:free | Google AI Studio | 0.067 | 1.000 | HTTP 429 on 28 of 30 | no |
| liquid/lfm-3-1.7b:free | Liquid | 0.033 | 1.000 | 29 cut off at 4096 tokens | yes |
| thinkingmachines/inkling-2-small:free | Thinking Machines | 0.000 | – | HTTP 403 on all: "only available on agentic harnesses" | yes |

\* OpenRouter data-policy flag `training`, from `20261113-143154_policies.csv`. All seven
providers retain prompts. "Quotes found" is over valid replies only. Only the two NVIDIA models
pass. T2 teacher verdicts (terms_review.md): unknown for all except Liquid (prohibited).

### Options considered

1. **Pinned Nemotron 3 Ultra, fallbacks Flint and Gemma 4 31B** (proposed, provisional). Three
   providers. Only the pinned model passed. Flint failed only by running out of tokens; Gemma
   produced 2 valid replies of 2 served and failed on upstream rate limits.
2. **Pinned Nemotron 3 Ultra, fallback Nemotron 3.5 Super VL.** Both pass, but they share one
   provider, which breaks the owner's three-provider rule and gives no protection when Nvidia's
   free endpoints are down.
3. **Re-run S4 for the non-NVIDIA models** (Flint, Gemma, dots) with a higher token limit and,
   for Gemma, slower pacing (about 105 requests), before choosing fallbacks.

### Decision (proposed)

Option 1 in `configs/models.yaml`, with option 3 recommended before acceptance:

```yaml
llm:
  openrouter:
    pinned: nvidia/nemotron-3-ultra-550b-a55b:free
    fallbacks: [recursive/flint-v1.0:free, google/gemma-4-31b-it:free]
```

- The non-training preference could not be applied: only NVIDIA models passed, and NVIDIA trains
  on prompts.
- Between the two NVIDIA models, Ultra had the higher JSON validity (1.000 vs 0.967) and Arabic
  validity (5/5 vs 4/5); Super VL was faster (median 19.5 s vs 27.2 s).
- **The T2 teacher requirement is not met.** No model is confirmed usable as the T2 teacher;
  all chosen models are "unknown". It needs the owner's or supervisor's reading of the
  "compete" clauses (NVIDIA §4.12, Google, AtlasCloud) or the provider's written confirmation.

### Risks to flag

- **NVIDIA API Trial ToS** §1.2 allows "limited trial purposes only and without use … in
  production", and §1.4 limits use without a subscription to internal testing and evaluation.
  Whether a user study counts as production needs the owner's or supervisor's judgement.
- **Recursive:** OpenRouter flags `training: false`, but Recursive's model terms §5.2(c) allow
  training on customer data. The consent wording must not rely on the flag.
- Free endpoints change without notice (Inkling became unusable through the API).

## Consequences

- `tests/unit/test_openrouter_models.py::test_repo_config_has_pinned_and_two_fallbacks` checks
  the three-provider rule on `configs/models.yaml`.
- A bug found after the run: the secondary "numbers" check (not part of the pass rule) did not
  allow the 1.5 s pause length stated in the prompt. It was fixed (`core.LONG_PAUSE_S`, new
  test) and the saved replies were re-checked with `run.py --rescore 20261113-143329`. Only
  `numbers_ok_rate` changed; the pass results did not.
- The decision is wrong if, in the product, the pinned model's JSON validity on real sessions
  falls below 95 %, or if a re-run shows a fallback still below the pass condition.
- Supervisor to be told of the privacy decision (FINAL_11 §5).

## Files changed for this ADR

- Part A: `docs/plan/FINAL_11_ETHICS_PRIVACY.md`, `docs/plan/FINAL_13_SETUP_GUIDE.md`,
  `src/presenceai/lang/en/ui.yaml`, `src/presenceai/lang/ar/ui.yaml`
- Model ids: `configs/models.yaml`, `src/presenceai/settings.py`, `scripts/check_setup.py`,
  `.env.example`, `tests/unit/test_openrouter_models.py`
- Spike: `experiments/configs/s4_openrouter.yaml`, `experiments/s4_openrouter/` (core.py, run.py,
  policies.py, prompts_en.yaml, prompts_ar.yaml, terms_review.md, README.md),
  `tests/unit/test_s4_core.py`, `experiments/results/s4/`
- Report: `docs/report-drafts/S4_summary.md`, `docs/report-drafts/README.md`, this ADR
