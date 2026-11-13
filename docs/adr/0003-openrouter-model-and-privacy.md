# 0003 — OpenRouter free models and prompt privacy (spike S4)

- Status: **proposed**. Part A (privacy), the model choice in Part B and Flint as the emergency
  fallback were decided by the owner on 2026-11-13. The re-run (20261113-211500) and the Arabic
  wording ratings are done; the owner accepts the ADR after CI is green.
- Date: 2026-11-13
- Decider: project owner
- Plan reference: docs/plan/FINAL_12_ROADMAP_AND_THESIS.md §1 (S4) and §4 (risks); FINAL_11 §3–§5;
  FINAL_13 §3.3; FINAL_03 §5; FINAL_06 §5–§6; FINAL_07 line 69
- Config keys affected: `configs/models.yaml: llm.openrouter.pinned`, `llm.openrouter.fallbacks`
- Evidence, all produced by `experiments/s4_openrouter/` (run.py, policies.py):
  - First run `20261113-143329` (7 models; settings `experiments/configs/s4_openrouter.yaml`):
    `experiments/results/s4/20261113-143329_requests.csv`, `_replies.jsonl`, `_summary.csv`,
    `_summary_rescored.csv`
  - Re-run `20261113-211500` (Flint and dots; settings `experiments/configs/s4_rerun.yaml`):
    `experiments/results/s4/20261113-211500_requests.csv`, `_replies.jsonl`, `_summary.csv`
  - Arabic ratings (owner): `<run>_ar_rating_sheet.csv` (blinded, as rated), unblinded by
    `ratings.py` into `<run>_ar_ratings_unblinded.csv` and `<run>_ar_ratings_summary.csv`, for
    both runs
  - Provider data policies: `experiments/results/s4/20261113-143154_policies.csv`
  - Licence and terms: `experiments/s4_openrouter/terms_review.md`

## Part A — Prompt privacy (owner's decision, 2026-11-13)

### Context

The earlier rule allowed only OpenRouter free endpoints that neither log nor train on prompts.
Very few capable free models met it.

### Decision

OpenRouter free endpoints that may log or train on prompts are allowed. Participants are told this
in consent switch (d), and can decline it; the app then uses templates. Protections stay: text
only, personal names redacted, no media, no ids.

- **Reason:** more capable free models at no cost.
- **Request settings:**
  - Requests send `provider.data_collection: "allow"` and never a `user` field.
  - The OpenRouter account privacy setting must allow these endpoints (FINAL_13 step 3.3).
- **Consent wording:**
  - All three chosen providers can train on prompts:
    - NVIDIA: OpenRouter flag `training: true`, and NVIDIA Trial ToS §3.3.
    - Recursive: its model terms §5.2(c), although OpenRouter flags it `training: false`.
  - Consent switch (d) must therefore say that the provider stores the text and may use it to
    improve its models. It already does (`src/presenceai/lang/en/ui.yaml` and
    `lang/ar/ui.yaml`, key `hosted_ai`).
- **Unchanged:** FINAL_07 line 69.
  - Speak & Improve text or audio is never sent to OpenRouter; its terms require local models
    or no-retention APIs.
  - This decision does not relax that rule.

### Enforcement now and later

- **Spike only:** `experiments/s4_openrouter/core.py: check_payload` refuses any payload that:
  - is not text only;
  - contains a `user` field or a listed name;
  - lacks `data_collection: "allow"`.

  Tests are in `tests/unit/test_s4_core.py`.
- **Not yet in the product (P7 items, FINAL_11 §4):**
  - a test that inspects the product request payload (text only, names redacted);
  - a test that consent (d) off means no OpenRouter call.

  The product provider (`src/presenceai/coach/providers/openrouter.py`) does not exist yet.

## Part B — Model choice

### Context

S4 asks whether OpenRouter free models return usable feedback JSON. The pass condition
(FINAL_12 §1) is **≥ 95 % JSON validity and ≥ 90 % of quotes found verbatim on 30 English
prompts**.

**First run (20261113-143329):**
- 7 models × (30 English + 5 Arabic synthetic prompts) = 245 requests.
- Temperature 0.2, `max_tokens` 4096, reasoning effort low (excluded from the reply), one retry.

**Re-run (20261113-211500), requested by the owner:**
- Only Flint and dots, whose first-run failures were output-limit cut-offs: 70 requests.
- `max_tokens` 16384.
- Reasoning off (`enabled: false`) for dots. Flint got `effort: minimal`, because OpenRouter's
  model list reports reasoning as mandatory for Flint.
- Gemma was not re-run: its failures were rate limits.

### Results (English, n = 30 per model)

| Model | Provider | Run | JSON valid | Quotes found | Latency p50 / p95 (s) | Passed |
|---|---|---|---|---|---|---|
| nvidia/nemotron-3-ultra-550b-a55b:free | Nvidia | 143329 | **1.000** | **1.000** | 27.2 / 54.6 | yes |
| nvidia/nemotron-3.5-super-vl:free | Nvidia | 143329 | **0.967** | **1.000** | 19.5 / 38.2 | yes |
| recursive/flint-v1.0:free | Recursive | 143329 | 0.833 | 1.000 | 21.4 / 27.9 | no (5 cut off) |
| recursive/flint-v1.0:free | Recursive | **211500** | **1.000** | **1.000** | 18.8 / 37.8 | yes |
| dots-studio/dots-3-note-preview:free | AtlasCloud | 143329 | 0.400 | 1.000 | 47.4 / 53.9 | no (18 cut off) |
| dots-studio/dots-3-note-preview:free | AtlasCloud | **211500** | **1.000** | **1.000** | 4.7 / 6.9 | yes |
| google/gemma-4-31b-it:free | Google AI Studio | 143329 | 0.067 | 1.000 | 0.6 / 25.4 | no (HTTP 429 on 28) |
| liquid/lfm-3-1.7b:free | Liquid | 143329 | 0.033 | 1.000 | 19.6 / 26.6 | no (29 cut off) |
| thinkingmachines/inkling-2-small:free | Thinking Machines | 143329 | 0.000 | – | 0.3 / 0.5 | no (HTTP 403 on all) |

Notes on the table:
- Latency is per request, including failed ones, from each run's `_summary.csv`.
- "Quotes found" is over valid replies only.
- Inkling returned "only available on agentic harnesses".

**Arabic (n = 5):**

| Model | Valid replies | Quotes found |
|---|---|---|
| Nemotron 3 Ultra | 5/5 | 15/15 |
| Nemotron 3.5 Super VL | 4/5 | 12/12 |
| Flint (re-run) | 5/5 | 15/15 |
| dots (re-run) | 5/5 | 5/15 |

dots' Arabic quotes contained Latin fragments not in the transcript (for example in ar01). The
check added after the rating (`latin_quote_replies` in `<run>_summary_rescored.csv`) flags all
5 dots Arabic replies in the re-run and no reply from any other model.

**Owner's Arabic wording ratings** (1–5; 1 = unusable, 3 = understandable but awkward, 5 = natural
MSA). Ratings were given blind. Only values 1–5 count as rated; the owner entered 0 for replies
with nothing to rate. From `<run>_ar_ratings_summary.csv`:

| Model | Run | Rated / replies | Mean |
|---|---|---|---|
| nvidia/nemotron-3-ultra-550b-a55b:free | 143329 | 5 / 5 | 4.00 |
| nvidia/nemotron-3.5-super-vl:free | 143329 | 5 / 5 | 3.80 |
| recursive/flint-v1.0:free | 143329 | 4 / 5 | 3.75 |
| recursive/flint-v1.0:free | 211500 | 5 / 5 | 3.80 |
| dots-studio/dots-3-note-preview:free | 211500 | 5 / 5 | 4.00 |
| dots, gemma, lfm, inkling | 143329 | 0 / 5 | – |

Notes on the ratings:
- The first-run sheet includes two replies that were not valid JSON (Flint and Super VL, ar01),
  rated 3 with the note that they were not JSON and lacked insight.
- dots was rated 4.00 for wording, although only 5/15 of its quotes were found. Wording and
  quote fidelity are separate criteria.

**Qualitative findings from the Arabic rating** (owner, 2026-11-13, all replies in both runs):
1. *Code-switching.* A good Arabic reply knows when to keep English terminology inside the
   Arabic text. This became a design rule (Consequences).
2. *Usefulness.* Most replies were rated 4 with the note that they had good structure and tips,
   but should focus more on how the insight helps the user, with an extra suggestion or
   rephrasing.

### Options considered

1. **Three providers** (agent's first proposal): pinned Nemotron 3 Ultra, fallbacks Flint and
   Gemma.
   - Rejected by the owner: neither fallback had passed at that point.
2. **NVIDIA only**: pinned Nemotron 3 Ultra, fallback Nemotron 3.5 Super VL.
   - Both passed.
   - One provider is a single point of failure.
3. **Option 2 plus an emergency fallback from another provider**, if one passes a re-run.
   - Flint and dots both passed the re-run.
   - Flint was chosen over dots for these reasons:
     - All of Flint's Arabic quotes were found, against 5/15 for dots.
     - Flint had no flagged reply.
     - dots had one `content_filter` finish (en27, the reply was still valid) and one
       banned-term hit (en30).
   - The en30 hit was a false positive: the word "happy" appeared inside a phrase quoted from
     the transcript.
   - dots was faster (p50 4.7 s against 18.8 s).

### Decision

Option 3, owner's choice on 2026-11-13 (NVIDIA only for now), plus the re-run result:

```yaml
llm:
  openrouter:
    pinned: nvidia/nemotron-3-ultra-550b-a55b:free
    fallbacks: [nvidia/nemotron-3.5-super-vl:free, recursive/flint-v1.0:free]
```

**Runtime chain** (FINAL_06 §6): Nemotron 3 Ultra → Nemotron 3.5 Super VL → Flint (emergency) →
`coach_slm` (if installed) → templates.
- The chain runs only when consent switch (d) is on.
- Gemma is removed.
- The non-training preference could not be applied, because only NVIDIA models passed the
  first run and NVIDIA trains on prompts.

### Single-provider risk and mitigation

The pinned model and the first fallback share one provider, NVIDIA. An NVIDIA outage, a rate
limit or a withdrawn free tier takes out both. Mitigations:
1. **Templates floor:** deterministic EN/AR templates always work, with no network.
2. **`coach_slm`:** the local T2 model, once trained and installed.
3. **Emergency fallback** from another provider: Recursive Flint, which passed the re-run.
4. **Model frozen during study weeks:** no model id change while participants are enrolled
   (FINAL_12 §4).

### NVIDIA trial terms

The NVIDIA API Trial ToS limit use:
- §1.2: "for limited trial purposes only and without use … in production";
- §1.4: internal testing and evaluation only, without a subscription.

**Owner's working reading (2026-11-13), pending the supervisor's confirmation:** a private,
non-commercial university research prototype with volunteer participants is evaluation, not
production. If the supervisor disagrees:
- NVIDIA moves out of the chain;
- Flint becomes the pinned model;
- this ADR is superseded.

**Trade-off if Flint becomes pinned:** Flint's latency in the re-run was p50 18.8 s / p95 37.8 s
per report. This is close to Super VL (19.5 / 38.2 s) and faster than Ultra (27.2 / 54.6 s), so
it is no slower than the current chain. It is still a long wait for the user, and without NVIDIA
the chain would have no second hosted model before `coach_slm` and templates.

### T2 teacher (open)

No provider is confirmed as usable for generating T2 teacher data (`terms_review.md`):
- NVIDIA, Recursive, Google, AtlasCloud and Thinking Machines: **unknown**, because each depends
  on a "compete" clause or is not stated;
- Liquid: **prohibited**.

**Proposed P4 option** (not yet decided):
- Generate the teacher data on Kaggle with an open-weight model under Apache-2.0 or MIT, run
  locally in the Kaggle notebook rather than through a hosted API.
- Then only the model licence applies, and no provider's terms of service do.
- Candidates must be checked against FINAL_06 §5 (quote, number, banned-term and JSON filters).

## Consequences

- **Test:** `tests/unit/test_openrouter_models.py::test_repo_config_has_pinned_and_at_least_one_fallback`
  checks the config shape. It replaces the earlier three-provider test, which encoded a rule the
  owner withdrew.
- **Emergency fallback settings:** Flint needed a larger output cap (16384) than the first run's
  4096. The product provider must use per-model output caps (`model_overrides` in
  `s4_rerun.yaml` shows the settings that passed).
- **Bug fixed after the first run:**
  - The secondary "numbers" check (not part of the pass rule) did not allow the 1.5 s pause
    length stated in the prompt.
  - It was fixed (`core.LONG_PAUSE_S`, new test).
  - The saved replies were re-checked with `run.py --rescore 20261113-143329`.
  - Only `numbers_ok_rate` changed; the pass results did not.
- **Arabic code-switching rule** (owner, 2026-11-13, from the Arabic rating):
  - In the coach's own wording, common English technical and academic terms (e.g. *slides*,
    *feedback*, *pace*) stay in English script where Arabic-speaking students would naturally
    use them.
  - They are never transliterated into Arabic letters. The Arabic equivalent may be given on
    first use. Everything else is Modern Standard Arabic.
  - **Quotes are excluded:** quoted spans must match the transcript exactly, and the quote
    check is unchanged.
  - Rule text for the prompt template: `src/presenceai/lang/ar/coach_prompt.yaml`
    (`language_rule`), with an English counterpart in `lang/en/coach_prompt.yaml`.
  - Check: `presenceai.coach.guardrails.latin_outside_transcript` flags Latin-script text in an
    Arabic quote that is not in the transcript (the dots failure). Tests are in
    `tests/unit/test_arabic_code_switching.py`.
  - Plan docs: FINAL_04 §5 (language packs), FINAL_06 §5 (T2 teacher prompts and data filter)
    and §6 (provider chain), FINAL_10 E11 (rater criterion).
  - **P7 items:**
    - the product prompt builder that inserts `language_rule`, and its test;
    - rejecting Arabic quotes flagged by `latin_outside_transcript` in the provider chain;
    - the rater guide and page criterion.

    None exists yet.
  - The S4 spike prompt was **not** changed, so both recorded runs stay reproducible.
- **Known false positive:** the banned-term check also matches words inside quoted transcript
  text (dots, en30). Recorded for the product guardrail design, not changed here.
- **The decision is wrong if:**
  - in the product, the pinned model's JSON validity on real sessions falls below 95 %; or
  - NVIDIA's free endpoints become unavailable during the study.
- **Still open (followed up after acceptance):**
  - the supervisor told of the privacy decision (FINAL_11 §5);
  - the supervisor's view on the NVIDIA trial terms.

## Files changed for this ADR

- **Part A:**
  - `docs/plan/FINAL_11_ETHICS_PRIVACY.md`
  - `docs/plan/FINAL_13_SETUP_GUIDE.md`
  - `src/presenceai/lang/en/ui.yaml`
  - `src/presenceai/lang/ar/ui.yaml`
- **Model ids and chain:**
  - `configs/models.yaml`
  - `src/presenceai/settings.py`
  - `scripts/check_setup.py`
  - `.env.example`
  - `tests/unit/test_openrouter_models.py`
  - `docs/plan/FINAL_03_TECH_STACK.md` (§5)
  - `docs/plan/FINAL_06_MODELS_AND_TRAINING.md` (§6)
  - `docs/plan/FINAL_12_ROADMAP_AND_THESIS.md` (§4 risk table)
- **Arabic code-switching:**
  - `src/presenceai/lang/ar/coach_prompt.yaml`
  - `src/presenceai/lang/en/coach_prompt.yaml`
  - `src/presenceai/coach/guardrails.py`
  - `tests/unit/test_arabic_code_switching.py`
  - `docs/plan/FINAL_04_MULTIMODAL_MEASUREMENT.md` (§5)
  - `docs/plan/FINAL_06_MODELS_AND_TRAINING.md` (§5, §6)
  - `docs/plan/FINAL_10_EVALUATION.md` (E11)
- **Spike:**
  - `experiments/configs/s4_openrouter.yaml`
  - `experiments/configs/s4_rerun.yaml`
  - `experiments/s4_openrouter/`: core.py, run.py, policies.py, prompts_en.yaml,
    prompts_ar.yaml, ratings.py, terms_review.md, README.md
  - `tests/unit/test_s4_core.py`
  - `experiments/results/s4/`
- **Report:**
  - `docs/report-drafts/S4_summary.md`
  - `docs/report-drafts/README.md`
  - this ADR
