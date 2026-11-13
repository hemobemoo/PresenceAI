# Spike S4: OpenRouter free models for feedback wording

## 1. Aim

Spike S4 examined whether free models served through OpenRouter could turn the application's
measurements and a practice-talk transcript into structured feedback that the application could
check automatically. Each feedback item had to quote the transcript verbatim, so that every
statement carried evidence. The spike was also used to choose the models for the optional
"AI wording" feature. The pass condition was taken from the project roadmap
(`docs/plan/FINAL_12_ROADMAP_AND_THESIS.md`, §1).

## 2. Method

**Models.** Seven free models were shortlisted and approved by the project owner on 2026-11-13.
They came from six serving providers (Table 1).

**Prompts.** Each model received the same 30 English and 5 Arabic transcripts. All transcripts
were synthetic and written for the spike, and personal names in them were fictional. Each
request contained:

- a fixed system prompt;
- five measurements (duration, word count, speech rate, filler words and pauses longer than
  1.5 s);
- the transcript, with personal names replaced by `[NAME]` before sending.

**Payload.** Each request contained text only, no media, no identifiers and no `user` field, and
set `provider.data_collection` to `"allow"`. A guard function rejected any other payload before
it was sent.

**First run** (id `20261113-143329`):

- 245 requests in total.
- Temperature 0.2 and an output limit of 4,096 tokens.
- Low reasoning effort, with reasoning text excluded from the reply.
- Requests were sent to every model in turn, at least 4 s apart, with one retry on rate-limit or
  server errors.

**Re-run** (id `20261113-211500`), at the owner's request:

- Only the two models whose first-run failures were mainly output-limit cut-offs were re-run:
  Flint and dots, 70 requests in total.
- The output limit was raised to 16,384 tokens.
- Reasoning was disabled for dots. Flint was set to minimal reasoning effort instead, because
  OpenRouter's model list reported reasoning as mandatory for it.
- All other settings were unchanged.

**Automatic checks on each reply:**

- **JSON validity:** the reply parsed as JSON and matched the reply schema of two to four items,
  each with an observation, an evidence quote and a suggestion.
- **Quote found:** each evidence quote of at least three words appeared verbatim in the
  transcript that was sent, after normalising whitespace, case and punctuation.
- **Secondary checks** were recorded but did not count towards the pass condition. The first
  tested whether every number in the reply was one of the measurements. The second tested
  whether the reply contained a term the project bans, such as emotion or personality words.

**Provider policies.** Each provider's data policy was recorded from its public OpenRouter model
page (`policies.py`, 2026-11-13). Each licence and set of terms of service was read for how
prompts are used and whether outputs may be used to train other models (`terms_review.md`).

## 3. Pass conditions

A model passed if, on the 30 English prompts:

- at least 95 % of replies were valid JSON; and
- at least 90 % of the quotes in valid replies were found in the transcript.

The Arabic replies were checked in the same way. Five prompts were too few for a pass decision,
so the owner rates their wording separately (Section 6).

## 4. Results

Table 1 combines the summary files of both runs (`<run id>_summary.csv`) with the provider data
policies (`20261113-143154_policies.csv`).

**Table 1.** English results per model and run (n = 30 prompts each).

| Model | Provider | Run | JSON valid | Quotes found | Latency p50 / p95 (s) | Trains on prompts (flag) | Passed |
|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3-ultra-550b-a55b:free | Nvidia | 143329 | 1.000 | 1.000 | 27.2 / 54.6 | yes | yes |
| nvidia/nemotron-3.5-super-vl:free | Nvidia | 143329 | 0.967 | 1.000 | 19.5 / 38.2 | yes | yes |
| recursive/flint-v1.0:free | Recursive | 143329 | 0.833 | 1.000 | 21.4 / 27.9 | no | no |
| recursive/flint-v1.0:free | Recursive | 211500 | 1.000 | 1.000 | 18.8 / 37.8 | no | yes |
| dots-studio/dots-3-note-preview:free | AtlasCloud | 143329 | 0.400 | 1.000 | 47.4 / 53.9 | no | no |
| dots-studio/dots-3-note-preview:free | AtlasCloud | 211500 | 1.000 | 1.000 | 4.7 / 6.9 | no | yes |
| google/gemma-4-31b-it:free | Google AI Studio | 143329 | 0.067 | 1.000 | 0.6 / 25.4 | no | no |
| liquid/lfm-3-1.7b:free | Liquid | 143329 | 0.033 | 1.000 | 19.6 / 26.6 | yes | no |
| thinkingmachines/inkling-2-small:free | Thinking Machines | 143329 | 0.000 | – | 0.3 / 0.5 | yes | no |

Latency is measured per request and includes failed requests.

**Quotes.** Every quote in a valid English reply was found in the transcript, for all models.

**Causes of failure in the first run** (`20261113-143329_requests.csv`):

- **Output limit.** Replies were cut off at the 4,096-token limit, usually with empty content.
  This affected 29 Liquid, 18 dots, 5 Flint and 1 Nemotron 3.5 Super VL English replies.
- **Rate limits.** Google AI Studio returned HTTP 429 on 28 of 30 English requests, even after
  the retry. Both replies it did serve were valid.
- **No API access.** Every Inkling request returned HTTP 403. The message stated that the model
  was available only through listed agent applications.

**Re-run.** With the larger output limit, Flint and dots both passed. Neither had a cut-off reply.
One dots reply ended with a content-filter finish but was still valid.

**Arabic** (n = 5):

- Nemotron 3 Ultra and Flint (re-run) each returned 5 valid replies, with all 15 quotes found.
- Nemotron 3.5 Super VL returned 4 valid replies, with all 12 quotes found.
- dots (re-run) returned 5 valid replies, but only 5 of 15 quotes were found. Its quotes
  contained Latin-script fragments that did not occur in the transcripts.

**Banned terms.** One dots reply (en30) was flagged. The flagged word was part of a phrase quoted
from the transcript, so the flag was a false positive of the check.

**Provider policies:**

- All seven providers retain prompts.
- OpenRouter flags Recursive as not training on prompts, but Recursive's own model terms permit
  training on customer data.
- On training other models with the outputs, Liquid's terms prohibit it. For the other
  providers, permission could not be confirmed either way.

**Correction to a secondary check.** After the first run, the secondary numbers check was found
to reject the 1.5 s pause length stated in the prompt. The check was corrected, and the saved
replies were re-checked without new requests (`20261113-143329_summary_rescored.csv`). Only the
secondary rate changed; the pass results did not.

## 5. Decision

The decisions below were recorded in ADR 0003, which remained "proposed" at the time of writing.

**Model chain.** On 2026-11-13 the project owner chose NVIDIA models only for the pinned model and
the first fallback. Both had passed the first run. After the re-run, a model from another
provider was added as an emergency fallback. The runtime chain was therefore:

1. `nvidia/nemotron-3-ultra-550b-a55b:free` (pinned). It had the highest JSON validity in
   English (1.000) and in Arabic (5 of 5).
2. `nvidia/nemotron-3.5-super-vl:free` (fallback).
3. `recursive/flint-v1.0:free` (emergency fallback). It was preferred to dots because all of its
   Arabic quotes were found and none of its replies was flagged. dots was faster.
4. The local coaching model, once trained and installed.
5. Deterministic templates.

The Gemma model was removed from the configuration. Its failures reflected the provider's rate
limits, so it was not re-run.

**Single-provider risk.** Because the first two models shared one provider, a single NVIDIA
outage or withdrawal could disable both. The mitigations recorded were:

- the emergency fallback from another provider;
- the local coaching model;
- the template floor, which needs no network;
- no change to the model ids during study weeks.

The owner's preference for a provider that does not train on prompts could not be applied,
because only NVIDIA models passed the first run.

**Privacy.** On 2026-11-13 the project owner decided to allow free endpoints that log or train
on prompts. This was a documented design choice, made so that more capable free models could be
used at no cost.

- In exchange, participants are told in consent switch (d) that the provider stores the
  transcript text and may use it to improve its models. All three chosen providers can train on
  prompts, so this statement applies to every model in the chain.
- Declining switch (d) turns the feature off, and the application uses templates.
- The protections kept are: text only; personal names redacted; no media; no identifiers; no
  `user` field.
- Speak & Improve data is still never sent to OpenRouter (`docs/plan/FINAL_07_DATA_SOURCES.md`).

**Items left open at the time of writing:**

- **NVIDIA trial terms.** These terms limit use to trial and evaluation without production use.
  The owner's working reading was that a private, non-commercial university research prototype
  with volunteer participants is evaluation. This reading was pending the supervisor's
  confirmation. If it is not confirmed, the NVIDIA models leave the chain and Flint becomes the
  pinned model.
- **T2 teacher.** No provider was confirmed as allowing its outputs to be used as training data
  for the later fine-tuning task T2. The option proposed for phase P4 was to generate the
  teacher data on Kaggle with an open-weight model under an Apache-2.0 or MIT licence. Only the
  model licence would then apply, and no provider's terms.
- **Arabic ratings.** The owner's 1–5 ratings of the Arabic wording had not yet been returned.

## 6. Limitations

- All transcripts were synthetic. Real practice talks may be longer or less fluent.
- Each model was run once on each prompt, so run-to-run variation was not estimated.
- The re-run changed two settings at the same time, the output limit and the reasoning setting,
  so their separate effects are not known.
- Only the models that failed on output limits were re-run. The other models were compared under
  the first-run settings.
- Free endpoints, their rate limits and their access rules can change without notice, as the
  Inkling result shows.
- The Gemma result reflects the provider's rate limits on the day of the run, not the model's
  output quality.
- Five Arabic prompts were too few for a pass decision. The owner's wording ratings are recorded
  separately in the two rating sheets, `20261113-143329_ar_rating_sheet.csv` and
  `20261113-211500_ar_rating_sheet.csv`.
- The payload and consent rules were tested in the spike code only. Product-level tests are
  planned for phase P7.

## Sources

- **Results**, in `experiments/results/s4/`:
  - first run: `20261113-143329_requests.csv`, `_replies.jsonl`, `_summary.csv`,
    `_summary_rescored.csv` and `_ar_rating_sheet.csv`;
  - re-run: `20261113-211500_requests.csv`, `_replies.jsonl`, `_summary.csv` and
    `_ar_rating_sheet.csv`;
  - provider policies: `20261113-143154_policies.csv`.
- **Spike code and terms review:** `experiments/s4_openrouter/`, with unit tests in
  `tests/unit/test_s4_core.py`.
- **Settings:** `experiments/configs/s4_openrouter.yaml` and `experiments/configs/s4_rerun.yaml`.
- **Configuration:** `configs/models.yaml` (`llm.openrouter`).
- **Decision record:** `docs/adr/0003-openrouter-model-and-privacy.md`.
- **Commits:** `de10057` (privacy decision), `fc2c432` (spike harness), `7594412` (first-run
  results), and the re-run commit on branch `spike/s4-openrouter`.
