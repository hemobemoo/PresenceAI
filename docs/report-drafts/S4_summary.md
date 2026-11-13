# Spike S4: OpenRouter free models for feedback wording

## 1. Aim

Spike S4 examined whether free models served through OpenRouter could turn the application's
measurements and a practice-talk transcript into structured feedback that the application can
check. Each feedback item had to quote the transcript verbatim, so that every statement carries
evidence. The spike was also used to choose one pinned model and two fallbacks for the optional
"AI wording" feature. The pass condition was taken from the project roadmap
(`docs/plan/FINAL_12_ROADMAP_AND_THESIS.md`, §1).

## 2. Method

Seven free models were shortlisted and approved by the project owner on 2026-11-13. They came from
six serving providers (Table 1). Each model received the same 30 English and 5 Arabic transcripts.
All transcripts were synthetic and written for the spike, and personal names in them were
fictional. Each request contained a fixed system prompt, five measurements (duration, word count,
speech rate, filler words and pauses longer than 1.5 s) and the transcript. Personal names were
replaced by `[NAME]` before sending.

Each request contained text only, no media, no identifiers and no `user` field, and set
`provider.data_collection` to `"allow"`. A guard function rejected any other payload before it was
sent. Requests used temperature 0.2, a 4,096-token output limit and low reasoning effort, with
reasoning text excluded from the reply. They were sent in turn to every model at least 4 s apart,
with one retry on rate-limit or server errors. The run (id `20261113-143329`) made 245 requests.

Each reply was checked automatically:

- JSON validity: the reply parsed as JSON and matched the reply schema of two to four items, each
  with an observation, an evidence quote and a suggestion.
- Quote found: each evidence quote of at least three words appeared verbatim in the transcript
  that was sent, after whitespace, case and punctuation normalisation.
- Two secondary checks were recorded but did not count towards the pass condition. The first
  tested whether every number in the reply was one of the measurements. The second tested
  whether the reply contained a term the project bans, for example emotion or personality words.

Each provider's data policy was recorded from its public OpenRouter model page
(`policies.py`, 2026-11-13). Its licence and terms of service were read for prompt use and for
training other models on outputs (`terms_review.md`).

## 3. Pass conditions

A model passed if, on the 30 English prompts, at least 95 % of replies were valid JSON and at
least 90 % of the quotes in valid replies were found in the transcript. The Arabic replies were
checked in the same way, but the five Arabic prompts were too few for a pass decision. Their
wording is rated separately by the owner (Section 6).

The owner set three further rules for the final choice:

- The pinned model and the two fallbacks must come from three different providers.
- At least one of them must be usable as the teacher model for the later fine-tuning task T2.
- When two models are close, a provider that does not train on prompts is preferred as the
  pinned model.

## 4. Results

Table 1 is taken from `experiments/results/s4/20261113-143329_summary.csv` and
`20261113-143154_policies.csv`.

**Table 1.** English results per model (n = 30 prompts each) and provider data policy.

| Model | Provider | JSON valid | Quotes found | Median latency (s) | Trains on prompts (flag) | Passed |
|---|---|---|---|---|---|---|
| nvidia/nemotron-3-ultra-550b-a55b:free | Nvidia | 1.000 | 1.000 | 27.2 | yes | yes |
| nvidia/nemotron-3.5-super-vl:free | Nvidia | 0.967 | 1.000 | 19.5 | yes | yes |
| recursive/flint-v1.0:free | Recursive | 0.833 | 1.000 | 21.4 | no | no |
| dots-studio/dots-3-note-preview:free | AtlasCloud | 0.400 | 1.000 | 47.4 | no | no |
| google/gemma-4-31b-it:free | Google AI Studio | 0.067 | 1.000 | 0.6 | no | no |
| liquid/lfm-3-1.7b:free | Liquid | 0.033 | 1.000 | 19.6 | yes | no |
| thinkingmachines/inkling-2-small:free | Thinking Machines | 0.000 | – | 0.3 | yes | no |

Every quote in a valid reply was found in the transcript, for all models. The failures had three
causes, recorded in `20261113-143329_requests.csv`:

- **Output limit.** The reply was cut off at the 4,096-token limit, usually with empty content,
  in 29 Liquid, 18 dots, 5 Flint and 1 Nemotron 3.5 Super VL English replies.
- **Rate limit.** Google AI Studio returned HTTP 429 on 28 of 30 English requests, even after
  the retry. Both replies it did serve were valid.
- **No API access.** Every Inkling request returned HTTP 403. The message said the model was
  available only through listed agent applications.

All seven providers retain prompts. OpenRouter flags Recursive as not training on prompts, but
Recursive's own model terms allow training on customer data. On training other models with the
outputs, Liquid's terms prohibit it, and the terms of the other providers could not be confirmed
either way.

In Arabic (n = 5), Nemotron 3 Ultra returned 5 valid replies and Nemotron 3.5 Super VL returned 4,
with all quotes found.

After the run, the secondary numbers check was found to reject the 1.5 s pause length stated in
the prompt. The check was corrected and the saved replies were re-checked without new requests
(`20261113-143329_summary_rescored.csv`). Only the secondary rate changed. For example, for
Nemotron 3 Ultra it rose from 0.000 to 0.467; most of the remaining numbers it rejected were
suggested target rates. The pass results did not change.

## 5. Decision

Only the two NVIDIA models met the pass condition. They share one provider, so the
three-provider rule could not be met with passing models alone. The following choice was
therefore proposed in ADR 0003 and is provisional until the owner decides:

- **Pinned:** `nvidia/nemotron-3-ultra-550b-a55b:free`. It had the highest JSON validity in
  English (1.000) and in Arabic (5 of 5).
- **Fallback 1:** `recursive/flint-v1.0:free`. All of its failures were output-limit cut-offs.
- **Fallback 2:** `google/gemma-4-31b-it:free`. All of its failures were upstream rate limits.

A re-run of the non-NVIDIA models with a higher output limit and slower pacing (about 105
requests) was recommended before the fallbacks are accepted.

The non-training preference could not be applied, because the NVIDIA provider trains on prompts.
None of the three chosen models could be confirmed as usable for T2. That rule remains open and
depends on how the "competing" clauses in the providers' terms are read.

The project owner also decided on 2026-11-13 to allow free endpoints that log or train on
prompts. This privacy decision was a documented design choice. It was made so that more capable
free models could be used at no cost. In exchange, participants are told in consent switch (d)
that the provider stores the transcript text and may use it to improve its models. Declining
switch (d) turns the feature off, and the application then uses templates. The protections kept
are:

- text only;
- personal names redacted;
- no media;
- no identifiers;
- no `user` field.

Speak & Improve data is still never sent to OpenRouter (`docs/plan/FINAL_07_DATA_SOURCES.md`).

## 6. Limitations

- All transcripts were synthetic. Real practice talks may be longer or less fluent.
- Each model was run once on each prompt, so run-to-run variation was not estimated.
- Free endpoints, their rate limits and their access rules can change without notice, as the
  Inkling result shows.
- The Gemma result reflects the provider's rate limits on the day of the run, not the model's
  output quality.
- The output limit and reasoning setting were the same for every model and were not tuned per
  model.
- Five Arabic prompts were too few for a pass decision. The owner's 1–5 wording ratings
  (`20261113-143329_ar_rating_sheet.csv`) had not been collected when this draft was written.
- The NVIDIA trial terms limit use to trial and evaluation without production use. Whether a
  user study falls within this has not been settled.
- The payload and consent rules were tested in the spike code only. Product-level tests are
  planned for phase P7.

## Sources

- Results: `experiments/results/s4/20261113-143329_requests.csv`, `_replies.jsonl`,
  `_summary.csv`, `_summary_rescored.csv`, `_ar_rating_sheet.csv`; `20261113-143154_policies.csv`.
- Spike code and terms review: `experiments/s4_openrouter/` (unit tests
  `tests/unit/test_s4_core.py`); settings `experiments/configs/s4_openrouter.yaml`.
- Configuration: `configs/models.yaml` (`llm.openrouter`).
- Decision record: `docs/adr/0003-openrouter-model-and-privacy.md`.
- Commits: `de10057` (privacy decision), `fc2c432` (spike harness); results commit on branch
  `spike/s4-openrouter`.
