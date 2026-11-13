# Spike S4 — OpenRouter free models

Question, pass rule and decision: `docs/adr/0003-openrouter-model-and-privacy.md`.
Report draft: `docs/report-drafts/S4_summary.md`.

| File | Role |
|---|---|
| `core.py` | Prompts, measurements, name redaction, payload guard, reply schema, quote check, summary, data-policy parser (unit-tested in `tests/unit/test_s4_core.py`) |
| `run.py` | Sends every prompt to every shortlisted model and writes the results |
| `policies.py` | Records each model's provider data policy from its public OpenRouter page |
| `prompts_en.yaml`, `prompts_ar.yaml` | 30 English and 5 Arabic **synthetic** practice-talk transcripts with fictional names |
| `terms_review.md` | Licence and provider terms on training with outputs (T2 teacher) and on prompt use |
| `../configs/s4_openrouter.yaml` | Shortlist, request settings, pass thresholds (FINAL_12 §1) |

What is sent: text only (a fixed system prompt, measurements and the transcript with fictional
names replaced by `[NAME]`), no media, no ids and no `user` field, with
`provider.data_collection: "allow"`. `core.check_payload` refuses any other payload. The API key
is read from `.env` through `presenceai.settings` and is never printed or written.

## Commands

```bash
uv run python experiments/s4_openrouter/run.py --dry-run   # payload checks only, no network
uv run python experiments/s4_openrouter/policies.py        # provider data policies
uv run python experiments/s4_openrouter/run.py             # full run, ~245 requests
```

## Arabic rating

`experiments/results/s4/<run>_ar_rating_sheet.csv` lists every Arabic reply under a letter code
(A–G), not the model name. Fill `rating_1_to_5` for wording (1 = unusable, 3 = understandable but
awkward, 5 = natural Modern Standard Arabic) and optional `notes`. Open
`<run>_ar_rating_key.csv` only after rating.
