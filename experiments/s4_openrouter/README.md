# Spike S4 — OpenRouter free models

Question, pass rule and decision: `docs/adr/0003-openrouter-model-and-privacy.md`.
Report draft: `docs/report-drafts/S4_summary.md`.

| File | Role |
|---|---|
| `core.py` | Prompts, measurements, name redaction, payload guard, reply schema, quote check, summary, data-policy parser (unit-tested in `tests/unit/test_s4_core.py`) |
| `run.py` | Sends every prompt to every shortlisted model and writes the results |
| `ratings.py` | Unblinds the owner's Arabic rating sheets with their keys; per-model mean and n rated |
| `policies.py` | Records each model's provider data policy from its public OpenRouter page |
| `prompts_en.yaml`, `prompts_ar.yaml` | 30 English and 5 Arabic **synthetic** practice-talk transcripts with fictional names |
| `terms_review.md` | Licence and provider terms on training with outputs (T2 teacher) and on prompt use |
| `../configs/s4_openrouter.yaml` | Shortlist, request settings, pass thresholds (FINAL_12 §1); run 20261113-143329 |
| `../configs/s4_rerun.yaml` | Re-run of Flint and dots with a higher output cap and reasoning off/minimal; run 20261113-211500 |

What is sent: text only (a fixed system prompt, measurements and the transcript with fictional
names replaced by `[NAME]`), no media, no ids and no `user` field, with
`provider.data_collection: "allow"`. `core.check_payload` refuses any other payload. The API key
is read from `.env` through `presenceai.settings` and is never printed or written.

## Commands

```bash
uv run python experiments/s4_openrouter/run.py --dry-run   # payload checks only, no network
uv run python experiments/s4_openrouter/policies.py        # provider data policies
uv run python experiments/s4_openrouter/run.py             # full run, ~245 requests
uv run python experiments/s4_openrouter/run.py --config experiments/configs/s4_rerun.yaml  # 70 requests
uv run python experiments/s4_openrouter/run.py --rescore 20261113-143329  # re-check saved replies, no network
uv run python experiments/s4_openrouter/ratings.py 20261113-143329 20261113-211500  # after rating
```

## Arabic rating

There is one sheet per run (`20261113-143329` and `20261113-211500`; the letter codes differ
between sheets). `experiments/results/s4/<run>_ar_rating_sheet.csv` lists every Arabic reply under a letter code
(A–G, or A–B in the re-run), not the model name. Fill `rating_1_to_5` for wording (1 = unusable, 3 = understandable but
awkward, 5 = natural Modern Standard Arabic) and optional `notes`. Open
`<run>_ar_rating_key.csv` only after rating.
