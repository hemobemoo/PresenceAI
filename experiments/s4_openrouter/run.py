"""Spike S4 driver: send the fixed prompts to each shortlisted OpenRouter free model and score them.

    uv run python experiments/s4_openrouter/run.py --dry-run   # build and check payloads only
    uv run python experiments/s4_openrouter/run.py                    # full run (~245 requests)
    uv run python experiments/s4_openrouter/run.py --models google/gemma-4-31b-it:free --limit 2

The key comes from OPENROUTER_API_KEY via presenceai.settings (read from .env); it is never
printed, logged or written. Payloads are text only, personal names are redacted, there is no
`user` field, and provider.data_collection is "allow" (ADR 0003). core.check_payload enforces this
before every request.

Outputs in --out (default experiments/results/s4):
    <run>_requests.csv          one row per request (status, provider, latency, checks)
    <run>_replies.jsonl         raw reply text per request (synthetic inputs, model outputs only)
    <run>_summary.csv           one row per model and language; PASS on English rows
    <run>_ar_rating_sheet.csv   blinded Arabic replies for the owner to rate 1-5
    <run>_ar_rating_key.csv     code -> model mapping for the rating sheet (open after rating)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core

from presenceai.settings import get_settings

REQUEST_FIELDS = [
    "run_id",
    "model",
    "lang",
    "prompt_id",
    "attempts",
    "http_status",
    "error",
    "provider",
    "finish_reason",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "strict_json",
    "json_valid",
    "schema_error",
    "n_items",
    "n_quotes",
    "n_quotes_found",
    "numbers_ok",
    "banned_term",
]


def _post(
    url: str, key: str, payload: dict[str, Any], timeout: float
) -> tuple[int, dict[str, Any], str]:
    req = urllib.request.Request(  # noqa: S310 (fixed https URL from the config)
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": "PresenceAI spike S4",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
            return resp.status, json.loads(resp.read().decode("utf-8")), ""
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            msg = json.loads(body).get("error", {}).get("message", "")
        except (json.JSONDecodeError, AttributeError):
            msg = body[:200]
        return e.code, {}, str(msg)[:200]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return 0, {}, type(e).__name__


def _score(
    body: dict[str, Any], p: core.Prompt, m: core.Measures, cfg: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    choice = (body.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    usage = body.get("usage") or {}
    chk = core.check_reply(content, cfg["items_min"], cfg["items_max"])
    sent = core.redact_names(p.transcript, p.names)
    found = [
        core.quote_found(i.evidence_quote, sent, p.lang, cfg["quote_min_words"]) for i in chk.items
    ]
    row = {
        "provider": body.get("provider", ""),
        "finish_reason": choice.get("finish_reason", ""),
        "prompt_tokens": usage.get("prompt_tokens", ""),
        "completion_tokens": usage.get("completion_tokens", ""),
        "strict_json": chk.strict_json,
        "json_valid": chk.json_valid,
        "schema_error": chk.error,
        "n_items": len(chk.items),
        "n_quotes": len(chk.items),
        "n_quotes_found": sum(found),
        "numbers_ok": all(core.numbers_ok(f"{i.observation} {i.suggestion}", m) for i in chk.items)
        if chk.json_valid
        else "",
        "banned_term": any(core.item_has_banned_term(i, p.lang) for i in chk.items),
    }
    return row, content


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", help="comma-separated; default: shortlist in the config")
    ap.add_argument("--lang", default="en,ar")
    ap.add_argument("--limit", type=int, help="first N prompts per language")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=core.REPO / "experiments" / "results" / "s4")
    args = ap.parse_args(argv)

    cfg = core.load_config()
    models = args.models.split(",") if args.models else list(cfg["models"])
    langs = [lang.strip() for lang in args.lang.split(",")]
    prompts: list[core.Prompt] = []
    for lang in langs:
        ps = core.load_prompts(core.HERE / f"prompts_{lang}.yaml")
        n = cfg["arabic_prompts"] if lang == "ar" else len(ps)
        prompts.extend(ps[: min(n, args.limit or n)])

    jobs: list[tuple[core.Prompt, core.Measures, str, dict[str, Any]]] = []
    for p in prompts:  # round robin: every model sees each prompt at about the same time
        m = core.compute_measures(p, cfg["fillers"][p.lang])
        for model in models:
            jobs.append((p, m, model, core.build_payload(model, p, m, cfg["request"])))
    print(
        f"{len(jobs)} requests: {len(models)} models x {len(prompts)} prompts ({','.join(langs)})"
    )
    if args.dry_run:
        print("dry run: all payloads passed check_payload; nothing sent")
        return 0

    secret = get_settings().openrouter_api_key
    if secret is None or not secret.get_secret_value():
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2
    key = secret.get_secret_value()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)
    req_path = args.out / f"{run_id}_requests.csv"
    rep_path = args.out / f"{run_id}_replies.jsonl"
    rows: list[dict[str, Any]] = []
    replies: list[dict[str, Any]] = []
    last_start = 0.0
    with (
        req_path.open("w", newline="", encoding="utf-8") as fcsv,
        rep_path.open("w", encoding="utf-8") as fjson,
    ):
        w = csv.DictWriter(fcsv, fieldnames=REQUEST_FIELDS)
        w.writeheader()
        for k, (p, m, model, payload) in enumerate(jobs, 1):
            attempts, status, body, err, latency = 0, 0, {}, "", None
            while attempts < 2:
                wait = cfg["interval_s"] - (time.monotonic() - last_start)
                if wait > 0:
                    time.sleep(wait)
                last_start = time.monotonic()
                attempts += 1
                t0 = time.perf_counter()
                status, body, err = _post(cfg["endpoint"], key, payload, cfg["timeout_s"])
                latency = round((time.perf_counter() - t0) * 1000, 1)
                if status == 200 and not body.get("error"):
                    break
                if body.get("error"):
                    err = str(body["error"].get("message", ""))[:200]
                if attempts < 2 and (status in cfg["retry_statuses"] or status == 0):
                    time.sleep(cfg["retry_wait_s"])
                    continue
                break
            row: dict[str, Any] = dict.fromkeys(REQUEST_FIELDS, "")
            row.update(
                run_id=run_id,
                model=model,
                lang=p.lang,
                prompt_id=p.id,
                attempts=attempts,
                http_status=status,
                error=err,
                latency_ms=latency,
                strict_json=False,
                json_valid=False,
                n_quotes=0,
                n_quotes_found=0,
                banned_term=False,
            )
            content = None
            if status == 200 and not body.get("error"):
                scored, content = _score(body, p, m, cfg)
                row.update(scored)
            w.writerow(row)
            fcsv.flush()
            rec: dict[str, Any] = {
                "run_id": run_id,
                "model": model,
                "lang": p.lang,
                "prompt_id": p.id,
                "content": content,
            }
            fjson.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fjson.flush()
            rows.append(row)
            replies.append(rec)
            print(
                f"[{k}/{len(jobs)}] {model} {p.id}: HTTP {status} "
                f"valid={row['json_valid']} quotes={row['n_quotes_found']}/{row['n_quotes']}"
                + (f" err={err[:80]}" if err else "")
            )

    summary = core.summarise(rows, cfg["pass"]["min_json_valid"], cfg["pass"]["min_quote_found"])
    sum_path = args.out / f"{run_id}_summary.csv"
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    _write_rating_sheet(args.out, run_id, [r for r in replies if r["lang"] == "ar"], models)
    for s in summary:
        print(
            f"{s['model']:45s} {s['lang']} n={s['n']} json={s['json_valid_rate']} "
            f"quote={s['quote_found_rate']} pass={s['passed']}"
        )
    print(f"wrote {req_path.name}, {rep_path.name}, {sum_path.name}")
    return 0


def _write_rating_sheet(
    out: Path, run_id: str, replies: list[dict[str, Any]], models: list[str]
) -> None:
    if not replies:
        return
    codes = [chr(ord("A") + i) for i in range(len(models))]
    random.Random(run_id).shuffle(codes)  # noqa: S311 (blinding order, not security)
    # seeded by run id, so the key can be re-derived from the run id
    code_of = dict(zip(models, codes, strict=True))
    sheet = []
    for r in replies:
        text = ""
        chk = core.check_reply(r["content"], 1, 99)
        if chk.json_valid:
            text = "\n".join(
                f"{n}. {i.observation} — «{i.evidence_quote}» — {i.suggestion}"
                for n, i in enumerate(chk.items, 1)
            )
        elif r["content"]:
            text = f"(not valid JSON) {r['content'][:1500]}"
        else:
            text = "(no reply)"
        sheet.append(
            {"prompt_id": r["prompt_id"], "model_code": code_of[r["model"]], "reply": text}
        )
    sheet.sort(key=lambda s: (s["prompt_id"], s["model_code"]))
    with (out / f"{run_id}_ar_rating_sheet.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f, fieldnames=["prompt_id", "model_code", "reply", "rating_1_to_5", "notes"]
        )
        w.writeheader()
        for s in sheet:
            w.writerow({**s, "rating_1_to_5": "", "notes": ""})
    with (out / f"{run_id}_ar_rating_key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_code", "model"])
        for model in models:
            w.writerow([code_of[model], model])


if __name__ == "__main__":
    raise SystemExit(main())
