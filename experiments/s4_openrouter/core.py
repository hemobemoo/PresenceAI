"""Spike S4 logic without network: prompts, redaction, payload rules, reply checks, summary.

Unit-tested in tests/unit/test_s4_core.py. The network driver is run.py.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from presenceai.coach.guardrails import (
    _normalise_ar,
    contains_banned_term,
    latin_outside_transcript,
)

Lang = Literal["en", "ar"]
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = REPO / "experiments" / "configs" / "s4_openrouter.yaml"
NAME_TOKEN = "[NAME]"  # noqa: S105 (placeholder text, not a secret)
LONG_PAUSE_S = 1.5  # pause length named in the prompt; an allowed number in replies

SYSTEM_PROMPT = (
    "You are a presentation coach. You receive the transcript of a short practice talk and "
    "measurements of it. Write exactly 3 feedback items.\n"
    "Rules:\n"
    "1. Describe only observable behaviour: words, pace, pauses, filler words and structure. "
    "Never mention or guess emotions, mood, personality, confidence, honesty, intelligence or "
    "any other trait of the speaker.\n"
    "2. Each item quotes its evidence copied word for word from the transcript: one contiguous "
    "span of 4 to 15 words. Do not correct, shorten, paraphrase or translate the quoted words.\n"
    "3. Use only numbers that appear in the measurements.\n"
    "4. Give one concrete suggestion per item.\n"
    "{language_rule}"
    "Return only a JSON object and no other text, in this form:\n"
    '{{"items": [{{"observation": "...", "evidence_quote": "...", "suggestion": "..."}}]}}'
)
LANGUAGE_RULE = {
    "en": "",
    "ar": (
        "5. Write observation and suggestion in Modern Standard Arabic. The evidence_quote stays "
        "in the transcript's own Arabic words.\n"
    ),
}
ALLOWED_PAYLOAD_KEYS = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "response_format",
    "reasoning",
    "provider",
}


# ---------------------------------------------------------------- prompts and measurements


@dataclass(frozen=True)
class Prompt:
    id: str
    lang: Lang
    topic: str
    duration_s: float
    long_pauses: int
    transcript: str
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class Measures:
    duration_s: float
    words: int
    wpm: int
    fillers: int
    fillers_per_min: float
    long_pauses: int

    def numbers(self) -> set[str]:
        vals = {
            self.duration_s,
            self.words,
            self.wpm,
            self.fillers,
            self.fillers_per_min,
            self.long_pauses,
            LONG_PAUSE_S,
        }
        out: set[str] = set()
        for v in vals:
            out.add(_num(float(v)))
        return out


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data  # pyright: ignore[reportUnknownVariableType]


def load_prompts(path: Path) -> list[Prompt]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    lang = data["lang"]
    out = [
        Prompt(
            id=str(p["id"]),
            lang=lang,
            topic=str(p["topic"]),
            duration_s=float(p["duration_s"]),
            long_pauses=int(p["long_pauses"]),
            transcript=" ".join(str(p["transcript"]).split()),
            names=tuple(p.get("names") or ()),
        )
        for p in data["prompts"]
    ]
    if len({p.id for p in out}) != len(out):
        raise ValueError(f"duplicate prompt id in {path}")
    return out


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[\w\[\]']+", text, flags=re.UNICODE)


def count_fillers(text: str, fillers: Sequence[str], lang: Lang) -> int:
    norm = normalise(text, lang)
    padded = f" {norm} "
    return sum(padded.count(f" {normalise(f, lang)} ") for f in fillers)


def compute_measures(p: Prompt, fillers: Sequence[str]) -> Measures:
    words = len(_word_tokens(p.transcript))
    minutes = p.duration_s / 60
    n_fill = count_fillers(p.transcript, fillers, p.lang)
    return Measures(
        duration_s=p.duration_s,
        words=words,
        wpm=round(words / minutes),
        fillers=n_fill,
        fillers_per_min=round(n_fill / minutes, 1),
        long_pauses=p.long_pauses,
    )


# ---------------------------------------------------------------- redaction and payload


def redact_names(text: str, names: Iterable[str]) -> str:
    """Replace each personal name (full form first, then each part) with [NAME]."""
    parts: list[str] = []
    for name in names:
        parts.append(name)
        parts.extend(t for t in name.split() if len(t) >= 3)
    for part in sorted(set(parts), key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(part)}(?!\w)", NAME_TOKEN, text, flags=re.IGNORECASE)
    return re.sub(rf"{re.escape(NAME_TOKEN)}(\s+{re.escape(NAME_TOKEN)})+", NAME_TOKEN, text)


def user_message(transcript: str, m: Measures) -> str:
    return (
        "Measurements:\n"
        f"- duration: {_num(m.duration_s)} s\n"
        f"- words: {m.words}\n"
        f"- speech rate: {m.wpm} words per minute\n"
        f"- filler words: {m.fillers} ({_num(m.fillers_per_min)} per minute)\n"
        f"- pauses longer than {_num(LONG_PAUSE_S)} s: {m.long_pauses}\n\n"
        f"Transcript:\n{transcript}"
    )


def request_for(cfg: dict[str, Any], model: str) -> dict[str, Any]:
    """Request settings for one model: `request`, with keys replaced by `model_overrides[model]`."""
    return {**cfg["request"], **(cfg.get("model_overrides") or {}).get(model, {})}


def build_payload(
    model: str, p: Prompt, m: Measures, request_cfg: dict[str, Any]
) -> dict[str, Any]:
    """The full request body. Text only; names redacted; no `user` field (ADR 0003)."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(language_rule=LANGUAGE_RULE[p.lang]),
            },
            {"role": "user", "content": user_message(redact_names(p.transcript, p.names), m)},
        ],
        **request_cfg,
    }
    check_payload(payload, p.names)
    return payload


class PayloadError(ValueError):
    pass


def check_payload(payload: dict[str, Any], names: Iterable[str] = ()) -> None:
    """Guard: only allowed keys, no `user` field, text-only messages, no names, data_collection."""
    extra = set(payload) - ALLOWED_PAYLOAD_KEYS
    if extra:
        raise PayloadError(f"keys not allowed in payload: {sorted(extra)}")
    for msg in payload["messages"]:
        if set(msg) != {"role", "content"} or not isinstance(msg["content"], str):
            raise PayloadError("messages must be {role, content} with text content only")
        for name in names:
            for part in [name, *name.split()]:
                if len(part) >= 3 and re.search(
                    rf"(?<!\w){re.escape(part)}(?!\w)", msg["content"], flags=re.IGNORECASE
                ):
                    raise PayloadError("a personal name is present in the payload")
    if (payload.get("provider") or {}).get("data_collection") != "allow":
        raise PayloadError("provider.data_collection must be 'allow' (ADR 0003)")


# ---------------------------------------------------------------- reply checks


class FeedbackItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    observation: str
    evidence_quote: str
    suggestion: str

    @field_validator("observation", "evidence_quote", "suggestion")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("empty")
        return v


class FeedbackReply(BaseModel):
    model_config = ConfigDict(extra="ignore")
    items: list[FeedbackItem]


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


@dataclass
class ReplyCheck:
    strict_json: bool = False  # content parses as JSON as returned
    json_valid: bool = False  # parses (code fence allowed) and matches the schema
    error: str = ""
    items: list[FeedbackItem] = field(default_factory=list[FeedbackItem])


def check_reply(content: str | None, items_min: int, items_max: int) -> ReplyCheck:
    r = ReplyCheck()
    if not content or not content.strip():
        r.error = "empty content"
        return r
    obj: Any = None
    try:
        obj = json.loads(content)
        r.strict_json = True
    except json.JSONDecodeError:
        m = _FENCE.match(content)
        if m is None:
            r.error = "not JSON"
            return r
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            r.error = "not JSON inside code fence"
            return r
    try:
        reply = FeedbackReply.model_validate(obj)
    except ValidationError as e:
        r.error = f"schema: {e.errors()[0]['type']} at {e.errors()[0]['loc']}"
        return r
    if not items_min <= len(reply.items) <= items_max:
        r.error = f"schema: {len(reply.items)} items"
        return r
    r.json_valid = True
    r.items = reply.items
    return r


_QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "،": ",", "؛": ";"})


def normalise(text: str, lang: Lang) -> str:
    t = text.translate(_QUOTE_CHARS).casefold()
    if lang == "ar":
        t = _normalise_ar(t)
    t = re.sub(r"[^\w\s']", " ", t, flags=re.UNICODE)
    t = t.replace("'", "")
    return " ".join(t.split())


def quote_found(quote: str, transcript: str, lang: Lang, min_words: int) -> bool:
    q = normalise(quote, lang)
    if len(q.split()) < min_words:
        return False
    return f" {q} " in f" {normalise(transcript, lang)} "


_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def numbers_ok(text: str, m: Measures) -> bool:
    """Every number in the text is one of the measurements (secondary check, not in PASS)."""
    allowed = m.numbers()
    tokens = _NUMBER.findall(text.translate(_AR_DIGITS))
    return all(_num(float(tok)) in allowed for tok in tokens)


def latin_quote(quote: str, sent_transcript: str, lang: Lang) -> bool:
    """Arabic only: the quote holds Latin-script text that is not in the transcript (ADR 0003)."""
    return lang == "ar" and bool(latin_outside_transcript(quote, sent_transcript))


def item_has_banned_term(item: FeedbackItem, lang: Lang) -> bool:
    return contains_banned_term(f"{item.observation} {item.suggestion}", lang)


# ---------------------------------------------------------------- summary


def _p(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def summarise(
    rows: Sequence[dict[str, Any]], min_json: float, min_quote: float
) -> list[dict[str, Any]]:
    """One summary row per (model, lang). PASS is judged on English rows only."""
    out: list[dict[str, Any]] = []
    keys = sorted({(r["model"], r["lang"]) for r in rows})
    for model, lang in keys:
        rs = [r for r in rows if r["model"] == model and r["lang"] == lang]
        n = len(rs)
        valid = [r for r in rs if r["json_valid"]]
        quotes = sum(int(r["n_quotes"]) for r in valid)
        found = sum(int(r["n_quotes_found"]) for r in valid)
        lat = [float(r["latency_ms"]) for r in rs if r["latency_ms"] not in ("", None)]
        json_rate = len(valid) / n if n else 0.0
        quote_rate = found / quotes if quotes else 0.0
        providers = sorted({str(r["provider"]) for r in rs if r["provider"]})
        row: dict[str, Any] = {
            "model": model,
            "lang": lang,
            "n": n,
            "http_ok": sum(1 for r in rs if str(r["http_status"]) == "200"),
            "retried": sum(1 for r in rs if int(r["attempts"]) > 1),
            "strict_json_rate": round(sum(1 for r in rs if r["strict_json"]) / n, 3) if n else 0,
            "json_valid_rate": round(json_rate, 3),
            "quotes": quotes,
            "quote_found_rate": round(quote_rate, 3),
            "all_quotes_found_rate": round(
                sum(1 for r in valid if r["n_quotes_found"] == r["n_quotes"]) / n, 3
            )
            if n
            else 0,
            "numbers_ok_rate": round(sum(1 for r in valid if r["numbers_ok"]) / len(valid), 3)
            if valid
            else 0,
            "banned_term_replies": sum(1 for r in valid if r["banned_term"]),
            "latin_quote_replies": sum(1 for r in valid if r.get("latin_quote") in (True, "True")),
            "finish_length": sum(1 for r in rs if r["finish_reason"] == "length"),
            "latency_ms_p50": _p(lat, 0.5),
            "latency_ms_p95": _p(lat, 0.95),
            "latency_ms_mean": round(statistics.fmean(lat), 1) if lat else None,
            "providers": ";".join(providers),
            "passed": "",
        }
        if lang == "en":
            row["passed"] = json_rate >= min_json and quote_rate >= min_quote
        out.append(row)
    return out


# ---------------------------------------------------------------- provider data policy


@dataclass(frozen=True)
class Policy:
    model: str
    provider: str
    training: bool | None
    training_openrouter: bool | None
    retains_prompts: bool | None
    retention_days: int | None
    requires_user_ids: bool | None
    terms_url: str
    privacy_url: str


def parse_policy(html: str, model: str) -> Policy | None:
    """Data policy of the endpoint serving `model` (exact variant slug) from its OpenRouter page."""
    text = html.replace('\\"', '"')
    for m in re.finditer(r'"data_policy":(\{[^{}]*\})', text):
        back = text[max(0, m.start() - 8000) : m.start()]
        slugs = re.findall(r'"model_variant_slug":"([^"]+)"', back)
        names = re.findall(r'"provider_name":"([^"]+)"', back)
        if not slugs or slugs[-1] != model:
            continue
        d = json.loads(m.group(1))
        return Policy(
            model=model,
            provider=names[-1] if names else "",
            training=d.get("training"),
            training_openrouter=d.get("trainingOpenRouter"),
            retains_prompts=d.get("retainsPrompts"),
            retention_days=d.get("retentionDays"),
            requires_user_ids=d.get("requiresUserIDs"),
            terms_url=d.get("termsOfServiceURL", ""),
            privacy_url=d.get("privacyPolicyURL", ""),
        )
    return None


# ---------------------------------------------------------------- Arabic ratings


def rating_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per model: replies, ratings on the 1-5 scale and their mean. Other values (blank, 0,
    text) count as not rated; the owner used 0 for "no reply to rate"."""
    out: list[dict[str, Any]] = []
    for model in sorted({str(r["model"]) for r in rows}):
        vals: list[int] = []
        rs = [r for r in rows if r["model"] == model]
        for r in rs:
            v = str(r.get("rating_1_to_5", "")).strip()
            if v.isdigit() and 1 <= int(v) <= 5:
                vals.append(int(v))
        out.append(
            {
                "model": model,
                "n_replies": len(rs),
                "n_rated": len(vals),
                "mean_rating": round(statistics.fmean(vals), 2) if vals else None,
            }
        )
    return out
