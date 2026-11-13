"""Tests for experiments/s4_openrouter/core.py: payload guard, redaction, reply checks, summary."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SPIKE = REPO / "experiments" / "s4_openrouter"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("s4_core", SPIKE / "core.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["s4_core"] = module
    spec.loader.exec_module(module)
    return module


core = _load()
CFG = core.load_config()


def _prompt(**kw: Any) -> Any:
    base: dict[str, Any] = {
        "id": "t1",
        "lang": "en",
        "topic": "test",
        "duration_s": 60.0,
        "long_pauses": 2,
        "transcript": "Hello, my name is Lina Haddad. Um, today I will talk about uh rivers.",
        "names": ("Lina Haddad",),
    }
    base.update(kw)
    return core.Prompt(**base)


# ---- prompts and measurements


@pytest.mark.parametrize("lang,count", [("en", 30), ("ar", 5)])
def test_prompt_files_load(lang: str, count: int):
    ps = core.load_prompts(SPIKE / f"prompts_{lang}.yaml")
    assert len(ps) == count
    assert all(p.lang == lang for p in ps)


def test_measures_are_computed_from_transcript():
    m = core.compute_measures(_prompt(), CFG["fillers"]["en"])
    assert m.words == 14
    assert m.wpm == 14
    assert m.fillers == 2
    assert m.long_pauses == 2


def test_arabic_fillers_counted():
    p = _prompt(lang="ar", transcript="يعني المشروع تطبيق. امم لذلك يعني نعم", names=())
    assert core.compute_measures(p, CFG["fillers"]["ar"]).fillers == 3


# ---- redaction and payload guard (ADR 0003)


def test_names_redacted_full_and_parts():
    text = "Lina Haddad spoke. Later Lina and Dr. Haddad answered."
    out = core.redact_names(text, ["Lina Haddad"])
    assert "Lina" not in out and "Haddad" not in out
    assert out.count("[NAME]") == 3


def test_payload_is_text_only_redacted_and_has_no_user_field():
    p = _prompt()
    m = core.compute_measures(p, CFG["fillers"]["en"])
    payload = core.build_payload("org/model:free", p, m, CFG["request"])
    assert "user" not in payload
    assert payload["provider"]["data_collection"] == "allow"
    assert all(isinstance(msg["content"], str) for msg in payload["messages"])
    body = json.dumps(payload)
    assert "Lina" not in body and "Haddad" not in body


def test_every_shipped_prompt_builds_a_valid_payload():
    for lang in ("en", "ar"):
        for p in core.load_prompts(SPIKE / f"prompts_{lang}.yaml"):
            m = core.compute_measures(p, CFG["fillers"][lang])
            core.build_payload("org/model:free", p, m, CFG["request"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda pl: pl.update(user="abc"),
        lambda pl: pl["messages"][1].update(content=[{"type": "image_url", "image_url": "x"}]),
        lambda pl: pl["messages"][1].update(content="Lina said hi"),
        lambda pl: pl.update(provider={"data_collection": "deny"}),
    ],
    ids=["user-field", "non-text", "name-present", "data-collection-deny"],
)
def test_check_payload_rejects(mutate: Any):
    p = _prompt()
    m = core.compute_measures(p, CFG["fillers"]["en"])
    payload = core.build_payload("org/model:free", p, m, CFG["request"])
    mutate(payload)
    with pytest.raises(core.PayloadError):
        core.check_payload(payload, p.names)


# ---- reply checks


ITEM = {"observation": "o", "evidence_quote": "q", "suggestion": "s"}


def test_reply_strict_json_valid():
    r = core.check_reply(json.dumps({"items": [ITEM, ITEM, ITEM]}), 2, 4)
    assert r.strict_json and r.json_valid


def test_reply_in_code_fence_is_valid_but_not_strict():
    r = core.check_reply("```json\n" + json.dumps({"items": [ITEM, ITEM]}) + "\n```", 2, 4)
    assert r.json_valid and not r.strict_json


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "Here is my feedback: ...",
        json.dumps({"items": [ITEM]}),
        json.dumps({"items": [{**ITEM, "suggestion": " "}] * 3}),
        json.dumps({"feedback": [ITEM] * 3}),
    ],
)
def test_reply_invalid(content: str | None):
    assert not core.check_reply(content, 2, 4).json_valid


def test_quote_found_ignores_case_punctuation_and_quote_marks():
    t = "Um, today I’ll talk about rivers, and why they flood."
    assert core.quote_found("today i'll talk about RIVERS and why", t, "en", 3)
    assert not core.quote_found("today I will talk about rivers", t, "en", 3)
    assert not core.quote_found("about rivers", t, "en", 3)  # too short


def test_quote_found_arabic_normalised():
    t = "يعني المشروعُ تطبيقٌ يساعد الطلاب، على إيجاد القاعات"
    assert core.quote_found("المشروع تطبيق يساعد الطلاب على ايجاد", t, "ar", 3)


def test_quote_found_against_redacted_text():
    p = _prompt()
    sent = core.redact_names(p.transcript, p.names)
    assert core.quote_found("my name is [NAME]", sent, "en", 3)


def test_numbers_ok():
    m = core.compute_measures(_prompt(), CFG["fillers"]["en"])
    assert core.numbers_ok("You used 2 fillers at 14 words per minute.", m)
    assert not core.numbers_ok("You used 7 fillers.", m)


def test_pause_length_named_in_prompt_is_an_allowed_number():
    m = core.compute_measures(_prompt(), CFG["fillers"]["en"])
    assert f"{core.LONG_PAUSE_S} s" in core.user_message("x", m)
    assert core.numbers_ok(f"No pause was longer than {core.LONG_PAUSE_S} seconds.", m)


# ---- summary


def _row(model: str, valid: bool, quotes: int, found: int, lang: str = "en") -> dict[str, Any]:
    return {
        "model": model,
        "lang": lang,
        "http_status": 200,
        "attempts": 1,
        "strict_json": valid,
        "json_valid": valid,
        "n_quotes": quotes if valid else 0,
        "n_quotes_found": found if valid else 0,
        "numbers_ok": True,
        "banned_term": False,
        "finish_reason": "stop",
        "latency_ms": 1000.0,
        "provider": "P",
    }


def test_summary_pass_rule():
    rows = [_row("a", True, 3, 3)] * 19 + [_row("a", False, 0, 0)]
    rows += [_row("b", True, 3, 2)] * 20
    s = {r["model"]: r for r in core.summarise(rows, 0.95, 0.90)}
    assert s["a"]["json_valid_rate"] == 0.95 and s["a"]["passed"] is True
    assert s["b"]["quote_found_rate"] == pytest.approx(0.667, abs=1e-3)
    assert s["b"]["passed"] is False


def test_summary_arabic_rows_have_no_pass_verdict():
    s = core.summarise([_row("a", True, 3, 3, lang="ar")], 0.95, 0.90)
    assert s[0]["passed"] == ""


# ---- provider data policy parsing


def test_parse_policy_picks_the_free_variant():
    def ep(slug: str, name: str, training: bool) -> str:
        pol = json.dumps({"training": training, "retainsPrompts": True, "retentionDays": 30})
        return f'"model_variant_slug":"{slug}","provider_name":"{name}","data_policy":{pol}'

    html = ("{" + ep("org/m", "Paid", False) + "}{" + ep("org/m:free", "Free", True) + "}").replace(
        '"', '\\"'
    )
    pol = core.parse_policy(html, "org/m:free")
    assert pol is not None
    assert pol.provider == "Free" and pol.training is True and pol.retention_days == 30
    assert core.parse_policy(html, "org/other:free") is None
