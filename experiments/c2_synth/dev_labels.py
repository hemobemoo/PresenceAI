"""The owner's central-message labels on the C2 development part (ADR 0021 §4c).

    uv run python experiments/c2_synth/dev_labels.py draw            # writes the sheet once
    uv run python experiments/c2_synth/dev_labels.py status          # progress and flags
    uv run python experiments/c2_synth/dev_labels.py score           # after the sheet is done
    uv run python experiments/c2_synth/dev_labels.py score --partial # done items only
    uv run python experiments/c2_synth/dev_labels.py positions       # where main points sit
    uv run python experiments/c2_synth/dev_labels.py relabel draw    # owner consistency check
    uv run python experiments/c2_synth/dev_labels.py relabel status
    uv run python experiments/c2_synth/dev_labels.py relabel score

- **relabel** (ADR 0025 item 7): 20 random items of the finished sheet, new neutral keys and a
  new order, no first labels shown (`relabel_sheet.yaml`, `relabel_key.json`). Label them about
  a week after the first sheet, without opening `sheet.yaml`. `relabel score` writes agreement
  counts only to `experiments/results/c2/dev_relabel.json` (Cohen's kappa marked unreliable).

Required before P5 (owner, 2026-11-25): the central-message cues are tuned only on these labels,
never on the planted labels (ADR 0021 §4b-4c).

- **draw:** every accepted transcript of the **dev part**, in one seeded shuffled order across
  all groups, so any stopping point stays blind and covers every group. The sheet
  (`data/c2/dev_labels/sheet.yaml`) shows a neutral key and the text only; the id, planted group,
  hook and a hash of the text go to `key.json`. Do not open `key.json` while labelling.
- **The owner fills, per item:** `main_point_stated` (yes/no); `main_point_sentence` (required
  when yes: the sentence that states it, copied from the text); `restated_in_closing` (yes/no:
  the main point comes back in the closing, which the recurrence measure needs).
- **status:** how many items are done, and flags: yes without a sentence, a sentence that is not
  in the text, a sentence or a restatement without a main point, a value that is not yes/no. A
  sheet whose text was changed is refused.
- **score:** runs the current central-message detector (the app's spaCy parser, local; no
  network) and writes **counts only** to `experiments/results/c2/dev_labels.json`: detector vs
  owner for "stated" and "restated in closing", how often the detector's thesis sentence matches
  the owner's sentence, and the owner's labels by planted group. No transcript text.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "data" / "c2" / "generated.jsonl"
OUT_DIR = ROOT / "data" / "c2" / "dev_labels"
RESULTS = ROOT / "experiments" / "results" / "c2" / "dev_labels.json"
POSITIONS = ROOT / "experiments" / "results" / "c2" / "dev_positions.json"
RELABEL_RESULTS = ROOT / "experiments" / "results" / "c2" / "dev_relabel.json"
RELABEL_N = 20  # owner, 2026-11-26 (ADR 0025 item 7)
RELABEL_SEED = 20261203
SEED = 20261125
FIELDS = ("main_point_stated", "main_point_sentence", "restated_in_closing")
SENTENCE_COUNTS = ("owner_yes", "detector_found", "matches", "owner_sentence_among_statements")

Sheet = list[dict[str, Any]]


@dataclass(frozen=True)
class Detected:
    thesis: str | None  # the detector's thesis sentence (first thesis-cue sentence in the opening)
    statements: tuple[str, ...]  # every thesis-cue sentence
    in_closing: bool  # a closing sentence repeats a key phrase of the thesis


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plain(text: str) -> str:
    from presenceai.text.document import plain

    return plain(text)


def draw(rows: list[dict[str, Any]], seed: int = SEED) -> tuple[Sheet, dict[str, Any]]:
    """(sheet items, key): every dev item once, one shuffled order across groups, blinded."""
    rng = random.Random(seed)  # noqa: S311 - reproducible order, not security
    chosen = sorted((r for r in rows if r["part"] == "dev"), key=lambda r: r["id"])
    rng.shuffle(chosen)
    sheet, key = [], {}
    for n, r in enumerate(chosen, start=1):
        k = f"d{n:03d}"
        sheet.append({"key": k, "text": r["text"], **dict.fromkeys(FIELDS)})
        key[k] = {"id": r["id"], "group": r["group"], "hook": r["hook"], "sha256": _hash(r["text"])}
    return sheet, key


def yes_no(value: Any) -> bool | None:
    """None when empty; a YAML boolean or "yes"/"no" strings; anything else is refused."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return value
    text = value.strip().lower() if isinstance(value, str) else ""
    if text in ("yes", "y", "true"):
        return True
    if text in ("no", "n", "false"):
        return False
    raise ValueError(f"not yes/no: {value!r}")


def _check(sheet: Sheet, key: dict[str, Any]) -> None:
    if [s["key"] for s in sheet] != list(key):
        raise ValueError("sheet keys differ from key.json")
    changed = [s["key"] for s in sheet if _hash(s["text"]) != key[s["key"]]["sha256"]]
    if changed:
        raise ValueError(f"text changed for {changed}; restore it from the drawn sheet")


def _item(s: dict[str, Any]) -> tuple[dict[str, bool | None], list[str]]:
    """(parsed labels, flags) for one item."""
    flags: list[str] = []
    parsed: dict[str, bool | None] = {}
    for f in ("main_point_stated", "restated_in_closing"):
        try:
            parsed[f] = yes_no(s.get(f))
        except ValueError:
            parsed[f] = None
            flags.append(f"{f}: not yes/no")
    sentence = s.get("main_point_sentence")
    sentence = sentence.strip() if isinstance(sentence, str) and sentence.strip() else None
    stated = parsed["main_point_stated"]
    if stated is True and sentence is None:
        flags.append("main_point_stated is yes but the sentence missing")
    if sentence is not None and _plain(sentence) not in _plain(s["text"]):
        flags.append("main_point_sentence not found in the text")
    if stated is False and sentence is not None:
        flags.append("sentence given but main_point_stated is no")
    if stated is False and parsed["restated_in_closing"] is True:
        flags.append("restated without a main point")
    return parsed, flags


def _done(parsed: dict[str, bool | None], flags: list[str]) -> bool:
    return not flags and all(v is not None for v in parsed.values())


def status(sheet: Sheet, key: dict[str, Any]) -> dict[str, Any]:
    _check(sheet, key)
    done, flags = 0, {}
    for s in sheet:
        parsed, f = _item(s)
        if f:
            flags[s["key"]] = "; ".join(f)
        done += _done(parsed, f)
    return {"items": len(sheet), "done": done, "flags": flags}


def _confusion(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    """pairs: (detector, owner)."""
    c = Counter(pairs)
    return {"tp": c[True, True], "fp": c[True, False], "fn": c[False, True], "tn": c[False, False]}


def _same(a: str, b: str) -> bool:
    pa, pb = _plain(a), _plain(b)
    return bool(pa) and bool(pb) and (pa in pb or pb in pa)


def score(
    sheet: Sheet, key: dict[str, Any], detect: Callable[[str], Detected], *, partial: bool = False
) -> dict[str, Any]:
    st = status(sheet, key)
    if st["flags"]:
        raise ValueError(f"{len(st['flags'])} flagged items; fix them first (run status)")
    items = [s for s in sheet if _done(*_item(s))]
    if len(items) < len(sheet) and not partial:
        raise ValueError(f"sheet not finished: {len(items)} of {len(sheet)} done (or --partial)")
    stated, closing = [], []
    sent = Counter()
    by_group: dict[str, Counter[str]] = {}
    for s in items:
        labels, _ = _item(s)
        owner_stated, owner_closing = (
            bool(labels["main_point_stated"]),
            bool(labels["restated_in_closing"]),
        )
        d = detect(s["text"])
        stated.append((d.thesis is not None, owner_stated))
        closing.append((d.in_closing, owner_closing))
        group = key[s["key"]]["group"]
        by_group.setdefault(group, Counter())["yes" if owner_stated else "no"] += 1
        if owner_stated:
            mine = s["main_point_sentence"]
            sent["owner_yes"] += 1
            if d.thesis is not None:
                sent["detector_found"] += 1
                sent["matches"] += _same(d.thesis, mine)
            sent["owner_sentence_among_statements"] += any(_same(x, mine) for x in d.statements)
    return {
        "items": len(items),
        "of": len(sheet),
        "stated": _confusion(stated),
        "restated_in_closing": _confusion(closing),
        "sentence": {k: sent[k] for k in SENTENCE_COUNTS},
        "owner_stated_by_planted_group": {g: dict(c) for g, c in sorted(by_group.items())},
    }


def relabel_draw(
    sheet: Sheet, key: dict[str, Any], *, n: int = RELABEL_N, seed: int = RELABEL_SEED
) -> tuple[Sheet, dict[str, Any]]:
    """The owner's consistency check (ADR 0025 item 7): `n` random items of the finished first
    sheet, new neutral keys, a new order, none of the first labels shown."""
    st = status(sheet, key)
    if st["flags"] or st["done"] < len(sheet):
        raise ValueError("the first sheet must be finished and unflagged before the relabel draw")
    rng = random.Random(seed)  # noqa: S311 - reproducible sample, not security
    chosen = rng.sample(sorted(s["key"] for s in sheet), n)
    rng.shuffle(chosen)
    by_key = {s["key"]: s for s in sheet}
    re_sheet, re_key = [], {}
    for i, k in enumerate(chosen, start=1):
        rk = f"r{i:02d}"
        re_sheet.append({"key": rk, "text": by_key[k]["text"], **dict.fromkeys(FIELDS)})
        re_key[rk] = {"first": k, "sha256": key[k]["sha256"]}
    return re_sheet, re_key


def _kappa(c: Counter[tuple[bool, bool]]) -> float | None:
    n = sum(c.values())
    if not n:
        return None
    po = (c[True, True] + c[False, False]) / n
    a1, b1 = (c[True, True] + c[True, False]) / n, (c[True, True] + c[False, True]) / n
    pe = a1 * b1 + (1 - a1) * (1 - b1)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def _agreement(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """pairs: (first, now)."""
    c = Counter(pairs)
    return {"agree": c[True, True] + c[False, False], "both_yes": c[True, True],
            "both_no": c[False, False], "first_yes_now_no": c[True, False],
            "first_no_now_yes": c[False, True], "kappa": _kappa(c)}  # fmt: skip


def relabel_score(
    re_sheet: Sheet, re_key: dict[str, Any], sheet: Sheet, key: dict[str, Any]
) -> dict[str, Any]:
    """Agreement of the relabel with the first labels; counts only."""
    st = status(re_sheet, re_key)
    if st["flags"]:
        raise ValueError(f"{len(st['flags'])} flagged items; fix them first (relabel status)")
    if st["done"] < len(re_sheet):
        raise ValueError(f"relabel not finished: {st['done']} of {len(re_sheet)} done")
    status(sheet, key)
    first = {s["key"]: s for s in sheet}
    stated, closing, sent = [], [], Counter()
    for s in re_sheet:
        f = first[re_key[s["key"]]["first"]]
        now, _ = _item(s)
        was, _ = _item(f)
        stated.append((bool(was["main_point_stated"]), bool(now["main_point_stated"])))
        closing.append((bool(was["restated_in_closing"]), bool(now["restated_in_closing"])))
        if was["main_point_stated"] and now["main_point_stated"]:
            sent["both_yes"] += 1
            sent["same"] += _same(f["main_point_sentence"], s["main_point_sentence"])
    n = len(re_sheet)
    return {
        "items": n,
        "stated": _agreement(stated),
        "restated_in_closing": _agreement(closing),
        "sentence": {"both_yes": sent["both_yes"], "same": sent["same"]},
        "kappa_reliable": False,  # ADR 0025 item 7: kappa at n = 20 is marked unreliable
        "note": f"Cohen's kappa at n = {n} is reported but unreliable (ADR 0025 item 7)",
    }


def _sublist(hay: list[str], needle: list[str]) -> int | None:
    n = len(needle)
    return next((i for i in range(len(hay) - n + 1) if hay[i : i + n] == needle), None)


def positions(
    sheet: Sheet, key: dict[str, Any], *, opening_pct: float = 20, closing_pct: float = 15
) -> dict[str, Any]:
    """Where the owner's main-point sentences sit, by word share; counts only. Regions follow the
    app's rule (`Document.in_opening` / `in_closing`): opening if the sentence starts in the first
    `opening_pct` % of the words, else closing if it ends in the last `closing_pct` %, else
    middle. Words are the normalised words (`words_of`) of the text and of the sentence."""
    from presenceai.text.document import words_of

    status(sheet, key)  # refuses a changed text
    regions, deciles, located = Counter(), Counter(), 0
    for s in sheet:
        labels, flags = _item(s)
        if flags or labels["main_point_stated"] is not True:
            continue
        words, mine = words_of(s["text"]), words_of(s["main_point_sentence"])
        start = _sublist(words, mine)
        if start is None or not mine:
            regions["not_located"] += 1
            continue
        located += 1
        n, last = len(words), start + len(mine) - 1
        opening_end = max(1, round(n * opening_pct / 100))
        closing_start = n - max(1, round(n * closing_pct / 100))
        region = "opening" if start < opening_end else "middle"
        if region == "middle" and last >= closing_start:
            region = "closing"
        regions[region] += 1
        deciles[min(9, start * 10 // n)] += 1
    return {
        "sentences": sum(regions.values()),
        "opening_pct": opening_pct,
        "closing_pct": closing_pct,
        "regions": {k: regions[k] for k in ("opening", "middle", "closing", "not_located")},
        "start_share_decile": {f"{d * 10}-{d * 10 + 10}%": deciles[d] for d in range(10)},
        "located": located,
    }


def app_detector(parser: Any = None) -> Callable[[str], Detected]:
    """The app's central-message detector on a text: whitespace words with synthetic times
    (punctuation is kept, so sentences split as in a transcript), the local spaCy parser."""
    from presenceai.schemas.speech import Segment, Transcript, Word
    from presenceai.settings import get_settings
    from presenceai.text import central_message
    from presenceai.text.analyze import analyze_text
    from presenceai.text.nlp import spacy_parser

    parser = parser or spacy_parser()  # tests pass a fake parser
    pauses = get_settings().speech.pauses
    k_statements, k_opening, k_closing = central_message.KEYS[:3]

    def transcript(text: str) -> Transcript:
        out, words = "", []
        for i, token in enumerate(text.split()):
            out += " " if out else ""
            start = len(out)
            out += token
            words.append(Word(text=token, start_s=i * 0.4, end_s=i * 0.4 + 0.3, probability=0.95,
                              start_char=start, end_char=len(out)))  # fmt: skip
        seg = Segment(start_s=0.0, end_s=words[-1].end_s, words=tuple(words), avg_logprob=-0.1,
                      no_speech_prob=0.0)  # fmt: skip
        return Transcript(language="en", duration_s=words[-1].end_s, text=out, segments=(seg,))

    def detect(text: str) -> Detected:
        m = analyze_text(transcript(text), parser=parser, pauses=pauses).measurements

        def spans(k: str) -> tuple[str, ...]:
            if k not in m or not m[k].value:
                return ()
            return tuple(e.transcript.text for e in m[k].evidence if e.transcript is not None)

        thesis = spans(k_opening)
        return Detected(thesis=thesis[0] if thesis else None, statements=spans(k_statements),
                        in_closing=bool(spans(k_closing)))  # fmt: skip

    return detect


def _read(path: Path) -> Any:
    return yaml.safe_load(path.read_text("utf-8"))


def _relabel(action: str, sheet: Sheet, key: dict[str, Any]) -> int:
    re_path, re_key_path = OUT_DIR / "relabel_sheet.yaml", OUT_DIR / "relabel_key.json"
    if action == "draw":
        if re_path.exists():
            raise SystemExit(f"{re_path} exists; the relabel sheet is drawn once")
        re_sheet, re_key = relabel_draw(sheet, key)
        re_path.write_text(yaml.safe_dump(re_sheet, allow_unicode=True, sort_keys=False,
                                          width=100), encoding="utf-8")  # fmt: skip
        re_key_path.write_text(json.dumps(re_key, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(re_sheet)} items to {re_path.relative_to(ROOT)}; "
              "label them without looking at sheet.yaml")  # fmt: skip
        return 0
    re_sheet, re_key = _read(re_path), json.loads(re_key_path.read_text("utf-8"))
    if action == "status":
        st = status(re_sheet, re_key)
        print(f"{st['done']} of {st['items']} done; {len(st['flags'])} flagged")
        for k, f in st["flags"].items():
            print(f"  {k}: {f}")
        return 1 if st["flags"] else 0
    if action == "score":
        result = relabel_score(re_sheet, re_key, sheet, key)
        RELABEL_RESULTS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    raise SystemExit(f"unknown relabel action {action!r}: draw | status | score")


def main(argv: list[str]) -> int:
    action = argv[1] if len(argv) > 1 else "status"
    sheet_path, key_path = OUT_DIR / "sheet.yaml", OUT_DIR / "key.json"
    if action == "draw":
        if sheet_path.exists():
            raise SystemExit(f"{sheet_path} exists; the sheet is drawn once")
        rows = [json.loads(x) for x in GENERATED.read_text("utf-8").splitlines() if x.strip()]
        sheet, key = draw(rows)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        sheet_path.write_text(yaml.safe_dump(sheet, allow_unicode=True, sort_keys=False, width=100),
                              encoding="utf-8")  # fmt: skip
        key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(sheet)} dev items to {sheet_path.relative_to(ROOT)} (key kept apart)")
        return 0
    sheet, key = _read(sheet_path), json.loads(key_path.read_text("utf-8"))
    if action == "relabel":
        return _relabel(argv[2] if len(argv) > 2 else "status", sheet, key)
    if action == "status":
        st = status(sheet, key)
        print(f"{st['done']} of {st['items']} done; {len(st['flags'])} flagged")
        for k, f in st["flags"].items():
            print(f"  {k}: {f}")
        return 1 if st["flags"] else 0
    if action == "score":
        result = score(sheet, key, app_detector(), partial="--partial" in argv)
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    if action == "positions":
        from presenceai.rubric.loader import load_thresholds

        values = load_thresholds("s1_presentation").values
        result = positions(sheet, key, opening_pct=values["opening_share_pct"],
                           closing_pct=values["closing_share_pct"])  # fmt: skip
        POSITIONS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    raise SystemExit(
        f"unknown action {action!r}: draw | status | score [--partial] | positions"
        " | relabel draw|status|score"
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
