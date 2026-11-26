"""The owner's central-message labels on the C2 development part (ADR 0021 §4c)."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from typing import Any

import pytest

from presenceai.evalconfig import REPO_ROOT


def _load(name: str) -> Any:
    path = REPO_ROOT / "experiments" / "c2_synth" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"c2_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"c2_{name}"] = module
    spec.loader.exec_module(module)
    return module


dl = _load("dev_labels")
GROUPS = ("control", "no_central_message", "no_conclusion", "no_supporting_material",
          "undefined_jargon")  # fmt: skip
THESIS = "My main point is that solar roofs pay back."
OTHER = "Panels are cheap now."


def _rows(per_group: int = 6) -> list[dict[str, Any]]:
    rows = []
    for g in GROUPS:
        for i in range(per_group):
            for part in ("dev", "test"):
                rows.append({"id": f"{g}-{part}-{i}", "group": g, "part": part, "hook": i % 2 == 0,
                             "text": f"{THESIS} {OTHER} Item {g} {part} {i}."})  # fmt: skip
    return rows


def _fill(sheet: list[dict[str, Any]]) -> None:
    for s in sheet:
        s.update(main_point_stated="yes", main_point_sentence=THESIS, restated_in_closing="no")


def test_draw_takes_every_dev_item_once_blinded():
    sheet, key = dl.draw(_rows())
    assert len(sheet) == 30
    assert Counter(key[s["key"]]["group"] for s in sheet) == dict.fromkeys(GROUPS, 6)
    assert all(k["id"].split("-")[1] == "dev" for k in key.values())
    for s in sheet:  # the sheet shows no planted information
        assert set(s) == {"key", "text", *dl.FIELDS}
        assert all(s[f] is None for f in dl.FIELDS)
    assert dl.draw(_rows()) == (sheet, key)  # seeded


def test_draw_order_mixes_groups_from_the_start():
    # One shuffled order: any stopping point covers every group (owner, 2026-11-25).
    sheet, key = dl.draw(_rows(per_group=24))
    first = {key[s["key"]]["group"] for s in sheet[:20]}
    assert first == set(GROUPS)
    groups = [key[s["key"]]["group"] for s in sheet]
    assert groups != sorted(groups)


def test_status_counts_done_items_and_flags_problems():
    sheet, key = dl.draw(_rows())
    assert dl.status(sheet, key) == {"items": 30, "done": 0, "flags": {}}
    _fill(sheet)
    sheet[0]["main_point_sentence"] = None  # yes needs a sentence
    sheet[1]["main_point_sentence"] = "A sentence that is not in the talk."
    sheet[2].update(main_point_stated="no", main_point_sentence=None, restated_in_closing="yes")
    sheet[3]["restated_in_closing"] = "maybe"
    sheet[4]["restated_in_closing"] = None  # not done, but not a flag
    st = dl.status(sheet, key)
    flags = st["flags"]
    assert "sentence missing" in flags[sheet[0]["key"]]
    assert "sentence not found in the text" in flags[sheet[1]["key"]]
    assert "restated without a main point" in flags[sheet[2]["key"]]
    assert "not yes/no" in flags[sheet[3]["key"]]
    assert sheet[4]["key"] not in flags
    assert st["done"] == 30 - 5


def test_sentence_is_found_with_normalised_spacing_and_case():
    sheet, key = dl.draw(_rows())
    _fill(sheet)
    sheet[0]["main_point_sentence"] = "  my MAIN point is that solar roofs pay back "
    assert dl.status(sheet, key)["flags"] == {}


def test_a_changed_text_is_refused():
    sheet, key = dl.draw(_rows())
    sheet[0]["text"] += " edited"
    with pytest.raises(ValueError, match="changed"):
        dl.status(sheet, key)


def _detect(thesis: str | None, closing: bool) -> Any:
    def detect(text: str) -> Any:
        stated = (thesis,) if thesis else ()
        return dl.Detected(thesis=thesis, statements=stated, in_closing=closing)

    return detect


def test_score_reports_detector_agreement_and_sentence_matches_as_counts():
    sheet, key = dl.draw(_rows())
    _fill(sheet)
    sheet[0].update(main_point_stated="no", main_point_sentence=None)
    r = dl.score(sheet, key, _detect(THESIS, closing=False))
    assert r["items"] == 30
    assert r["stated"] == {"tp": 29, "fp": 1, "fn": 0, "tn": 0}
    assert r["restated_in_closing"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 30}
    assert r["sentence"] == {"owner_yes": 29, "detector_found": 29, "matches": 29,
                             "owner_sentence_among_statements": 29}  # fmt: skip
    assert set(r["owner_stated_by_planted_group"]) == set(GROUPS)
    assert json.dumps(r).find("solar") == -1  # counts only, never transcript text

    r = dl.score(sheet, key, _detect(OTHER, closing=True))
    assert r["sentence"]["matches"] == 0 and r["sentence"]["detector_found"] == 29
    assert r["restated_in_closing"]["fp"] == 30


def test_score_refuses_flagged_or_unfinished_sheets_unless_partial():
    sheet, key = dl.draw(_rows())
    _fill(sheet)
    sheet[0]["restated_in_closing"] = None
    with pytest.raises(ValueError, match="not finished"):
        dl.score(sheet, key, _detect(THESIS, closing=False))
    assert dl.score(sheet, key, _detect(THESIS, closing=False), partial=True)["items"] == 29
    sheet[1]["main_point_sentence"] = None
    with pytest.raises(ValueError, match="flag"):
        dl.score(sheet, key, _detect(THESIS, closing=False), partial=True)


def test_app_detector_finds_the_thesis_sentence_without_a_model(fake_parser):
    detect = dl.app_detector(parser=fake_parser)
    talk = (f"{THESIS} " + " ".join(f"{OTHER}" for _ in range(12))
            + " In the end, solar roofs pay back for every home.")  # fmt: skip
    d = detect(talk)
    assert d.thesis is not None and _same_plain(d.thesis, THESIS)
    assert d.statements and d.in_closing
    assert detect(" ".join(OTHER for _ in range(10))).thesis is None


def _same_plain(a: str, b: str) -> bool:
    from presenceai.text.document import plain

    return plain(a) == plain(b)


def test_positions_counts_regions_by_word_share_without_text():
    filler = " ".join(f"Word{i} here." for i in range(20))  # 40 words
    first, last = "Alpha point is clear.", "Omega point is clear."
    rows = [{"id": f"x-dev-{i}", "group": GROUPS[i % 5], "part": "dev", "hook": False,
             "text": f"{first} {filler} {last}"} for i in range(4)]  # fmt: skip
    sheet, key = dl.draw(rows)
    for n, s in enumerate(sheet):
        s.update(main_point_stated="yes", restated_in_closing="no",
                 main_point_sentence=[first, last, "Word10 here.", first][n])  # fmt: skip
    sheet[3].update(main_point_stated="no", main_point_sentence=None)
    r = dl.positions(sheet, key)
    assert r["regions"] == {"opening": 1, "middle": 1, "closing": 1, "not_located": 0}
    assert r["sentences"] == 3 and sum(r["start_share_decile"].values()) == 3
    assert json.dumps(r).find("Alpha") == -1


def _first_sheet() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sheet, key = dl.draw(_rows(per_group=8))  # 40 dev items
    _fill(sheet)
    for s in sheet[::4]:
        s.update(main_point_stated="no", main_point_sentence=None)
    return sheet, key


def test_relabel_draw_is_20_items_new_keys_new_order_no_first_labels():
    sheet, key = _first_sheet()
    re_sheet, re_key = dl.relabel_draw(sheet, key)
    assert len(re_sheet) == 20 and len(set(re_key[s["key"]]["first"] for s in re_sheet)) == 20
    for s in re_sheet:  # blind to the first labels and to the first keys
        assert set(s) == {"key", "text", *dl.FIELDS}
        assert all(s[f] is None for f in dl.FIELDS)
        assert s["key"].startswith("r")
    firsts = [re_key[s["key"]]["first"] for s in re_sheet]
    assert firsts != sorted(firsts, key=[s["key"] for s in sheet].index)  # new order
    assert dl.relabel_draw(sheet, key) == (re_sheet, re_key)  # seeded


def test_relabel_draw_needs_a_finished_first_sheet():
    sheet, key = _first_sheet()
    sheet[0]["restated_in_closing"] = None
    with pytest.raises(ValueError, match="finished"):
        dl.relabel_draw(sheet, key)


def test_relabel_score_counts_agreement_with_the_first_labels():
    sheet, key = _first_sheet()
    re_sheet, re_key = dl.relabel_draw(sheet, key)
    first = {s["key"]: s for s in sheet}
    for s in re_sheet:  # the owner gives the same labels again ...
        f = first[re_key[s["key"]]["first"]]
        s.update({k: f[k] for k in dl.FIELDS})
    r = dl.relabel_score(re_sheet, re_key, sheet, key)
    assert r["items"] == 20 and r["stated"]["agree"] == 20
    assert r["stated"]["kappa"] == 1.0 and r["kappa_reliable"] is False
    assert r["sentence"]["both_yes"] == r["sentence"]["same"]
    s0 = next(s for s in re_sheet if dl.yes_no(s["main_point_stated"]))
    s0["restated_in_closing"] = "yes"  # ... except one flipped "restated" (first label: no)
    r = dl.relabel_score(re_sheet, re_key, sheet, key)
    assert r["stated"]["agree"] == 20 and r["restated_in_closing"]["agree"] == 19
    assert r["restated_in_closing"]["first_no_now_yes"] == 1
    assert json.dumps(r).find("solar") == -1  # counts only


def test_relabel_score_refuses_unfinished_or_flagged():
    sheet, key = _first_sheet()
    re_sheet, re_key = dl.relabel_draw(sheet, key)
    with pytest.raises(ValueError, match="not finished"):
        dl.relabel_score(re_sheet, re_key, sheet, key)
