"""speech/repetitions.py: repeated 1–3-word phrases within 2 s (FINAL_04 §2, ADR 0012)."""

from typing import Any

import pytest

from presenceai.schemas.assessment import Confidence
from presenceai.speech import fillers, repetitions


def _find(sb: Any, cfg: Any, text: str, language: str = "en", step: float = 0.5) -> Any:
    t = sb.spaced(text, step=step, language=language)
    return t, repetitions.find_repetitions(t, fillers.load_lexicon(language), cfg)  # pyright: ignore[reportArgumentType]


def _quotes(t: Any, events: Any) -> list[str]:
    return [t.span(e.first_word, e.last_word + 1).text for e in events]


def test_two_word_repetition(sb: Any, speech_cfg: Any):
    t, ev = _find(sb, speech_cfg, "I think I think that works")
    assert _quotes(t, ev) == ["I think I think"]
    assert ev[0].ngram == 2 and ev[0].copies == 2


def test_longest_phrase_wins(sb: Any, speech_cfg: Any):
    _t, ev = _find(sb, speech_cfg, "we can do we can do it")
    assert [(e.ngram, e.copies) for e in ev] == [(3, 2)]


def test_runs_count_every_extra_copy(sb: Any, speech_cfg: Any):
    t, ev = _find(sb, speech_cfg, "I I I went")
    assert _quotes(t, ev) == ["I I I"] and ev[0].copies == 3


def test_case_and_punctuation_are_ignored(sb: Any, speech_cfg: Any):
    t, ev = _find(sb, speech_cfg, "Hello, hello everyone")
    assert _quotes(t, ev) == ["Hello, hello"]


def test_hesitation_between_copies_is_skipped(sb: Any, speech_cfg: Any):
    t, ev = _find(sb, speech_cfg, "the uh the plan")
    assert _quotes(t, ev) == ["the uh the"]


def test_repeated_hesitations_are_fillers_not_repetitions(sb: Any, speech_cfg: Any):
    _, ev = _find(sb, speech_cfg, "uh uh well")
    assert ev == ()


def test_more_than_2s_apart_is_not_a_repetition(sb: Any, speech_cfg: Any):
    t = sb.transcript([("the", 0.0, 0.3), ("the", 2.5, 2.8)])  # 2.2 s between copies
    assert repetitions.find_repetitions(t, fillers.load_lexicon("en"), speech_cfg) == ()
    t2 = sb.transcript([("the", 0.0, 0.3), ("the", 2.2, 2.5)])  # 1.9 s
    assert len(repetitions.find_repetitions(t2, fillers.load_lexicon("en"), speech_cfg)) == 1


@pytest.mark.parametrize(
    "text",
    [
        "يعني يعني الفكرة",  # a doubled discourse marker is a repetition (owner, 2026-11-17)
        "الفكرة الفِكرة واضحة",  # diacritics ignored
        "قال قـــال لي",  # tatweel ignored
        "أنا انا هنا",  # alef forms unified
    ],
)
def test_arabic_repetitions_use_the_same_normalisation(sb: Any, speech_cfg: Any, text: str):
    _, ev = _find(sb, speech_cfg, text, language="ar")
    assert len(ev) == 1 and ev[0].ngram == 1 and ev[0].first_word == 0


def test_measures_with_quote_evidence_and_cap(sb: Any, speech_cfg: Any):
    text = " ".join([f"w{i}" for i in range(76)] + ["the", "the", "I", "I"])
    t, ev = _find(sb, speech_cfg, text)
    regions = (sb.region(0.0, t.words[-1].end_s),)
    m, nm = repetitions.summarise(ev, t, regions, speech_cfg, verbatim=False)
    assert nm == {}
    assert m["repetitions.count"].value == 2
    assert m["repetitions.per_100_words"].value == pytest.approx(2 / 80 * 100)
    assert [e.transcript.text for e in m["repetitions.count"].evidence if e.transcript] == [
        "the the",
        "I I",
    ]
    assert m["repetitions.count"].confidence == Confidence.MEDIUM  # capped: base Whisper
    assert "lower bound" in m["repetitions.count"].method


def test_no_words_is_not_measurable(sb: Any, speech_cfg: Any):
    m, nm = repetitions.summarise((), sb.transcript([]), (), speech_cfg, verbatim=False)
    assert m == {} and "repetitions.count" in nm
