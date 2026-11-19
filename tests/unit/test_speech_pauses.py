"""speech/pauses.py: VAD gaps -> pause events with position, confidence and evidence (ADR 0011)."""

from typing import Any

import pytest

from presenceai.schemas.assessment import Confidence
from presenceai.schemas.speech import PausePosition, Segment, SpeechRegion, Transcript, Word
from presenceai.speech import pauses


def _transcript(tokens: list[tuple[str, float, float]], p: float = 0.95) -> Transcript:
    text, words = "", []
    for token, start, end in tokens:
        if text:
            text += " "
        s = len(text)
        text += token
        words.append(
            Word(
                text=token,
                start_s=start,
                end_s=end,
                probability=p,
                start_char=s,
                end_char=len(text),
            )
        )
    segs = (
        (
            Segment(
                start_s=words[0].start_s,
                end_s=words[-1].end_s,
                words=tuple(words),
                avg_logprob=-0.1,
                no_speech_prob=0.0,
            ),
        )
        if words
        else ()
    )
    return Transcript(language="en", duration_s=60.0, text=text, segments=segs)


def _r(a: float, b: float) -> SpeechRegion:
    return SpeechRegion(start_s=a, end_s=b)


# "One. Two, three four." -> sentence boundary after "One.", clause after "Two,", mid after "three"
TOKENS = [("One.", 0.0, 0.5), ("Two,", 1.5, 2.0), ("three", 2.6, 3.0), ("four.", 4.2, 4.6)]
REGIONS = (_r(0.0, 0.5), _r(1.5, 2.0), _r(2.6, 3.0), _r(4.2, 4.6))


def test_positions_sentence_clause_mid(speech_cfg: Any):
    events = pauses.find_pauses(REGIONS, _transcript(TOKENS), speech_cfg)
    assert [e.position for e in events] == [
        PausePosition.SENTENCE_BOUNDARY,
        PausePosition.CLAUSE_BOUNDARY,
        PausePosition.MID_SENTENCE,
    ]
    assert [round(e.duration_s, 2) for e in events] == [1.0, 0.6, 1.2]
    assert [(e.word_before, e.word_after) for e in events] == [(0, 1), (1, 2), (2, 3)]


@pytest.mark.parametrize(
    ("token", "position"),
    [
        ("end.", PausePosition.SENTENCE_BOUNDARY),
        ("why?", PausePosition.SENTENCE_BOUNDARY),
        ("wow!", PausePosition.SENTENCE_BOUNDARY),
        ("ماذا؟", PausePosition.SENTENCE_BOUNDARY),
        ("so…", PausePosition.SENTENCE_BOUNDARY),
        ('said."', PausePosition.SENTENCE_BOUNDARY),  # closing quote after the mark
        ("first,", PausePosition.CLAUSE_BOUNDARY),
        ("أولاً،", PausePosition.CLAUSE_BOUNDARY),
        ("then;", PausePosition.CLAUSE_BOUNDARY),
        ("ثم؛", PausePosition.CLAUSE_BOUNDARY),
        ("word", PausePosition.MID_SENTENCE),
        ("3.5", PausePosition.MID_SENTENCE),  # a mark inside a token is not a boundary
    ],
)
def test_word_position(speech_cfg: Any, token: str, position: PausePosition):
    assert pauses.position_after(token, speech_cfg.pauses) == position


def test_gaps_shorter_than_the_smallest_threshold_are_not_pauses(speech_cfg: Any):
    regions = (_r(0.0, 1.0), _r(1.2, 2.0), _r(2.25, 3.0))  # 0.20 s and 0.25 s gaps
    t = _transcript([("a.", 0.1, 0.9), ("b.", 1.3, 1.9), ("c.", 2.3, 2.9)])
    events = pauses.find_pauses(regions, t, speech_cfg)
    assert [round(e.duration_s, 2) for e in events] == [0.25]


def test_sparse_punctuation_gives_unknown_low(speech_cfg: Any):
    """Fewer than 1 sentence mark per 30 words: positions cannot be placed reliably."""
    tokens = [(f"w{i}", i * 1.0, i * 1.0 + 0.4) for i in range(40)]
    tokens[10] = ("w10.", 10.0, 10.4)  # 1 mark for 40 words < 40 / 30
    regions = tuple(_r(s, e) for _, s, e in tokens)
    events = pauses.find_pauses(regions, _transcript(tokens), speech_cfg)
    assert events and all(e.position == PausePosition.UNKNOWN for e in events)
    assert all(e.confidence == Confidence.LOW for e in events)


def test_punctuation_rule_boundary(speech_cfg: Any):
    words = [f"w{i}" for i in range(30)]
    t30 = _transcript([(w + ("." if i == 29 else ""), i, i + 0.4) for i, w in enumerate(words)])
    assert pauses.punctuation_reliable(t30, speech_cfg.pauses)  # 1 mark per 30 words: enough
    t31 = _transcript([(f"w{i}" + ("." if i == 30 else ""), i, i + 0.4) for i in range(31)])
    assert not pauses.punctuation_reliable(t31, speech_cfg.pauses)
    assert not pauses.punctuation_reliable(_transcript([]), speech_cfg.pauses)


def test_confidence_follows_the_adjacent_words(speech_cfg: Any):
    t = _transcript(TOKENS, p=0.75)
    events = pauses.find_pauses(REGIONS, t, speech_cfg)
    assert {e.confidence for e in events} == {Confidence.MEDIUM}
    t_low = _transcript(TOKENS, p=0.5)
    assert {e.confidence for e in pauses.find_pauses(REGIONS, t_low, speech_cfg)} == {
        Confidence.LOW
    }


def test_gap_without_words_on_one_side_is_unknown(speech_cfg: Any):
    regions = (_r(0.0, 0.5), _r(1.5, 2.0))  # speech after the gap has no recognised word
    events = pauses.find_pauses(regions, _transcript([("Hello.", 0.0, 0.5)]), speech_cfg)
    assert events[0].position == PausePosition.UNKNOWN and events[0].word_after is None


def test_summary_counts_by_threshold_and_position(speech_cfg: Any):
    t = _transcript(TOKENS)
    events = pauses.find_pauses(REGIONS, t, speech_cfg)
    m, nm = pauses.summarise(events, REGIONS, t, speech_cfg)
    assert m["pauses.count_ge_0.25s"].value == 3
    assert m["pauses.count_ge_1s"].value == 2
    assert m["pauses.count_ge_2s"].value == 0
    assert m["pauses.count.sentence_boundary"].value == 1
    assert m["pauses.count.clause_boundary"].value == 1
    assert m["pauses.count.mid_sentence"].value == 1
    assert m["pauses.count.unknown"].value == 0
    assert m["pauses.longest_s"].value == pytest.approx(1.2)
    assert m["pauses.mean_s"].value == pytest.approx((1.0 + 0.6 + 1.2) / 3)
    assert nm == {}
    for meas in m.values():
        assert meas.evidence and meas.language == "en" and meas.live is False


def test_every_pause_is_evidence_with_time_and_quote(speech_cfg: Any):
    t = _transcript(TOKENS)
    events = pauses.find_pauses(REGIONS, t, speech_cfg)
    m, _ = pauses.summarise(events, REGIONS, t, speech_cfg)
    ev = m["pauses.count_ge_1s"].evidence
    assert [(e.time.start_s, e.time.end_s) for e in ev if e.time] == [(0.5, 1.5), (3.0, 4.2)]
    assert [e.transcript.text for e in ev if e.transcript] == ["One. Two,", "three four."]


def test_no_pauses_still_has_evidence_and_skips_mean(speech_cfg: Any):
    regions = (_r(0.0, 12.0),)
    t = _transcript([("Long.", 0.0, 12.0)])
    m, nm = pauses.summarise((), regions, t, speech_cfg)
    assert m["pauses.count_ge_0.25s"].value == 0
    assert m["pauses.count_ge_0.25s"].evidence[0].time is not None
    assert "pauses.mean_s" not in m and "pauses.mean_s" in nm


def test_no_speech_is_not_measurable(speech_cfg: Any):
    m, nm = pauses.summarise((), (), _transcript([]), speech_cfg)
    assert m == {} and "pauses.count_ge_0.25s" in nm


def test_a_word_stretched_into_the_pause_is_skipped_by_the_midpoint_rule(speech_cfg: Any):
    # ADR 0016 open item 6, checked on C1 dev and kept: "Two," runs 0.6 s into the pause
    # (midpoint inside it), so the word before the pause is "One." and the position "sentence"
    tokens = [("One.", 0.0, 0.5), ("Two,", 1.1, 2.1), ("three", 2.6, 3.0), ("four.", 3.1, 3.4)]
    regions = (_r(0.0, 1.5), _r(2.5, 3.4))
    (event,) = pauses.find_pauses(regions, _transcript(tokens), speech_cfg)
    assert event.word_before == 0 and event.position == PausePosition.SENTENCE_BOUNDARY
