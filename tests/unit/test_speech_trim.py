"""speech/trim.py: words Whisper invents before the first or after the last speech (ADR 0015).

C1 clip s10_r0 is the model case: after the last sentence ("... useful for all.") Whisper added
"Thank you very much for joining us. Thank you." squeezed into the trailing silence.
"""

from typing import Any

import pytest

from presenceai.schemas.speech import Segment, Transcript, Word
from presenceai.speech import fillers, pauses, rate, repetitions, trim

TAIL = ["you", "very", "much", "for", "joining", "us.", "Thank", "you."]


def s10_like(sb: Any) -> Any:
    """Real words inside 0.5-12.0 s, then invented words at 12.42-12.44 s (zero length)."""
    toks = [(f"word{i}", 0.5 + i * 0.5, 0.5 + i * 0.5 + 0.4) for i in range(23)]
    toks += [(t, 12.42, 12.42) for t in TAIL[:-1]] + [(TAIL[-1], 12.42, 12.44)]
    return sb.transcript(toks, duration_s=12.46)


REGIONS_S10 = ((0.5, 5.0), (5.3, 11.9))


def regions(sb: Any, spans: Any) -> Any:
    return tuple(sb.region(a, b) for a, b in spans)


def cfg(speech_cfg: Any, tol: float | None = None) -> Any:
    if tol is None:
        return speech_cfg.transcript
    return speech_cfg.transcript.model_copy(update={"edge_tolerance_s": tol})


def test_tolerance_comes_from_config(speech_cfg: Any):
    assert speech_cfg.transcript.edge_tolerance_s == 0.2  # ADR 0015 (provisional)


def test_invented_words_after_the_last_speech_are_dropped(sb: Any, speech_cfg: Any):
    t = s10_like(sb)
    kept, dropped = trim.trim_to_speech(t, regions(sb, REGIONS_S10), cfg(speech_cfg))
    assert [w.text for w in dropped] == TAIL
    assert [w.text for w in kept.words] == [f"word{i}" for i in range(23)]
    for word in TAIL:
        assert word not in kept.text.split()


def test_words_inside_speech_are_kept(sb: Any, speech_cfg: Any):
    t = sb.spaced("one two three four five six", step=0.5, dur=0.3)  # 0.0 - 2.8 s
    kept, dropped = trim.trim_to_speech(t, regions(sb, [(0.0, 1.0), (1.5, 2.8)]), cfg(speech_cfg))
    assert dropped == ()
    assert kept == t  # nothing dropped: the very same transcript


def test_words_in_pauses_between_regions_are_kept(sb: Any, speech_cfg: Any):
    """Ends only (ADR 0015): Whisper's word times drift into pauses, so interior words stay."""
    t = sb.transcript([("a", 0.2, 0.6), ("b", 2.0, 2.4), ("c", 4.0, 4.5)])
    kept, dropped = trim.trim_to_speech(t, regions(sb, [(0.2, 0.7), (4.0, 4.6)]), cfg(speech_cfg))
    assert dropped == () and [w.text for w in kept.words] == ["a", "b", "c"]


def test_first_word_stretched_back_to_zero_is_kept(sb: Any, speech_cfg: Any):
    """Whisper often starts the first word at 0.0 s although speech starts at 0.5 s."""
    t = sb.transcript([("Many", 0.0, 0.72), ("people", 0.8, 1.2)])
    kept, dropped = trim.trim_to_speech(t, regions(sb, [(0.5, 1.3)]), cfg(speech_cfg))
    assert dropped == () and len(kept.words) == 2


def test_tolerance_decides_at_the_edge(sb: Any, speech_cfg: Any):
    t = sb.transcript([("in", 0.0, 0.9), ("near", 1.15, 1.3), ("far", 1.35, 1.5)])
    r = regions(sb, [(0.0, 1.0)])
    _, dropped = trim.trim_to_speech(t, r, cfg(speech_cfg))  # 0.2 s: near kept, far dropped
    assert [w.text for w in dropped] == ["far"]
    _, dropped0 = trim.trim_to_speech(t, r, cfg(speech_cfg, 0.0))
    assert [w.text for w in dropped0] == ["near", "far"]


def test_words_before_the_first_speech_are_dropped(sb: Any, speech_cfg: Any):
    t = sb.transcript([("Thanks.", 0.1, 0.3), ("Hello", 2.0, 2.4), ("all", 2.5, 2.9)])
    kept, dropped = trim.trim_to_speech(t, regions(sb, [(2.0, 3.0)]), cfg(speech_cfg))
    assert [w.text for w in dropped] == ["Thanks."]
    assert kept.text == "Hello all"


def test_no_speech_drops_every_word(sb: Any, speech_cfg: Any):
    t = sb.transcript([("Thank", 1.0, 1.3), ("you.", 1.3, 1.6)])
    kept, dropped = trim.trim_to_speech(t, (), cfg(speech_cfg))
    assert len(dropped) == 2
    assert kept.words == () and kept.text == "" and kept.segments == ()
    assert kept.language == t.language and kept.duration_s == t.duration_s


def test_offsets_are_rebuilt_and_segments_follow_kept_words(speech_cfg: Any, sb: Any):
    def word(text: str, a: float, b: float, start_char: int) -> Word:
        return Word(text=text, start_s=a, end_s=b, probability=0.9,
                    start_char=start_char, end_char=start_char + len(text))  # fmt: skip

    w1 = (word("Thanks.", 0.1, 0.3, 0),)
    w2 = (word("Hello", 2.0, 2.4, 8), word("all.", 2.5, 2.9, 14))
    w3 = (word("Bye.", 9.0, 9.0, 19),)
    t = Transcript(
        language="en", duration_s=10.0, text="Thanks. Hello all. Bye.",
        segments=(
            Segment(start_s=0.0, end_s=2.9, words=w1 + w2, avg_logprob=-0.2, no_speech_prob=0.1),
            Segment(start_s=9.0, end_s=9.0, words=w3, avg_logprob=-1.0, no_speech_prob=0.6),
        ),
    )  # fmt: skip
    kept, dropped = trim.trim_to_speech(t, regions(sb, [(2.0, 3.0)]), cfg(speech_cfg))
    assert [w.text for w in dropped] == ["Thanks.", "Bye."]
    assert kept.text == "Hello all."
    (seg,) = kept.segments
    assert (seg.start_s, seg.end_s) == (2.0, 2.9)  # narrowed to the kept words
    assert (seg.avg_logprob, seg.no_speech_prob) == (-0.2, 0.1)
    assert [(w.start_char, w.end_char) for w in kept.words] == [(0, 5), (6, 10)]
    for w in kept.words:
        assert kept.text[w.start_char : w.end_char] == w.text


@pytest.mark.parametrize("tol", [0.0, 0.2, 0.5])
def test_an_invented_word_never_appears_in_an_evidence_quote(sb: Any, speech_cfg: Any, tol: float):
    """AGENTS.md rule 4: every quote comes from the user's words. Run every quoting measure on
    the trimmed transcript and check that no quote contains a dropped word."""
    t = s10_like(sb)
    r = regions(sb, REGIONS_S10)
    kept, dropped = trim.trim_to_speech(t, r, cfg(speech_cfg, tol))
    assert dropped
    quotes: list[str] = []
    m_rate, _, _ = rate.word_rate(kept, r, speech_cfg)
    events = pauses.find_pauses(r, kept, speech_cfg)
    m_pause, _ = pauses.summarise(events, r, kept, speech_cfg)
    lexicon = fillers.load_lexicon("en")
    m_fill, _ = fillers.summarise(
        fillers.find_lexical(kept, lexicon), (), kept, r, speech_cfg, verbatim=False
    )
    reps = repetitions.find_repetitions(kept, lexicon, speech_cfg)
    m_rep, _ = repetitions.summarise(reps, kept, r, speech_cfg, verbatim=False)
    for m in (m_rate | m_pause | m_fill | m_rep).values():
        quotes += [e.transcript.text for e in m.evidence if e.transcript is not None]
    assert quotes  # rate.wpm quotes the whole transcript
    for q in quotes:
        assert q in kept.text
        for w in q.split():
            assert w not in TAIL, f"invented word {w!r} quoted"
