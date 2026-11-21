"""Text detectors (P2-b, ADR 0018) on hand-written transcripts, with the fake parser (no model).

Word frequencies come from the installed wordfreq data (no network). A slow test runs the same
talk through the real spaCy model when it is downloaded.
"""

from __future__ import annotations

from typing import Any

import pytest

from presenceai.coach.guardrails import contains_banned_term, contains_verdict_term
from presenceai.rubric import loader
from presenceai.schemas.assessment import Confidence
from presenceai.text import analyze

# 7 sentences, 67 words: opening = first 13 words (S1, S2), closing = last 10 words (S7)
TALK = (
    "Have you ever frozen in front of an audience? "  # hook: question
    "Today I will show that short daily practice improves presentation skills. "  # thesis, intro
    "First, practice builds memory. "  # transition; key phrase "practice"
    "For example, 45 percent of students said ten minutes a day helped. "  # example + statistic
    "Next, imagine your talk is like a song you rehearse. "  # transition, analogy, rare "rehearse"
    "According to one coach, rehearsal is the skill that matters. "  # quotation; "skill" by lemma
    "In conclusion, practice a little every day to grow your skills."  # conclusion; key phrase
)


@pytest.fixture
def run(sb: Any, fake_parser: Any, speech_cfg: Any) -> Any:
    def go(talk: str, language: str = "en") -> Any:
        transcript = sb.spaced(talk, language=language)
        return analyze.analyze_text(transcript, parser=fake_parser, pauses=speech_cfg.pauses)

    return go


def _v(report: Any, key: str) -> float:
    return report.measurements[key].value


def _quotes(report: Any, key: str) -> list[str]:
    return [e.transcript.text for e in report.measurements[key].evidence]


def test_the_sample_talk(run: Any):
    r = run(TALK)
    assert r.sentences == 7 and r.punctuation_reliable
    assert _v(r, "text.central_message.statements") == 1
    assert _v(r, "text.central_message.in_opening") == 1
    assert _v(r, "text.central_message.in_closing") == 1
    assert _v(r, "text.central_message.key_phrase_recurrences") == 2
    assert _v(r, "text.organization.framing_parts") == 2
    assert _v(r, "text.organization.transitions") == 2
    assert _v(r, "text.hook.present") == 1
    assert _v(r, "text.support.items") == 4  # statistic + example, analogy, quotation
    assert _v(r, "text.support.types") == 4
    assert _v(r, "text.language.undefined_terms") == 1
    assert _quotes(r, "text.language.undefined_terms") == ["rehearse."] or _quotes(
        r, "text.language.undefined_terms"
    ) == ["rehearse"]
    assert _v(r, "text.language.mean_sentence_words") == round(67 / 7, 2)
    assert {m.confidence for m in r.measurements.values()} == {Confidence.MEDIUM}


def test_key_phrases_match_by_lemma_not_exact_string(run: Any):
    r = run(TALK)
    quotes = _quotes(r, "text.central_message.key_phrase_recurrences")
    assert any("the skill that matters" in q for q in quotes)  # thesis said "skills"


def test_every_quote_is_exact_and_zero_counts_quote_the_searched_part(run: Any, sb: Any):
    talk = "We met. We talked. It was fine. We left. We slept. We woke. We ate. We went home."
    r = run(talk)
    text = sb.spaced(talk).text
    for key, m in r.measurements.items():
        for e in m.evidence:
            assert e.transcript is not None, key
            assert text[e.transcript.start_char : e.transcript.end_char] == e.transcript.text
    assert _v(r, "text.central_message.in_closing") == 0
    assert _quotes(r, "text.central_message.in_closing") == ["We went home."]  # the closing


def test_transitions_count_once_per_sentence_and_only_near_its_start(run: Any):
    r = run("First and then next we start. We did this first. So, finally, we stop. The end.")
    assert _v(r, "text.organization.transitions") == 2  # "We did this first." does not count


def test_no_hook_and_near_misses(run: Any):
    r = run("So, let me begin. The topic is practice. It matters to us all. Thank you.")
    assert _v(r, "text.hook.present") == 0
    r = run("Most people think practice is boring. But it is not. We will see why. Thanks.")
    assert _v(r, "text.hook.present") == 1  # contrast lead, turn in the next sentence


@pytest.mark.parametrize(
    ("sentence", "items"),
    [
        ("In 2024 we met there.", 0),  # a year is a date, not a statistic
        ("We had 3 rounds and 5 judges.", 1),  # two statistics in one sentence: one item
        ("About 45 % of us agreed.", 1),
        ("We met one person.", 0),  # a number word without digits is not a statistic
        ("For example, 45 percent agreed.", 2),  # one statistic and one example
    ],
)
def test_supporting_items_count_per_sentence_and_type(run: Any, sentence: str, items: int):
    r = run(f"{sentence} Then we left.")
    assert _v(r, "text.support.items") == items


def test_rare_terms_definitions_and_acronyms(run: Any):
    r = run(
        "We use diarization here. It is useful. Then VAD, which means voice activity detection."
    )
    assert _v(r, "text.language.undefined_terms") == 1  # "VAD" is defined in its own sentence
    assert _quotes(r, "text.language.undefined_terms") == ["diarization"]
    r = run("We use diarization here. In other words, we split the speakers. AI and TV are fine.")
    assert (
        _v(r, "text.language.undefined_terms") == 0
    )  # defined in the next sentence; AI, TV common
    r = run("Diarization helps. Diarization works. Diarization scales.")
    assert _v(r, "text.language.undefined_terms") == 1  # counted once, at its first use


def test_unreliable_punctuation_drops_only_sentence_length(run: Any):
    r = run(" ".join(["we practise our talk every single day"] * 6))  # 42 words, no sentence mark
    assert not r.punctuation_reliable
    assert "text.language.mean_sentence_words" in r.not_measurable
    assert "text.language.undefined_terms" in r.measurements
    assert {m.confidence for m in r.measurements.values()} == {Confidence.LOW}


def test_arabic_content_is_not_measurable_never_low(run: Any):
    r = run("اليوم سأتحدث عن التدريب اليومي. أولًا التدريب مهم.", language="ar")
    assert r.measurements == {}
    assert set(r.not_measurable) == set(analyze.MEASURES)
    assert all("FINAL_04" in reason for reason in r.not_measurable.values())


def test_measures_cover_the_rubric(run: Any):
    rubric = loader.load_rubric("s1_presentation")
    planned_here = {k for k, m in rubric.measures.items() if k.startswith("text.")}
    assert planned_here == set(analyze.MEASURES)
    assert all(rubric.measures[k].status == "available" for k in planned_here)


def test_methods_are_observable_wording(run: Any):
    r = run(TALK)
    for key, m in r.measurements.items():
        assert not contains_verdict_term(m.method, "en"), key
        assert not contains_banned_term(m.method, "en"), key


@pytest.mark.slow
def test_real_spacy_on_the_sample_talk(sb: Any, speech_cfg: Any):
    from presenceai.speech.asr import ModelNotAvailable
    from presenceai.text.nlp import spacy_parser

    try:
        parser = spacy_parser()
    except ModelNotAvailable:
        pytest.skip("en_core_web_sm not downloaded (scripts/download_models.py --only spacy_en)")
    r = analyze.analyze_text(sb.spaced(TALK), parser=parser, pauses=speech_cfg.pauses)
    assert _v(r, "text.central_message.in_opening") == 1
    assert _v(r, "text.organization.transitions") == 2
    assert _v(r, "text.hook.present") == 1
    assert _v(r, "text.support.types") == 4


# --- owner's lexicon review (2026-11-21, ADR 0018) --------------------------------------------


def test_every_lexicon_entry_matches_its_own_wording():
    """No entry can silently fail: each survives the transcript's normalisation and matches."""
    from presenceai.lang import load_pack_file
    from presenceai.text.document import find, plain

    raw = load_pack_file("en", "text_cues.yaml")
    for kind, entries in raw.items():
        if kind == "normalise":
            continue
        for entry in entries:
            assert plain(entry), (kind, entry)
            assert plain(plain(entry)) == plain(entry), (kind, entry)
            assert (plain(entry), 0) in find(entry, kind), (kind, entry)
            assert (plain(entry), 0) in find(entry.capitalize() + ".", kind), (kind, entry)


@pytest.mark.parametrize(
    ("said", "expanded"),
    [("Today I'll talk", "today i will talk"), ("I'm gonna try", "i am going to try"),
     ("We’ll see", "we will see"), ("I'd like a", "i would like a")],
)  # fmt: skip
def test_contractions_are_expanded(said: str, expanded: str):
    from presenceai.text.document import plain

    assert plain(said) == expanded


def test_topic_announcement_is_introduction_not_thesis(run: Any):
    r = run("Today I'll talk about habits. Habits matter. We sleep. We eat. We rest. Thanks.")
    assert _v(r, "text.central_message.statements") == 0
    assert _v(r, "text.organization.framing_parts") == 1  # introduction cue
    r = run("Today we will cover sleep. Sleep matters. We rest. We eat. We go. Thanks.")
    assert _v(r, "text.organization.framing_parts") == 1
    r = run("So today I'm gonna talk about sleep. Sleep matters. We rest. We eat. Thanks.")
    assert _v(r, "text.organization.framing_parts") == 1  # "gonna" normalised


def test_that_is_why_is_not_a_definition(run: Any):
    r = run("We use diarization here. That is why we split the audio.")
    assert _v(r, "text.language.undefined_terms") == 1


@pytest.mark.parametrize(
    ("sentence", "analogy"),
    [
        ("I would like a minute.", False),
        ("I'd like a break.", False),
        ("Practice is like a song.", True),
        ("This is similar to a map.", True),
    ],
)
def test_like_a_is_an_analogy_only_when_it_compares(run: Any, sentence: str, analogy: bool):
    r = run(f"{sentence} We left.")
    assert _v(r, "text.support.items") == (1 if analogy else 0)


def test_similar_to_counts_once(run: Any):
    r = run("This is similar to a map, similar to a guide. We left.")
    assert _v(r, "text.support.items") == 1


def test_like_as_a_hesitation_filler_is_not_an_analogy(sb: Any, fake_parser: Any, speech_cfg: Any):
    from presenceai.schemas.speech import FillerEvent, FillerKind

    t = sb.spaced("It was like a big step. We left.")  # "like" is word 2
    hesitation = FillerEvent(kind=FillerKind.HESITATION, first_word=2, last_word=2,
                             start_s=1.0, end_s=1.3)  # fmt: skip
    r = analyze.analyze_text(t, parser=fake_parser, pauses=speech_cfg.pauses, fillers=(hesitation,))
    assert _v(r, "text.support.items") == 0
    marker = hesitation.model_copy(update={"kind": FillerKind.DISCOURSE_MARKER})
    r = analyze.analyze_text(t, parser=fake_parser, pauses=speech_cfg.pauses, fillers=(marker,))
    assert _v(r, "text.support.items") == 1  # discourse markers are every "like" (ADR 0012)


@pytest.mark.parametrize(
    ("sentence", "quotation"),
    [("I said that it works.", False), ("We said that it works.", False),
     ("Our coach said that it works.", True)],
)  # fmt: skip
def test_said_that_needs_another_speaker(run: Any, sentence: str, quotation: bool):
    r = run(f"{sentence} We left.")
    assert _v(r, "text.support.items") == (1 if quotation else 0)


def test_then_counts_only_as_the_first_word(run: Any):
    r = run("We then left. So then we ate. Then we slept. The end.")
    assert _v(r, "text.organization.transitions") == 1


@pytest.mark.parametrize(
    "closing",
    ["In closing, practice daily.", "To close, practice daily.", "To finish, practice daily.",
     "The bottom line is practice.", "Let me leave you with practice."],
)  # fmt: skip
def test_new_conclusion_cues(run: Any, closing: str):
    r = run(f"We met there. We talked a lot. We ate food. We went out. {closing}")
    assert _v(r, "text.organization.framing_parts") == 1
