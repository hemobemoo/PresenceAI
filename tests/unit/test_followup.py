"""S2 follow-up measures (S2-b, ADR 0022): latency, limitation cues, relevance, weaker answer."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from presenceai.pipeline.followup import measure_answer
from presenceai.rubric import levels, loader
from presenceai.schemas.followup import AnswerMeasures
from presenceai.schemas.speech import SpeechRegion
from presenceai.speech.latency import response_latency
from presenceai.text.document import cues, words_of
from presenceai.text.followup import limitation_cue, relevance, summarise

# Common hedging phrases. Hedging is not measured (ADR 0018) and has no lexicon yet, so this
# list stands in for it: no limitation cue may be or contain one (supervisor's condition,
# ADR 0022, written before the follow-up measures go live).
HEDGES = (
    "i am not sure", "i'm not sure", "not sure", "maybe", "perhaps", "i think", "i guess",
    "i believe", "probably", "possibly", "sort of", "kind of", "it seems", "might", "may be",
    "i suppose", "somewhat", "more or less", "to some extent", "i would say",
)  # fmt: skip


def _contains(words: list[str], phrase: list[str]) -> bool:
    n = len(phrase)
    return any(words[i : i + n] == phrase for i in range(len(words) - n + 1))


def test_no_limitation_cue_is_a_hedging_phrase():
    limitation = cues()["limitation"]
    assert limitation, "the limitation lexicon is empty"
    for cue in limitation:
        for hedge in HEDGES:
            assert not _contains(words_of(cue), words_of(hedge)), (cue, hedge)


@pytest.mark.parametrize(
    "answer",
    ["I'm not sure, maybe it would work.", "I think it probably holds.", "It seems fine."],
)
def test_hedged_answers_are_not_limitation_acknowledgements(answer: str):
    assert limitation_cue(answer) is None


@pytest.mark.parametrize(
    ("answer", "cue"),
    [
        ("One limitation is the sample size.", "one limitation"),
        ("We didn't test it on Arabic speakers.", "we didn't test"),
        ("We did not evaluate it with real learners; that is future work.", "we did not evaluate"),
        ("Noise robustness is out of scope here.", "out of scope"),
    ],
)
def test_explicit_limitations_are_found(answer: str, cue: str):
    assert limitation_cue(answer) == cue


def test_latency_is_the_first_speech_onset_or_missing():
    regions = (SpeechRegion(start_s=2.4, end_s=5.0), SpeechRegion(start_s=1.2, end_s=2.0))
    assert response_latency(regions) == 1.2
    assert response_latency(()) is None  # no speech: missing, never low


class _BagOfWords:
    """Fake embedder: normalised word counts over a fixed vocabulary (no model)."""

    name = "fake"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.kinds: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: Sequence[str], kind: str) -> np.ndarray:
        self.kinds.append(kind)
        out = np.zeros((len(texts), 64), np.float32)
        for row, t in enumerate(texts):
            for w in words_of(t):
                out[row, self.vocab.setdefault(w, len(self.vocab) % 64)] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


def test_relevance_is_question_answer_similarity():
    emb = _BagOfWords()
    q = "How did you test the pause detector?"
    assert relevance(q, "How did you test the pause detector?", emb) == pytest.approx(1.0)
    off = relevance(q, "Bananas grow in warm climates.", emb)
    assert off is not None and off < 0.2
    assert emb.kinds[:2] == ["query", "passage"]  # e5 roles: question = query, answer = passage
    # empty text is missing: the text cannot tell silence from a failed transcription
    assert relevance(q, "   ", emb) is None


def test_weaker_answer_decides():
    s = summarise([
        AnswerMeasures(index=0, latency_s=1.0, relevance=0.9, limitation=None),
        AnswerMeasures(index=1, latency_s=3.5, relevance=0.7, limitation="future work"),
    ])  # fmt: skip
    assert (s.relevance_min, s.relevance_min_index) == (0.7, 1)
    assert (s.latency_max_s, s.latency_max_index) == (3.5, 1)
    assert s.limitation_acknowledged and s.limitation_index == 1


def test_missing_values_are_left_out_never_low():
    s = summarise([
        AnswerMeasures(index=0, latency_s=None, relevance=None),
        AnswerMeasures(index=1, latency_s=2.0, relevance=None),
    ])  # fmt: skip
    assert s.relevance_min is None and s.latency_max_s == 2.0
    assert not s.limitation_acknowledged
    with pytest.raises(ValueError):
        summarise([])


@pytest.mark.slow
def test_real_e5_ranks_an_on_topic_answer_above_an_off_topic_one():
    # Needs the e5-small download. Made-up sentences, not calibration data: the owner's 20
    # pairs set the thresholds and stay out of every test (ADR 0022 §5).
    from presenceai.knowledge.embed import E5Embedder

    emb = E5Embedder()
    q = "How did you test the pause detector?"
    on = "We compared detected pauses with the known silences in 300 synthetic clips."
    off = "The cafeteria serves lunch from noon."
    r_on, r_off = relevance(q, on, emb), relevance(q, off, emb)
    assert r_on is not None and r_off is not None and r_on > r_off


# --- "no answer given" is decided from the audio (owner, 2026-11-23, ADR 0022) ---------------

Q = "How did you test the pause detector?"
SPEECH = (SpeechRegion(start_s=1.5, end_s=6.0),)


def _answer(**kw: Any) -> AnswerMeasures:
    args: dict[str, Any] = {"question": Q, "transcript": "", "regions": (),
                            "recording_ok": True, "window_s": 60.0, "elapsed_s": 60.0,
                            "embedder": _BagOfWords()}  # fmt: skip
    args.update(kw)
    return measure_answer(0, **args)


def test_case_1_full_window_of_silence_is_a_measured_non_answer():
    a = _answer()
    assert a.status == "no_answer" and a.relevance == 0.0 and a.latency_s == 60.0
    t = loader.load_thresholds("s2_project_defense")
    assert a.latency_s is not None and a.latency_s > t.values["fu_latency_3"]  # every limit
    d = next(d for d in loader.load_rubric("s2_project_defense").dimensions
             if d.id == "followup_handling")  # fmt: skip
    s = summarise([a])
    values = {"followup.relevance_min": s.relevance_min, "followup.latency_max_s": s.latency_max_s,
              "followup.limitation_acknowledged": float(s.limitation_acknowledged)}  # fmt: skip
    assert levels.dimension_level(d, values, t) == 1  # level 1, not "not measurable"


def test_case_2_speech_without_transcript_leaves_relevance_missing():
    a = _answer(regions=SPEECH)
    assert a.status == "speech_untranscribed"
    assert a.relevance is None and a.latency_s == 1.5  # speech was found: latency is known


def test_case_3_failed_recording_leaves_relevance_and_latency_missing():
    a = _answer(recording_ok=False, regions=SPEECH, transcript="We tested it on clips.")
    assert a.status == "recording_failed" and a.relevance is None and a.latency_s is None


def test_silence_before_the_window_ends_is_not_a_non_answer():
    a = _answer(elapsed_s=20.0)
    assert a.status == "incomplete" and a.relevance is None and a.latency_s is None


def test_answered_measures_everything():
    a = _answer(regions=SPEECH, transcript="We tested the pause detector; one limitation is noise.")
    assert a.status == "answered" and a.latency_s == 1.5
    assert a.relevance is not None and a.relevance > 0 and a.limitation == "one limitation"
    assert _answer(regions=SPEECH, transcript="Yes.", embedder=None).relevance is None  # Arabic
