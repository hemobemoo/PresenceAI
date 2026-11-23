"""S2 follow-up measures (S2-b, ADR 0022): latency, limitation cues, relevance, weaker answer."""

from collections.abc import Sequence

import numpy as np
import pytest

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
    assert relevance(q, "Bananas grow in warm climates.", emb) < 0.2
    assert emb.kinds[:2] == ["query", "passage"]  # e5 roles: question = query, answer = passage
    assert relevance(q, "   ", emb) == 0.0  # an empty answer is a measured non-answer


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
    assert relevance(q, on, emb) > relevance(q, off, emb)
