"""Speech already going when an answer window opens is not an onset (owner, P3-e)."""

from __future__ import annotations

from presenceai.pipeline.followup import measure_answer
from presenceai.schemas.speech import SpeechRegion
from presenceai.speech.latency import response_latency, running_until


def R(a: float, b: float) -> SpeechRegion:
    return SpeechRegion(start_s=a, end_s=b)


def test_answer_1_running_past_its_window_into_question_2():
    """The user is still finishing answer 1 when question 2 appears; they stop 0.8 s after the
    mark, read question 2 and start answering 2.5 s after the mark. Latency is 2.5 s, not 0."""
    preroll = 1.0  # VAD on 1 s before the mark + the window: the mark is at 1.0 s
    with_preroll = (R(0.0, 1.8), R(3.5, 20.0))
    run = running_until(with_preroll, preroll)
    assert run == 0.8
    window = (R(0.0, 0.8), R(2.5, 19.0))  # VAD on the window audio alone
    assert response_latency(window) == 0.0  # the old rule took the running speech as the onset
    assert response_latency(window, run) == 2.5
    a = measure_answer(
        1, question="Why this model?", transcript="because it was small enough", regions=window,
        recording_ok=True, window_s=60.0, elapsed_s=60.0, embedder=None, running_until_s=run,
    )  # fmt: skip
    assert a.status == "answered" and a.latency_s == 2.5


def test_no_running_speech_keeps_the_first_onset():
    assert running_until((R(0.2, 0.9), R(1.4, 5.0)), 1.0) == 0.0  # ended before the mark
    assert running_until((), 1.0) == 0.0
    assert response_latency((R(1.4, 5.0), R(6.0, 7.0)), 0.0) == 1.4


def test_only_running_speech_and_no_new_onset_gives_no_latency():
    run = running_until((R(0.0, 4.0),), 1.0)  # still talking 3 s into the window, then silence
    assert run == 3.0
    assert response_latency((R(0.0, 3.0),), run) is None  # missing, never low
