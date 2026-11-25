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


def test_a_window_with_only_in_flight_speech_is_silent():
    """Found in the browser (2026-11-25): the audio still in flight when the question appears
    carries the end of what was said before; a window with nothing new is still silent."""
    import numpy as np

    from presenceai.live.s2_audio import answer_is_silent

    rate = 16_000

    def vad(x: np.ndarray) -> tuple[SpeechRegion, ...]:
        """Each run of loud samples is one speech region."""
        loud = np.concatenate(([False], np.abs(x) > 0.01, [False]))
        edges = np.flatnonzero(np.diff(loud.astype(int)))
        return tuple(R(a / rate, b / rate) for a, b in zip(edges[::2], edges[1::2], strict=True))

    preroll = rate  # 1 s before the mark
    audio = np.zeros(preroll + 6 * rate, dtype=np.float32)
    audio[: preroll + rate // 4] = 0.3  # speech up to 0.25 s after the mark, then silence
    assert answer_is_silent(audio, preroll, rate, vad)
    audio[preroll + 3 * rate : preroll + 4 * rate] = 0.3  # a new answer 3 s into the window
    assert not answer_is_silent(audio, preroll, rate, vad)


def test_a_full_window_with_only_running_speech_scores_as_no_answer():
    a = measure_answer(
        0, question="Why?", transcript="and that is why", regions=(R(0.0, 0.25),),
        recording_ok=True, window_s=60.0, elapsed_s=60.0, embedder=None, running_until_s=0.25,
    )  # fmt: skip
    assert a.status == "no_answer" and a.latency_s == 60.0 and a.relevance == 0.0
