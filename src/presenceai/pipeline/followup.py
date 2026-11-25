"""One S2 follow-up answer from its recording (ADR 0022): status, latency, relevance, limitation.

"No answer given" is decided from the audio, never from an empty transcript (owner,
2026-11-23): the recording passed its checks, the full answer window elapsed and the voice
detector found no speech. Then relevance is 0.0 and the latency is the whole window, which is
longer than every latency limit, so the answer fails every follow-up condition and the dimension
scores level 1. Speech without a transcript, an early stop or a failed recording give missing
values instead (the partial-evidence wording applies, ADR 0019). P3 offers a retry before a
non-answer is scored.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from presenceai.knowledge.embed import Embedder
from presenceai.lang import text
from presenceai.schemas.assessment import Confidence, Evidence, Measurement, TimeSpan
from presenceai.schemas.followup import AnswerMeasures, FollowupReport, FollowupSummary
from presenceai.schemas.speech import SpeechRegion, SpeechReport, Transcript
from presenceai.speech.latency import response_latency
from presenceai.text.document import words_of
from presenceai.text.followup import limitation_cue, relevance, summarise

RELEVANCE = "followup.relevance_min"
LATENCY = "followup.latency_max_s"
LIMITATION = "followup.limitation_acknowledged"
# Measured statuses for the text measures: an answer with a transcript, or a measured non-answer.
_TEXT_MEASURED = ("answered", "no_answer")


def measure_answer(
    index: int,
    *,
    question: str,
    transcript: str,
    regions: Sequence[SpeechRegion],
    recording_ok: bool,
    window_s: float,
    elapsed_s: float,
    embedder: Embedder | None,
    running_until_s: float = 0.0,
) -> AnswerMeasures:
    """`embedder` None: relevance is not computed (e.g. Arabic, ADR 0022 item 6).
    `running_until_s`: speech already going when the window opened ends here; it is not an
    onset (P3-e, `speech/latency.py`)."""
    if not recording_ok:
        return AnswerMeasures(index=index, window_s=window_s, status="recording_failed")
    if not regions:
        if elapsed_s >= window_s:
            return AnswerMeasures(
                index=index, window_s=window_s, status="no_answer", latency_s=window_s,
                relevance=0.0,
            )  # fmt: skip
        return AnswerMeasures(index=index, window_s=window_s, status="incomplete")
    latency = response_latency(regions, running_until_s)
    if not transcript.strip():
        return AnswerMeasures(
            index=index, window_s=window_s, status="speech_untranscribed", latency_s=latency
        )
    return AnswerMeasures(
        index=index,
        window_s=window_s,
        status="answered",
        latency_s=latency,
        relevance=relevance(question, transcript, embedder) if embedder is not None else None,
        limitation=limitation_cue(transcript),
    )


def _answer_evidence(a: AnswerMeasures, t: Transcript | None) -> Evidence:
    """The whole answer: its words (quote and time), or the answer window for a non-answer."""
    if a.status == "no_answer" and a.window_s is not None:
        return Evidence(time=TimeSpan(start_s=0.0, end_s=a.window_s), answer=a.index)
    if t is None:
        raise ValueError(f"answer {a.index} has no transcript to quote")
    words = t.words
    if not words:  # text without timed words: the whole recording
        return Evidence(time=TimeSpan(start_s=0.0, end_s=t.duration_s), answer=a.index)
    return Evidence(
        transcript=t.span(0, len(words)),
        time=TimeSpan(start_s=words[0].start_s, end_s=words[-1].end_s),
        answer=a.index,
    )


def _cue_evidence(a: AnswerMeasures, t: Transcript, cue: str) -> Evidence:
    """The words of the limitation cue in the answer (quote and time); the whole answer if the
    cue's words cannot be lined up with the transcript words."""
    want = words_of(cue)
    flat = [(part, i) for i, w in enumerate(t.words) for part in words_of(w.text)]
    for k in range(len(flat) - len(want) + 1):
        if [p for p, _ in flat[k : k + len(want)]] == want:
            first, last = flat[k][1], flat[k + len(want) - 1][1]
            words = t.words
            return Evidence(
                transcript=t.span(first, last + 1),
                time=TimeSpan(start_s=words[first].start_s, end_s=words[last].end_s),
                answer=a.index,
            )
    return _answer_evidence(a, t)


def _measurements(
    s: FollowupSummary, transcripts: Sequence[Transcript | None], lang: Literal["en", "ar"]
) -> tuple[dict[str, Measurement], dict[str, str]]:
    """The three rubric measures, each with evidence from the answer that decides it."""
    found: dict[str, Measurement] = {}
    missing: dict[str, str] = {}
    by_index = {a.index: a for a in s.answers}
    not_computed = text(lang, "assessment.missing_other")

    def m(value: float, unit: str, conf: Confidence, method: str, ev: tuple[Evidence, ...]):
        return Measurement(
            value=value, unit=unit, confidence=conf, method=method, evidence=ev, language=lang,
            live=False,
        )  # fmt: skip

    if s.latency_max_s is not None and s.latency_max_index is not None:
        a = by_index[s.latency_max_index]
        span = TimeSpan(start_s=0.0, end_s=s.latency_max_s)
        found[LATENCY] = m(
            s.latency_max_s, "s", Confidence.MEDIUM,
            "first speech onset (voice activity detection) after the question ends; the slowest "
            "answer; a non-answer counts as its whole window",
            (Evidence(time=span, answer=a.index),),
        )  # fmt: skip
    else:
        missing[LATENCY] = not_computed

    if lang != "en":
        why = text(lang, "assessment.followup_not_measured_language", language=lang)
        missing[RELEVANCE] = missing[LIMITATION] = why
        return found, missing

    if s.relevance_min is not None and s.relevance_min_index is not None:
        a = by_index[s.relevance_min_index]
        found[RELEVANCE] = m(
            s.relevance_min, "similarity", Confidence.LOW,
            "cosine similarity of question (e5 query) and answer (e5 passage), "
            "multilingual-e5-small; the lowest over the answers; a non-answer scores 0",
            (_answer_evidence(a, transcripts[a.index]),),
        )  # fmt: skip
    else:
        missing[RELEVANCE] = not_computed

    measured = [a for a in s.answers if a.status in _TEXT_MEASURED]
    if s.limitation_acknowledged and s.limitation_index is not None:
        a = by_index[s.limitation_index]
        t = transcripts[a.index]
        if t is None or a.limitation is None:
            raise ValueError(f"answer {a.index}: a limitation cue needs its transcript")
        ev: tuple[Evidence, ...] = (_cue_evidence(a, t, a.limitation),)
        found[LIMITATION] = m(1.0, "yes/no", Confidence.MEDIUM, _LIMITATION_METHOD, ev)
    elif measured:
        ev = tuple(_answer_evidence(a, transcripts[a.index]) for a in measured)
        found[LIMITATION] = m(0.0, "yes/no", Confidence.MEDIUM, _LIMITATION_METHOD, ev)
    else:
        missing[LIMITATION] = not_computed
    return found, missing


_LIMITATION_METHOD = "explicit limitation phrase (lexicon lang/en/text_cues.yaml: limitation)"


def followup_report(
    questions: Sequence[str],
    reports: Sequence[SpeechReport | None],
    windows: Sequence[float],
    *,
    language: Literal["en", "ar"],
    embedder: Embedder | None,
    sources: Sequence[str] = (),
    running_until: Sequence[float] = (),
) -> FollowupReport:
    """Measure every answer and build the follow-up rubric measures.

    `reports[i]` None: the recording could not be read (recording_failed). `embedder` None or
    an Arabic talk: relevance is not computed (ADR 0022 item 6)."""
    if not (len(questions) == len(reports) == len(windows)) or not questions:
        raise ValueError("one question, one recording and one window per follow-up")
    running = tuple(running_until) or (0.0,) * len(questions)
    if len(running) != len(questions):
        raise ValueError("one running-speech end per follow-up")
    answers = [
        measure_answer(
            i,
            question=q,
            transcript=r.transcript.text if r is not None else "",
            regions=r.speech_regions if r is not None else (),
            recording_ok=r is not None,
            window_s=w,
            elapsed_s=r.transcript.duration_s if r is not None else 0.0,
            embedder=embedder if language == "en" else None,
            running_until_s=run,
        )
        for i, (q, r, w, run) in enumerate(zip(questions, reports, windows, running, strict=True))
    ]
    transcripts = tuple(r.transcript if r is not None else None for r in reports)
    summary = summarise(answers)
    found, missing = _measurements(summary, transcripts, language)
    names = tuple(sources) or tuple(
        r.source_name if r is not None else f"answer_{i + 1}" for i, r in enumerate(reports)
    )
    return FollowupReport(
        questions=tuple(questions),
        sources=tuple(Path(n).name for n in names),
        transcripts=transcripts,
        summary=summary,
        measurements=found,
        not_measurable=missing,
    )
