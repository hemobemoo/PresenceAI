"""Measures the rubric derives from the speech report (P2-c, ADR 0019).

`pauses.mid_sentence_per_min`: pauses of at least `mid_pause_min_s` (configs/rubric.yaml,
provisional) labelled `mid_sentence` (ADR 0011), per minute of speaking time. Speaking time is
the first speech onset to the last offset, as for `rate.wpm`; below the rate's minimum speaking
time it is not measurable. It is missing for Arabic and whenever the transcript's punctuation is
unreliable, because pause positions then cannot be told (ADR 0011, ADR 0016); English positions
rest on Whisper's punctuation (ADR 0014). Pauses with no word on one side are not counted.

`timing.overrun_s` (P3-a, ADR 0019 amendment): seconds from the session's time limit to the end
of the last speech region, 0 when the speech ends within it. The recording's start is time 0, so
it is measured only for recordings made in the app, which start with the session; an uploaded
file has no time limit and the measure is missing ("uploaded file: no time limit"). Medium
confidence: the end comes from the voice detector, and speech after the talk (a remark to
oneself) counts as speech.
"""

from __future__ import annotations

from presenceai.lang import text
from presenceai.schemas.assessment import Confidence, Evidence, Measurement, TimeSpan
from presenceai.schemas.speech import PausePosition, SpeechReport
from presenceai.settings import SpeechSettings
from presenceai.speech.pauses import punctuation_reliable

KEY = "pauses.mid_sentence_per_min"
OVERRUN_KEY = "timing.overrun_s"


def overrun(report: SpeechReport, limit_s: float | None) -> tuple[Measurement | None, str | None]:
    """Seconds spoken beyond `limit_s`, or (None, reason). `limit_s` None: an uploaded file."""
    lang = report.transcript.language
    if limit_s is None:
        return None, text(lang, "timing.no_limit_upload")
    if not report.speech_regions:
        return None, text(lang, "common.no_speech")
    end = report.speech_regions[-1].end_s
    value = max(0.0, end - limit_s)
    span = TimeSpan(start_s=limit_s, end_s=end) if value > 0 else TimeSpan(start_s=0.0, end_s=end)
    return (
        Measurement(
            value=round(value, 3),
            unit="s",
            confidence=Confidence.MEDIUM,
            method=text(lang, "timing.overrun", limit=limit_s),
            evidence=(Evidence(time=span),),
            language=lang,
            live=False,
        ),
        None,
    )


def mid_sentence_per_min(
    report: SpeechReport, cfg: SpeechSettings, min_s: float
) -> tuple[Measurement | None, str | None]:
    """The measurement, or (None, reason) when it is not measurable."""
    lang = report.transcript.language
    regions = report.speech_regions
    if lang != "en":
        return None, text(lang, "pauses.mid_per_min_language")
    if not regions:
        return None, text(lang, "common.no_speech")
    first, last = regions[0].start_s, regions[-1].end_s
    speaking = last - first
    if speaking < cfg.rate.min_speaking_time_s:
        reason = text(
            lang,
            "common.too_little_speech",
            speech_s=speaking,
            minimum=cfg.rate.min_speaking_time_s,
        )
        return None, reason
    if not punctuation_reliable(report.transcript, cfg.pauses):
        return None, text(lang, "pauses.mid_per_min_punctuation")
    chosen = [
        e
        for e in report.pauses
        if e.position == PausePosition.MID_SENTENCE and e.end_s - e.start_s >= min_s - 1e-9
    ]
    evidence = tuple(
        Evidence(
            time=TimeSpan(start_s=e.start_s, end_s=e.end_s),
            transcript=report.transcript.span(e.word_before, e.word_after + 1)
            if e.word_before is not None and e.word_after is not None
            else None,
        )
        for e in chosen
    ) or (Evidence(time=TimeSpan(start_s=first, end_s=last)),)
    return (
        Measurement(
            value=round(len(chosen) / speaking * 60, 3),
            unit="per min",
            confidence=Confidence.MEDIUM,
            method=text(lang, "pauses.mid_per_min", minimum=min_s),
            evidence=evidence,
            language=lang,
            live=False,
        ),
        None,
    )
