"""Pauses: gaps between VAD speech regions, placed in the transcript (FINAL_04 §2, ADR 0011).

A gap counts as a pause when it lasts at least the smallest threshold in `speech.pauses`. Its
position comes from the word before it:

- `sentence_boundary` after `. ! ? ؟ …`, `clause_boundary` after `, ، ; ؛` (both reported and
  never penalised), otherwise `mid_sentence`;
- `unknown` (confidence low) when the transcript has fewer than one sentence mark per
  `max_words_per_sentence_mark` words, so boundaries cannot be placed reliably (Arabic is the
  main case, but the rule applies to any language), or when no recognised word sits on one side.

The position's confidence follows the lower word probability of the two neighbouring words.
"""

from __future__ import annotations

import itertools

import numpy as np

from presenceai.lang import text
from presenceai.schemas.assessment import Confidence, Evidence, Measurement, TimeSpan
from presenceai.schemas.speech import PauseEvent, PausePosition, SpeechRegion, Transcript, Word
from presenceai.settings import PauseSettings, SpeechSettings
from presenceai.speech.confidence import band

_CLOSERS = "\"'”’»)]}"


def _last_mark(token: str) -> str:
    stripped = token.rstrip(_CLOSERS)
    return stripped[-1:] if stripped else ""


def position_after(token: str, cfg: PauseSettings) -> PausePosition:
    mark = _last_mark(token)
    if mark and mark in cfg.sentence_marks:
        return PausePosition.SENTENCE_BOUNDARY
    if mark and mark in cfg.clause_marks:
        return PausePosition.CLAUSE_BOUNDARY
    return PausePosition.MID_SENTENCE


def punctuation_reliable(transcript: Transcript, cfg: PauseSettings) -> bool:
    words = transcript.words
    if not words:
        return False
    marks = sum(1 for w in words if _last_mark(w.text) in cfg.sentence_marks)
    return marks * cfg.max_words_per_sentence_mark >= len(words)


def _mid(w: Word) -> float:
    return (w.start_s + w.end_s) / 2


def find_pauses(
    regions: tuple[SpeechRegion, ...], transcript: Transcript, cfg: SpeechSettings
) -> tuple[PauseEvent, ...]:
    shortest = min(cfg.pauses.thresholds_s)
    words = transcript.words
    mids = [_mid(w) for w in words]
    reliable = punctuation_reliable(transcript, cfg.pauses)
    events: list[PauseEvent] = []
    for a, b in itertools.pairwise(regions):
        if b.start_s - a.end_s < shortest - 1e-9:
            continue
        before = max((i for i, m in enumerate(mids) if m <= a.end_s), default=None)
        after = min((i for i, m in enumerate(mids) if m >= b.start_s), default=None)
        if not reliable or before is None or after is None:
            position, conf = PausePosition.UNKNOWN, Confidence.LOW
        else:
            position = position_after(words[before].text, cfg.pauses)
            p = min(words[before].probability, words[after].probability)
            conf = band(cfg.confidence, probability=p, speaking_time_s=None)
        events.append(
            PauseEvent(
                start_s=a.end_s,
                end_s=b.start_s,
                position=position,
                confidence=conf,
                word_before=before,
                word_after=after,
            )
        )
    return tuple(events)


def _evidence(e: PauseEvent, transcript: Transcript) -> Evidence:
    quote = None
    if e.word_before is not None and e.word_after is not None:
        quote = transcript.span(e.word_before, e.word_after + 1)
    return Evidence(transcript=quote, time=TimeSpan(start_s=e.start_s, end_s=e.end_s))


def _label(t: float) -> str:
    return f"{t:g}"


def summarise(
    events: tuple[PauseEvent, ...],
    regions: tuple[SpeechRegion, ...],
    transcript: Transcript,
    cfg: SpeechSettings,
) -> tuple[dict[str, Measurement], dict[str, str]]:
    keys_count = [f"pauses.count_ge_{_label(t)}s" for t in cfg.pauses.thresholds_s]
    keys_pos = [f"pauses.count.{p.value}" for p in PausePosition]
    keys_stats = ["pauses.mean_s", "pauses.p90_s", "pauses.longest_s"]
    lang = transcript.language
    if not regions:
        reason = text(lang, "common.no_speech")
        return {}, dict.fromkeys(keys_count + keys_pos + keys_stats, reason)

    span = TimeSpan(start_s=regions[0].start_s, end_s=regions[-1].end_s)
    words = transcript.words
    mean_p = sum(w.probability for w in words) / len(words) if words else None
    conf = band(cfg.confidence, probability=mean_p, speaking_time_s=span.end_s - span.start_s)

    def measure(value: float, unit: str, method: str, chosen: list[PauseEvent]) -> Measurement:
        evidence = tuple(_evidence(e, transcript) for e in chosen) or (Evidence(time=span),)
        return Measurement(
            value=value,
            unit=unit,
            confidence=conf,
            method=method,
            evidence=evidence,
            language=lang,
            live=False,
        )

    out: dict[str, Measurement] = {}
    for t, key in zip(cfg.pauses.thresholds_s, keys_count, strict=True):
        chosen = [e for e in events if e.duration_s >= t - 1e-9]
        out[key] = measure(
            len(chosen), "count", text(lang, "pauses.count_ge", threshold=_label(t)), chosen
        )
    for p, key in zip(PausePosition, keys_pos, strict=True):
        chosen = [e for e in events if e.position == p]
        position = text(lang, f"positions.{p.value}")
        out[key] = measure(
            len(chosen), "count", text(lang, "pauses.by_position", position=position), chosen
        )

    missing: dict[str, str] = {}
    if events:
        durations = np.array([e.duration_s for e in events])
        longest = max(events, key=lambda e: e.duration_s)
        out["pauses.mean_s"] = measure(
            float(durations.mean()), "s", text(lang, "pauses.mean"), list(events)
        )
        out["pauses.p90_s"] = measure(
            float(np.percentile(durations, 90)), "s", text(lang, "pauses.p90"), list(events)
        )
        out["pauses.longest_s"] = measure(
            longest.duration_s, "s", text(lang, "pauses.longest"), [longest]
        )
    else:
        shortest = _label(min(cfg.pauses.thresholds_s))
        missing = dict.fromkeys(keys_stats, text(lang, "pauses.no_pauses", shortest=shortest))
    return out, missing
