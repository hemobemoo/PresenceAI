"""Fillers (FINAL_04 §2, §5 as changed by ADR 0012).

Two detectors:

1. **Lexicon** on the transcript, with lists in `lang/<lang>/fillers.yaml`:
   - hesitations ("um", "uh", "امم", ...) are counted as fillers;
   - discourse markers ("like", "you know", "يعني", "طيب", ...) are reported separately as
     information only and never penalised. Every occurrence is counted, including ordinary uses,
     so marker counts are **upper bounds**.
2. **Acoustic candidates**: voiced sound inside a VAD speech region, with no recognised word on
   it, at least `min_duration_s` long, with pitch steadier than `max_pitch_sd_semitones`. These
   values are PROVISIONAL (ADR 0012). Whisper word timings decide what counts as "no word": a
   word stretched over a hesitation hides it (see the test and LIMITATIONS.md).

Base Whisper drops many fillers, so with a non-verbatim model the result is reported as **"at
least N per 100 words"** with capped confidence (English medium, Arabic low), and acoustic
candidates without a lexicon filler are shown only as extra, low-confidence candidates. No range
is claimed: on a real clip, 0.86–1.54 per 100 words did not contain the true 3.0. Only with a
verbatim model (T1) is the result a range (lexicon to lexicon + acoustic-only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from presenceai.lang import Language, load_pack_file
from presenceai.schemas.assessment import Confidence, Evidence, Measurement, TimeSpan
from presenceai.schemas.speech import (
    AcousticCandidate,
    FillerEvent,
    FillerKind,
    PitchTrack,
    SpeechRegion,
    Transcript,
)
from presenceai.settings import SpeechSettings
from presenceai.speech.confidence import band, cap
from presenceai.text.normalize import filler_key


@dataclass(frozen=True)
class Lexicon:
    language: Language
    hesitations: tuple[tuple[str, ...], ...]  # each entry as a tuple of filler keys
    discourse_markers: tuple[tuple[str, ...], ...]

    def is_hesitation(self, token: str) -> bool:
        return (filler_key(token, self.language),) in self.hesitations


def load_lexicon(language: Language) -> Lexicon:
    data = load_pack_file(language, "fillers.yaml")

    def keys(entries: object) -> tuple[tuple[str, ...], ...]:
        out = []
        for entry in cast("list[str]", entries or []):
            out.append(tuple(filler_key(t, language) for t in str(entry).split()))
        return tuple(sorted(set(out), key=len, reverse=True))  # longest first

    return Lexicon(language, keys(data.get("hesitations")), keys(data.get("discourse_markers")))


def find_lexical(transcript: Transcript, lexicon: Lexicon) -> tuple[FillerEvent, ...]:
    words = transcript.words
    toks = [filler_key(w.text, lexicon.language) for w in words]
    entries = [(e, FillerKind.HESITATION) for e in lexicon.hesitations] + [
        (e, FillerKind.DISCOURSE_MARKER) for e in lexicon.discourse_markers
    ]
    entries.sort(key=lambda ek: len(ek[0]), reverse=True)
    events: list[FillerEvent] = []
    i = 0
    while i < len(toks):
        for entry, kind in entries:
            n = len(entry)
            if tuple(toks[i : i + n]) == entry:
                events.append(
                    FillerEvent(
                        kind=kind,
                        first_word=i,
                        last_word=i + n - 1,
                        start_s=words[i].start_s,
                        end_s=words[i + n - 1].end_s,
                    )
                )
                i += n
                break
        else:
            i += 1
    return tuple(events)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index runs where mask is True."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def acoustic_candidates(
    regions: tuple[SpeechRegion, ...],
    transcript: Transcript,
    pitch: PitchTrack,
    cfg: SpeechSettings,
    lexical: tuple[FillerEvent, ...],
) -> tuple[AcousticCandidate, ...]:
    a = cfg.fillers.acoustic
    n = len(pitch.voiced)
    if n == 0 or not regions:
        return ()
    times = np.arange(n) * pitch.hop_s
    in_speech = np.zeros(n, dtype=bool)
    for r in regions:
        in_speech |= (times >= r.start_s) & (times < r.end_s)
    covered = np.zeros(n, dtype=bool)
    for w in transcript.words:
        covered |= (times >= w.start_s) & (times <= w.end_s)
    voiced = np.array(pitch.voiced, dtype=bool)
    f0 = np.array(pitch.f0_hz, dtype=float)

    free = in_speech & ~covered
    seed = free & voiced
    # merge voiced runs separated by short unvoiced dropouts (still inside free frames)
    gap = round(a.max_unvoiced_gap_s / pitch.hop_s)
    merged = seed.copy()
    for start, stop in _runs(~seed):
        if start > 0 and stop < n and stop - start <= gap and free[start:stop].all():
            merged[start:stop] = True

    hesitations = [e for e in lexical if e.kind == FillerKind.HESITATION]
    out: list[AcousticCandidate] = []
    for start, stop in _runs(merged):
        duration = (stop - start) * pitch.hop_s
        v = voiced[start:stop]
        if duration < a.min_duration_s - 1e-9 or v.mean() < a.min_voiced_fraction:
            continue
        semis = 12 * np.log2(f0[start:stop][v] / 100.0)
        sd = float(semis.std()) if semis.size > 1 else 0.0
        if sd > a.max_pitch_sd_semitones:
            continue
        s, e = float(times[start]), float(times[start] + duration)
        tol = a.agreement_tolerance_s
        agrees = any(h.start_s - tol <= e and s <= h.end_s + tol for h in hesitations)
        out.append(
            AcousticCandidate(
                start_s=s,
                end_s=e,
                voiced_fraction=float(v.mean()),
                pitch_sd_semitones=sd,
                agrees_with_lexicon=agrees,
            )
        )
    return tuple(out)


def summarise(
    lexical: tuple[FillerEvent, ...],
    acoustic: tuple[AcousticCandidate, ...],
    transcript: Transcript,
    regions: tuple[SpeechRegion, ...],
    cfg: SpeechSettings,
    *,
    verbatim: bool,
) -> tuple[dict[str, Measurement], dict[str, str]]:
    keys = tuple(k for k in _KEYS_ALL if verbatim or not k.endswith("_upper"))
    words = transcript.words
    hes = [e for e in lexical if e.kind == FillerKind.HESITATION]
    hes_words = sum(e.last_word - e.first_word + 1 for e in hes)
    base_words = len(words) - hes_words
    if base_words <= 0 or not regions:
        return {}, dict.fromkeys(keys, "no recognised words")

    lang = transcript.language
    span = TimeSpan(start_s=regions[0].start_s, end_s=regions[-1].end_s)
    mean_p = sum(w.probability for w in words) / len(words)
    conf = band(cfg.confidence, probability=mean_p, speaking_time_s=span.end_s - span.start_s)
    bound = ""
    if not verbatim:
        conf = cap(conf, Confidence(cfg.fillers.confidence_cap_without_verbatim[lang]))
        bound = " (lower bound: the base model drops fillers)"

    def ev(events: list[FillerEvent]) -> tuple[Evidence, ...]:
        out = tuple(
            Evidence(
                transcript=transcript.span(e.first_word, e.last_word + 1),
                time=TimeSpan(start_s=e.start_s, end_s=e.end_s),
            )
            for e in events
        )
        return out or (Evidence(time=span),)

    def measure(
        value: float, unit: str, c: Confidence, method: str, evidence: tuple[Evidence, ...]
    ) -> Measurement:
        return Measurement(
            value=value,
            unit=unit,
            confidence=c,
            method=method,
            evidence=evidence,
            language=lang,
            live=False,
        )

    acoustic_only = [c for c in acoustic if not c.agrees_with_lexicon]
    cand_ev = tuple(Evidence(time=TimeSpan(start_s=c.start_s, end_s=c.end_s)) for c in acoustic)
    only_ev = tuple(
        Evidence(time=TimeSpan(start_s=c.start_s, end_s=c.end_s)) for c in acoustic_only
    )
    markers = [e for e in lexical if e.kind == FillerKind.DISCOURSE_MARKER]
    per100 = 100.0 / base_words
    rate = len(hes) * per100
    upper_bound = (
        " (upper bound: every occurrence counted, including ordinary uses; information only)"
    )
    m: dict[str, Measurement] = {
        "fillers.count": measure(len(hes), "count", conf, "lexicon hesitations" + bound, ev(hes)),
    }
    if verbatim:
        # a verbatim transcript keeps fillers, so lexicon .. lexicon + acoustic-only is a range
        m["fillers.per_100_words"] = measure(
            rate, "per 100 words", conf, "lexicon hesitations / non-filler words", ev(hes)
        )
        m["fillers.per_100_words_upper"] = measure(
            (len(hes) + len(acoustic_only)) * per100,
            "per 100 words",
            Confidence.LOW,
            "upper end of the range: lexicon + acoustic candidates without a lexicon filler"
            " (provisional detector)",
            ev(hes) + only_ev,
        )
    else:
        # Base Whisper drops fillers: on a real clip the lexicon gave 0.86 and lexicon + acoustic
        # 1.54 per 100 words against a true 3.0, so no range is claimed (ADR 0012).
        m["fillers.per_100_words"] = measure(
            rate,
            "per 100 words (at least)",
            conf,
            f"at least {rate:.1f} per 100 words: lexicon hesitations / non-filler words"
            f"{bound}; the true rate can be much higher",
            ev(hes),
        )
    m |= {
        "fillers.acoustic_only_candidates": measure(
            len(acoustic_only),
            "count",
            Confidence.LOW,
            "extra candidates, not counted as fillers: voiced, steady-pitch sound with no word"
            " and no lexicon filler nearby (provisional detector)",
            only_ev or (Evidence(time=span),),
        ),
        "fillers.acoustic_candidates": measure(
            len(acoustic),
            "count",
            Confidence.LOW,
            "voiced, steady-pitch sound with no word (provisional detector)",
            cand_ev or (Evidence(time=span),),
        ),
        "discourse_markers.count": measure(
            len(markers),
            "count",
            Confidence.LOW,
            "lexicon discourse markers" + upper_bound,
            ev(markers),
        ),
        "discourse_markers.per_100_words": measure(
            len(markers) * per100,
            "per 100 words",
            Confidence.LOW,
            "lexicon discourse markers / non-filler words" + upper_bound,
            ev(markers),
        ),
    }
    return m, {}


_KEYS_ALL = (
    "fillers.count",
    "fillers.per_100_words",
    "fillers.per_100_words_upper",  # only with a verbatim model
    "fillers.acoustic_only_candidates",
    "fillers.acoustic_candidates",
    "discourse_markers.count",
    "discourse_markers.per_100_words",
)
