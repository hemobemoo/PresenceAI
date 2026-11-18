"""Repetitions: a 1–3-word phrase said again straight away (FINAL_04 §2, ADR 0012).

- Words are compared after `normalize_token` (case, edge punctuation; for Arabic also
  diacritics, tatweel, alef forms and stretched letters), the same normalisation as the
  filler lexicon.
- Hesitations ("uh", "امم") are skipped, so "the uh the" is a repetition and "uh uh" is two
  fillers, not a repetition. Discourse markers are ordinary words here: "يعني يعني" is a
  repetition (owner, 2026-11-17).
- The second copy must start within `window_s` of the end of the first. The longest phrase wins,
  and a run ("I I I") is one event with three copies, counted as two repetitions.
- False starts (cut-off words) need a verbatim transcript and are left out until T1 (P4).

With a non-verbatim model the counts are lower bounds, with the same confidence cap as fillers.
"""

from __future__ import annotations

from presenceai.lang import text
from presenceai.schemas.assessment import Confidence, Evidence, Measurement, TimeSpan
from presenceai.schemas.speech import RepetitionEvent, SpeechRegion, Transcript
from presenceai.settings import SpeechSettings
from presenceai.speech.confidence import band, cap
from presenceai.speech.fillers import Lexicon, load_lexicon
from presenceai.text.normalize import normalize_token


def find_repetitions(
    transcript: Transcript, lexicon: Lexicon, cfg: SpeechSettings
) -> tuple[RepetitionEvent, ...]:
    words = transcript.words
    idx: list[int] = []
    keys: list[str] = []
    for i, w in enumerate(words):
        key = normalize_token(w.text, lexicon.language)
        if key and not lexicon.is_hesitation(w.text):
            idx.append(i)
            keys.append(key)

    window = cfg.repetitions.window_s
    events: list[RepetitionEvent] = []
    i = 0
    while i < len(keys):
        found = None
        for n in range(min(cfg.repetitions.max_ngram, (len(keys) - i) // 2), 0, -1):
            phrase = keys[i : i + n]
            copies = 1
            while keys[i + copies * n : i + (copies + 1) * n] == phrase:
                prev_end = words[idx[i + copies * n - 1]].end_s
                next_start = words[idx[i + copies * n]].start_s
                if next_start - prev_end > window + 1e-9:
                    break
                copies += 1
            if copies >= 2:
                found = (n, copies)
                break
        if found is None:
            i += 1
            continue
        n, copies = found
        first, last = idx[i], idx[i + copies * n - 1]
        events.append(
            RepetitionEvent(
                first_word=first,
                last_word=last,
                ngram=n,
                copies=copies,
                start_s=words[first].start_s,
                end_s=words[last].end_s,
            )
        )
        i += copies * n
    return tuple(events)


def summarise(
    events: tuple[RepetitionEvent, ...],
    transcript: Transcript,
    regions: tuple[SpeechRegion, ...],
    cfg: SpeechSettings,
    *,
    verbatim: bool,
) -> tuple[dict[str, Measurement], dict[str, str]]:
    keys = ("repetitions.count", "repetitions.per_100_words")
    words = transcript.words
    if not words or not regions:
        return {}, dict.fromkeys(keys, text(transcript.language, "common.no_words"))
    lang = transcript.language
    span = TimeSpan(start_s=regions[0].start_s, end_s=regions[-1].end_s)
    mean_p = sum(w.probability for w in words) / len(words)
    conf = band(cfg.confidence, probability=mean_p, speaking_time_s=span.end_s - span.start_s)
    bound = ""
    if not verbatim:
        conf = cap(conf, Confidence(cfg.fillers.confidence_cap_without_verbatim[lang]))
        bound = text(lang, "repetitions.lower_bound")
    count = sum(e.copies - 1 for e in events)
    lexicon = load_lexicon(lang)
    base_words = sum(1 for w in words if not lexicon.is_hesitation(w.text)) or len(words)
    evidence = tuple(
        Evidence(
            transcript=transcript.span(e.first_word, e.last_word + 1),
            time=TimeSpan(start_s=e.start_s, end_s=e.end_s),
        )
        for e in events
    ) or (Evidence(time=span),)
    method = text(
        lang,
        "repetitions.method",
        max_ngram=cfg.repetitions.max_ngram,
        window_s=cfg.repetitions.window_s,
    )

    def measure(value: float, unit: str, how: str) -> Measurement:
        return Measurement(
            value=value,
            unit=unit,
            confidence=conf,
            method=how + bound,
            evidence=evidence,
            language=lang,
            live=False,
        )

    return {
        "repetitions.count": measure(count, "count", method),
        "repetitions.per_100_words": measure(
            count * 100.0 / base_words,
            "per 100 words",
            method + text(lang, "repetitions.per_100_suffix"),
        ),
    }, {}
