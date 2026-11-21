"""The transcript as sentences, regions and cue matches, shared by the text detectors (ADR 0018).

- **Sentences** come from the transcript's punctuation (a word ending in a sentence mark ends a
  sentence, `speech.pauses.sentence_marks`), as the rubric's definitions say (ADR 0017). English
  sentence positions therefore rest on Whisper's punctuation (ADR 0014).
- **Opening, closing, body** are shares of the words (`configs/rubric.yaml`: opening 20 %,
  closing 15 %). A sentence is in the opening when its first word is, in the closing when its
  last word is, and in the body otherwise.
- **Cues** are lexicon phrases (`lang/en/text_cues.yaml`) matched on whole words, ignoring case
  and punctuation. They are lexicon matches, not an understanding of meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

from presenceai.lang import load_pack_file
from presenceai.schemas.assessment import TranscriptSpan
from presenceai.schemas.speech import Transcript
from presenceai.settings import PauseSettings
from presenceai.speech.pauses import punctuation_reliable
from presenceai.text.nlp import Parser, Tok

_CLOSERS = "\"'”’»)]}"
_NON_WORD = re.compile(r"[^\w']+")


def plain(text: str) -> str:
    """Lower case, punctuation as spaces, single spaces: the form cues are matched on."""
    return " ".join(_NON_WORD.sub(" ", text.casefold()).split())


@cache
def cues() -> dict[str, tuple[str, ...]]:
    data = load_pack_file("en", "text_cues.yaml")
    return {k: tuple(plain(p) for p in v) for k, v in data.items()}


def match(text: str, kind: str, *, within_first_words: int | None = None) -> str | None:
    """The first cue of `kind` found in `text` as whole words, or None."""
    s = plain(text)
    for phrase in cues()[kind]:
        m = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", s)
        if m and (within_first_words is None or len(s[: m.start()].split()) < within_first_words):
            return phrase
    return None


@dataclass(frozen=True)
class Sentence:
    index: int
    first_word: int  # indices into Transcript.words, inclusive
    last_word: int
    start_char: int  # offsets into Transcript.text
    end_char: int
    text: str
    tokens: tuple[Tok, ...]  # offsets relative to `text`

    @property
    def n_words(self) -> int:
        return self.last_word - self.first_word + 1

    def tok_span(self, tok: Tok, transcript: Transcript) -> TranscriptSpan:
        a, b = self.start_char + tok.start, self.start_char + tok.end
        return TranscriptSpan(start_char=a, end_char=b, text=transcript.text[a:b])


@dataclass(frozen=True)
class Document:
    transcript: Transcript
    sentences: tuple[Sentence, ...]
    opening_end: int  # first word index after the opening
    closing_start: int  # first word index of the closing
    punctuation_reliable: bool

    @property
    def n_words(self) -> int:
        return len(self.transcript.words)

    def in_opening(self, s: Sentence) -> bool:
        return s.first_word < self.opening_end

    def in_closing(self, s: Sentence) -> bool:
        return s.last_word >= self.closing_start

    def in_body(self, s: Sentence) -> bool:
        return not self.in_opening(s) and not self.in_closing(s)

    def span(self, s: Sentence) -> TranscriptSpan:
        return self.transcript.span(s.first_word, s.last_word + 1)

    def region(self, name: str) -> TranscriptSpan:
        """Evidence for a count of zero: the part of the transcript that was searched."""
        n = self.n_words
        first, stop = {
            "opening": (0, self.opening_end),
            "closing": (self.closing_start, n),
            "body": (self.opening_end, self.closing_start),
        }.get(name, (0, n))
        if stop <= first:
            first, stop = 0, n
        return self.transcript.span(first, stop)


def _ends_sentence(word: str, marks: frozenset[str]) -> bool:
    stripped = word.rstrip(_CLOSERS)
    return bool(stripped) and stripped[-1] in marks


def build(
    transcript: Transcript,
    parser: Parser,
    pauses: PauseSettings,
    *,
    opening_pct: float,
    closing_pct: float,
) -> Document:
    words = transcript.words
    marks = frozenset(pauses.sentence_marks)
    bounds: list[tuple[int, int]] = []
    start = 0
    for i, w in enumerate(words):
        if _ends_sentence(w.text, marks):
            bounds.append((start, i))
            start = i + 1
    if start < len(words):
        bounds.append((start, len(words) - 1))
    texts = [transcript.text[words[a].start_char : words[b].end_char] for a, b in bounds]
    parsed = parser(texts) if texts else []
    sentences = tuple(
        Sentence(i, a, b, words[a].start_char, words[b].end_char, t, tuple(toks))
        for i, ((a, b), t, toks) in enumerate(zip(bounds, texts, parsed, strict=True))
    )
    n = len(words)
    opening = max(1, round(n * opening_pct / 100)) if n else 0
    closing = max(1, round(n * closing_pct / 100)) if n else 0
    return Document(
        transcript=transcript,
        sentences=sentences,
        opening_end=opening,
        closing_start=n - closing,
        punctuation_reliable=punctuation_reliable(transcript, pauses),
    )
