"""A fake parser for the text detectors: same `Tok` interface as text/nlp.py, no model.

Rules (enough for hand-written fixture transcripts, not a real tagger):
- tokens: words (with apostrophes) and single punctuation marks;
- lemma: lower case, a plural "s" removed (words longer than 3 letters, not ending in "ss");
- POS: PUNCT, NUM (digits or number words), PRON/DET/ADP/AUX/CCONJ/SCONJ for stop words,
  PROPN for a capitalised word that is not sentence-initial or for an all-capital acronym,
  ADV for "-ly", VERB for a few listed verbs, NOUN otherwise;
- entities: a number followed by "%" or "percent" is PERCENT; 1900-2099 is DATE; other digits
  are CARDINAL;
- noun-phrase head: the last NOUN/PROPN of a run of NOUN/PROPN tokens.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from presenceai.text.nlp import Tok

_TOKEN = re.compile(r"\w+(?:'\w+)?|[^\w\s]")
STOP = {
    "a", "an", "the", "i", "we", "you", "he", "she", "it", "they", "me", "us", "our", "my",
    "your", "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "will",
    "would", "can", "could", "do", "does", "did", "have", "has", "had", "to", "of", "in", "on",
    "at", "for", "with", "by", "from", "and", "or", "but", "so", "if", "then", "than", "as",
    "about", "into", "not", "no", "today", "what", "which", "who", "how", "why", "when", "there",
    "here", "all", "some", "most", "many", "more", "very", "just", "also", "let", "let's", "i'm",
}  # fmt: skip
NUMBER_WORDS = {"one", "two", "three", "four", "five", "ten", "hundred", "thousand", "million"}
VERBS = {
    "show", "think", "improve", "explain", "said", "say", "talk", "cover", "imagine", "use",
    "speak", "measure", "help", "means", "start", "conclude", "remember", "improves", "builds",
    "helped", "matters", "rehearse", "grow", "split", "met", "had", "agreed",
}  # fmt: skip
POS_OF_STOP = {"and": "CCONJ", "or": "CCONJ", "but": "CCONJ", "if": "SCONJ", "that": "SCONJ"}


def _lemma(word: str) -> str:
    w = word.lower()
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and w not in STOP:
        return w[:-1]
    return w


def _pos(word: str, first: bool) -> str:
    w = word.lower()
    if not re.match(r"\w", word):
        return "PUNCT"
    if word.isdigit() or w in NUMBER_WORDS:
        return "NUM"
    if w in STOP:
        return POS_OF_STOP.get(w, "PRON" if w in {"i", "we", "you", "it", "they"} else "DET")
    if len(word) >= 2 and word.isupper() and word.isalpha():
        return "PROPN"
    if word[0].isupper() and not first:
        return "PROPN"
    if w.endswith("ly"):
        return "ADV"
    if w in VERBS:
        return "VERB"
    return "NOUN"


def _parse(text: str) -> list[Tok]:
    raw = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text)]
    pos = [_pos(w, i == 0) for i, (w, _, _) in enumerate(raw)]
    toks: list[Tok] = []
    for i, (w, a, b) in enumerate(raw):
        nxt = raw[i + 1][0].lower() if i + 1 < len(raw) else ""
        ent = ""
        if w.isdigit():
            if nxt in {"%", "percent"}:
                ent = "PERCENT"
            elif len(w) == 4 and 1900 <= int(w) <= 2099:
                ent = "DATE"
            else:
                ent = "CARDINAL"
        nominal = pos[i] in {"NOUN", "PROPN"}
        head = nominal and not (i + 1 < len(pos) and pos[i + 1] in {"NOUN", "PROPN"})
        toks.append(
            Tok(
                text=w,
                lemma=_lemma(w),
                pos=pos[i],
                is_stop=w.lower() in STOP,
                like_num=pos[i] == "NUM",
                ent=ent,
                chunk_head=head,
                start=a,
                end=b,
            )
        )
    return toks


def fake_parser(texts: Sequence[str]) -> list[list[Tok]]:
    return [_parse(t) for t in texts]
