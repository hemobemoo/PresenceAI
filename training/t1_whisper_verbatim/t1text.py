"""T1 verbatim text rules (ADR 0027): the DisfluencySpeech text builder, filler and partial counts.

One text style for every T1 source, and for E3's DisfluencySpeech references (PR-3 uses the same
builder, owner 2026-11-27):

- fillers ("uh", "um") are kept;
- **partial words keep a trailing hyphen** ("th-"). DisfluencySpeech's `transcript_a`, used in S3
  (ADR 0005), writes them as the full word ("the"); T1 builds its text from
  `transcript_annotated` instead, so the partial stays;
- backchannels ("mm-hmm", "uh-huh") are kept as written, count as **zero** fillers and are never
  split at the hyphen into "mm" + "hmm";
- a partial filler ("um-") still counts as that filler.

Filler matching reuses the app's `filler_key` (lower case, edge punctuation stripped, letter runs
collapsed), so "Ummm," and "um-" both match "um". On Colab the build injects the app's
`normalize.py` next to this file.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

try:
    from presenceai.text.normalize import filler_key
except ImportError:  # on Colab: the build injects src/presenceai/text/normalize.py as normalize.py
    from normalize import filler_key  # pyright: ignore[reportMissingImports]

PARTIAL = "-"
_TAG = re.compile(r"<[^<>]*>")  # non-verbal tags such as <laughter>
_BRACE_OPEN = re.compile(r"\{[A-Z]\s")  # {F , {D , {C , {E , {A  (filler, discourse, ...)
_MARKUP_ONLY = re.compile(r"^[-/#+\[\]{}()]+$")
_SPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.?!;:])")


def build_ds_text(annotated: str) -> str:
    """DisfluencySpeech `transcript_annotated` -> T1 text: markup out, fillers and partials kept.

    `{F uh, }` -> `uh,`; `[ the, + the ]` -> `the, the`; `--`, lone `-`, `<tags>` removed;
    `th-` kept. Equals `transcript_a` except that partial words keep their hyphen.
    """
    text = _TAG.sub(" ", annotated)
    text = _BRACE_OPEN.sub(" ", text).replace("}", " ")
    kept = [t for t in _SPACE.split(text) if t and not _MARKUP_ONLY.match(t)]
    return _SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(kept)).strip()


def _key(token: str) -> str:
    return filler_key(token, "en")


def _keys(entries: Iterable[str]) -> frozenset[str]:
    return frozenset(_key(e) for e in entries)


def is_partial(token: str) -> bool:
    """A cut-off word: ends with the hyphen marker once other edge punctuation is stripped."""
    core = token.rstrip(",.?!;:\"'")
    return len(core) > 1 and core.endswith(PARTIAL) and not core.endswith(PARTIAL * 2)


def _is_filler(token: str, hes: frozenset[str], back: frozenset[str]) -> bool:
    """A hesitation. A partial counts only if its stem has two or more letters: "um-" is "um", but
    the cut-off word "m-" ("you m- mute") is not "mm", although letter-run collapsing gives both
    the key "m" (P4 data run d2, ADR 0027)."""
    k = _key(token)
    if not k or k in back or k not in hes:
        return False
    if is_partial(token):
        stem = token.rstrip(",.?!;:\"'").rstrip(PARTIAL)
        return sum(ch.isalpha() for ch in stem) >= 2
    return True


def count_fillers(text: str, hesitations: Iterable[str], backchannels: Iterable[str]) -> int:
    """Hesitations in a whitespace-tokenised text. Backchannels count zero; "um-" counts as "um"."""
    hes, back = _keys(hesitations), _keys(backchannels)
    return sum(1 for tok in text.split() if _is_filler(tok, hes, back))


def count_partials(text: str, hesitations: Iterable[str] = ()) -> int:
    """Partial words, not counting partial fillers ("um-" is a filler, not a cut-off word)."""
    hes = _keys(hesitations)
    return sum(1 for t in text.split() if is_partial(t) and not _is_filler(t, hes, frozenset()))
