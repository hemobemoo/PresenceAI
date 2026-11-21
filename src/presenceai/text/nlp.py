"""A small parsing interface for the text detectors (P2-b, ADR 0018).

The detectors need, per sentence: tokens with lemma, part of speech, stop-word flag, number flag,
entity type and whether the token heads a noun phrase. The app parses with spaCy's
`en_core_web_sm` 3.8.0, loaded from the folder that `scripts/download_models.py --only spacy_en`
unpacks (AGENTS.md rule 2: no other code downloads models). Tests use a fake parser
(`tests/fixtures/fake_nlp.py`), so they run without the model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from presenceai.speech.asr import MODELS_ROOT, ModelNotAvailable

SPACY_DIR = MODELS_ROOT / "en_core_web_sm-3.8.0"


@dataclass(frozen=True)
class Tok:
    text: str
    lemma: str
    pos: str  # Universal POS tag: NOUN, PROPN, VERB, ADJ, NUM, PRON, PUNCT, ...
    is_stop: bool
    like_num: bool
    ent: str  # entity type ("PERCENT", "CARDINAL", "DATE", ...) or ""
    chunk_head: bool  # heads a noun phrase
    start: int  # character offsets into the parsed text
    end: int


Parser = Callable[[Sequence[str]], list[list[Tok]]]


def _from_spacy(doc: Any) -> list[Tok]:
    heads = {c.root.i for c in doc.noun_chunks}
    return [
        Tok(
            text=t.text,
            lemma=t.lemma_,
            pos=t.pos_,
            is_stop=bool(t.is_stop),
            like_num=bool(t.like_num),
            ent=t.ent_type_,
            chunk_head=t.i in heads,
            start=t.idx,
            end=t.idx + len(t.text),
        )
        for t in doc
    ]


def spacy_parser(path: Path = SPACY_DIR) -> Parser:
    """The app's parser: spaCy from the local model folder."""
    if not (path / "config.cfg").is_file():
        raise ModelNotAvailable(
            f"{path.name} is not downloaded; run: "
            "uv run python scripts/download_models.py --only spacy_en"
        )
    import spacy  # heavy import, only when the text analysis runs

    nlp = spacy.load(path)

    def parse(texts: Sequence[str]) -> list[list[Tok]]:
        return [_from_spacy(doc) for doc in nlp.pipe(texts)]

    return parse
