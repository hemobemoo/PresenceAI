"""Build and load the knowledge base (P2-d, ADR 0020; FINAL_07 §F).

- Each book is checked before it is read: its file's SHA-256 must match configs/knowledge.yaml,
  and its own licence statement (dc:rights) must contain the recorded licence. Any difference
  stops the build.
- Passages are packed into chunks of at most `chunk_max_tokens` e5 tokens, never across a
  section; a longer paragraph is split at sentence ends (then at words).
- Every chunk keeps book, chapter, section and licence; `t2_allowed` is False for NC-SA books,
  and `t2_training_chunks` never returns them.
- Output (git-ignored, e.g. data/texts/kb/): chunks.jsonl, embeddings.npy and meta.json.
  JSON lines and NumPy instead of the parquet file FINAL_07 sketches: no new dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from presenceai.evalconfig import REPO_ROOT
from presenceai.knowledge.embed import Embedder
from presenceai.knowledge.epub import Passage, read_passages, rights
from presenceai.schemas.knowledge import Book, Chunk, KnowledgeSettings

CONFIG = REPO_ROOT / "configs" / "knowledge.yaml"
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class BookCheckError(ValueError):
    """A book's file or licence statement differs from what configs/knowledge.yaml records."""


def load_settings(path: Path = CONFIG) -> KnowledgeSettings:
    return KnowledgeSettings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def check_book(book: Book, raw_dir: Path) -> Path:
    path = raw_dir / book.file
    if not path.is_file():
        raise BookCheckError(f"{book.id}: {book.file} not found in {raw_dir}")
    if _sha256(path) != book.sha256:
        raise BookCheckError(f"{book.id}: SHA-256 differs from configs/knowledge.yaml")
    if book.licence_text.casefold() not in rights(path).casefold():
        raise BookCheckError(f"{book.id}: the book's licence statement is not {book.licence}")
    return path


def _pieces(text: str, embedder: Embedder, limit: int) -> list[str]:
    """`text` split into pieces of at most `limit` tokens: at sentence ends, then at words."""
    if embedder.count_tokens(text) <= limit:
        return [text]
    out: list[str] = []
    for sentence in _SENTENCE.split(text):
        if embedder.count_tokens(sentence) <= limit:
            out.append(sentence)
            continue
        words, cur = sentence.split(), []
        for w in words:
            if cur and embedder.count_tokens(" ".join([*cur, w])) > limit:
                out.append(" ".join(cur))
                cur = []
            cur.append(w)
        if cur:
            out.append(" ".join(cur))
    return out


@dataclass(frozen=True)
class _Group:
    chapter: str
    section: str | None
    texts: tuple[str, ...]


def _groups(passages: Iterable[Passage]) -> list[_Group]:
    groups: list[_Group] = []
    for p in passages:
        if groups and (groups[-1].chapter, groups[-1].section) == (p.chapter, p.section):
            groups[-1] = _Group(p.chapter, p.section, (*groups[-1].texts, p.text))
        else:
            groups.append(_Group(p.chapter, p.section, (p.text,)))
    return groups


def chunk_book(
    book: Book, passages: Sequence[Passage], embedder: Embedder, max_tokens: int
) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(g: _Group, text: str) -> None:
        chunks.append(
            Chunk(
                id=f"{book.id}/{len(chunks):05d}", book=book.id, chapter=g.chapter,
                section=g.section, licence=book.licence, t2_allowed=book.t2_allowed, text=text,
            )
        )  # fmt: skip

    for g in _groups(passages):
        cur: list[str] = []
        for paragraph in g.texts:
            for piece in _pieces(paragraph, embedder, max_tokens):
                if cur and embedder.count_tokens(" ".join([*cur, piece])) > max_tokens:
                    emit(g, " ".join(cur))
                    cur = []
                cur.append(piece)
        if cur:
            emit(g, " ".join(cur))
    return chunks


def build(
    raw_dir: Path, out_dir: Path, embedder: Embedder, settings: KnowledgeSettings
) -> dict[str, int]:
    """Check, read, chunk and embed every book; write the knowledge base. Returns chunk counts."""
    chunks: list[Chunk] = []
    counts: dict[str, int] = {}
    for book in settings.books:
        path = check_book(book, raw_dir)
        book_chunks = chunk_book(
            book, read_passages(path, settings.spine_prefixes), embedder, settings.chunk_max_tokens
        )
        counts[book.id] = len(book_chunks)
        chunks += book_chunks
    vectors = embedder.embed([c.text for c in chunks], "passage")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")
    np.save(out_dir / "embeddings.npy", vectors)
    meta = {"embedder": embedder.name, "books": {b.id: b.sha256 for b in settings.books},
            "chunk_max_tokens": settings.chunk_max_tokens, "chunks": counts}  # fmt: skip
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return counts


def load(out_dir: Path) -> tuple[list[Chunk], np.ndarray]:
    lines = (out_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
    vectors = np.load(out_dir / "embeddings.npy")
    if len(chunks) != len(vectors):
        raise ValueError("chunks.jsonl and embeddings.npy differ in length")
    return chunks, vectors


def t2_training_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Chunks that may become T2 training text: never NC-SA (FINAL_07 §F)."""
    return [c for c in chunks if c.t2_allowed]
