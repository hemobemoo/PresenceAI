"""Knowledge-base search and citations (P2-d, ADR 0020).

Cosine similarity over the e5 vectors with NumPy (a few thousand chunks need no vector store).
Every hit carries a citation with book, chapter, section, licence and the book's URL, so a
knowledge-base answer always cites its source (FINAL_07 §F).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from presenceai.knowledge.embed import Embedder
from presenceai.schemas.knowledge import Book, Chunk, Hit


def citation(chunk: Chunk, books: Sequence[Book]) -> str:
    book = next(b for b in books if b.id == chunk.book)
    where = chunk.chapter if chunk.section is None else f"{chunk.chapter}, {chunk.section}"
    return f"{book.title}, {where}. {book.publisher}. {book.licence}. {book.url}"


class KnowledgeBase:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        books: Sequence[Book],
        embedder: Embedder,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk")
        self.chunks, self.vectors, self.books, self.embedder = (
            list(chunks),
            vectors,
            books,
            embedder,
        )

    def search(self, question: str, k: int) -> list[Hit]:
        if not self.chunks:
            return []
        q = self.embedder.embed([question], "query")[0]
        scores = self.vectors @ q
        best = np.argsort(-scores)[:k]
        return [
            Hit(chunk=self.chunks[i], score=float(scores[i]),
                citation=citation(self.chunks[i], self.books))
            for i in best
        ]  # fmt: skip
