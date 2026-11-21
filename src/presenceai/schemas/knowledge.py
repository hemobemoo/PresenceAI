"""Knowledge-base types (P2-d, ADR 0020)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Book(_Frozen):
    id: str = Field(min_length=1)
    file: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    licence: str = Field(min_length=1)  # e.g. "CC BY 4.0"
    licence_text: str = Field(min_length=1)  # as the book's own metadata states it
    url: str = Field(min_length=1)
    t2_allowed: bool  # False for NC-SA books: never T2 training text (FINAL_07 §F)


class KnowledgeSettings(_Frozen):
    chunk_max_tokens: int = Field(gt=16)
    top_k: int = Field(gt=0)
    spine_prefixes: tuple[str, ...] = Field(min_length=1)
    books: tuple[Book, ...] = Field(min_length=1)


class Chunk(_Frozen):
    id: str = Field(min_length=1)  # <book>/<n>
    book: str = Field(min_length=1)  # Book.id
    chapter: str = Field(min_length=1)
    section: str | None = None
    licence: str = Field(min_length=1)
    t2_allowed: bool
    text: str = Field(min_length=1)


class Hit(_Frozen):
    chunk: Chunk
    score: float  # cosine similarity
    citation: str = Field(min_length=1)
