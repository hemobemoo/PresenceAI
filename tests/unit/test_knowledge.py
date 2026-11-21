"""Knowledge base (P2-d, ADR 0020): EPUB reading, licence checks, chunking, NC-SA exclusion,
retrieval with citations. A small EPUB is built in the test; a fake embedder replaces e5.
The slow test runs the real e5-small model and reports its RAM use."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from presenceai.evalconfig import REPO_ROOT
from presenceai.knowledge import build, epub, retrieve
from presenceai.schemas.knowledge import Book, KnowledgeSettings

OPF = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:rights>Test Book is licensed under a {rights}, except where otherwise noted.</dc:rights>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="front-matter-001" href="front.xhtml" media-type="application/xhtml+xml"/>
    <item id="chapter-001" href="c1.xhtml" media-type="application/xhtml+xml"/>
    <item id="chapter-002" href="c2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="front-matter-001"/><itemref idref="chapter-001"/>
    <itemref idref="chapter-002"/></spine>
</package>"""
NAV = """<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<body><nav epub:type="toc"><ol>
  <li><a href="front.xhtml">Introduction</a></li>
  <li><a href="c1.xhtml">Organizing Your Speech</a></li>
  <li><a href="c2.xhtml#top">Using Evidence</a></li>
</ol></nav></body></html>"""
C1 = """<html><body><h1><span>1</span></h1>
<p>Every speech needs a clear structure with an introduction.</p>
<h2>Transitions</h2><p>Transitions such as first and next guide the audience.</p>
<p>Signposts tell listeners where they are.<sup>1</sup></p>
<script>alert("never read")</script></body></html>"""
C2 = """<html><body><h2>Statistics</h2><p>Statistics make a claim concrete; cite the source.</p>
<ul><li>Use recent numbers.</li><li>Round sensibly.</li></ul></body></html>"""
FRONT = "<html><body><p>Front matter is skipped.</p></body></html>"
CC_BY = "Creative Commons Attribution 4.0 International License"
CC_NC_SA = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"


def make_epub(path: Path, rights: str = CC_BY) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:'
            'container" version="1.0"><rootfiles><rootfile full-path="EPUB/book.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        z.writestr("EPUB/book.opf", OPF.format(rights=rights))
        z.writestr("EPUB/nav.xhtml", NAV)
        z.writestr("EPUB/c1.xhtml", C1)
        z.writestr("EPUB/c2.xhtml", C2)
        z.writestr("EPUB/front.xhtml", FRONT)
    return path


def book(path: Path, *, licence: str = "CC BY 4.0", text: str = CC_BY, t2: bool = True,
         book_id: str = "test") -> Book:  # fmt: skip
    return Book(
        id=book_id, file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        title="Test Book", publisher="Tester", licence=licence, licence_text=text,
        url="https://example.org/book", t2_allowed=t2,
    )  # fmt: skip


class FakeEmbedder:
    """Bag-of-words vectors over a tiny vocabulary; tokens = words + 2."""

    name = "fake"
    VOCAB = ("structure", "introduction", "transitions", "statistics", "numbers", "source")

    def count_tokens(self, text: str) -> int:
        return len(text.split()) + 2

    def embed(self, texts: Sequence[str], kind: str) -> np.ndarray:
        rows = [[t.lower().count(w) + 1e-3 for w in self.VOCAB] for t in texts]
        v = np.array(rows, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def settings(*books: Book, max_tokens: int = 400) -> KnowledgeSettings:
    return KnowledgeSettings(chunk_max_tokens=max_tokens, top_k=3, spine_prefixes=("chapter",),
                             books=books)  # fmt: skip


def test_epub_chapters_sections_and_skips(tmp_path: Path):
    passages = epub.read_passages(make_epub(tmp_path / "b.epub"), ("chapter",))
    assert [(p.chapter, p.section) for p in passages] == [
        ("Organizing Your Speech", None),
        ("Organizing Your Speech", "Transitions"),
        ("Organizing Your Speech", "Transitions"),
        ("Using Evidence", "Statistics"),
        ("Using Evidence", "Statistics"),
        ("Using Evidence", "Statistics"),
    ]
    texts = " ".join(p.text for p in passages)
    assert "Front matter" not in texts and "never read" not in texts  # front matter, script
    assert "listeners where they are." in texts and "are.1" not in texts  # footnote mark dropped


def test_book_checks_stop_on_hash_or_licence(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = make_epub(raw / "b.epub")
    assert build.check_book(book(path), raw) == path
    wrong = book(path).model_copy(update={"sha256": "0" * 64})
    with pytest.raises(build.BookCheckError, match="SHA-256"):
        build.check_book(wrong, raw)
    nc = book(path, licence="CC BY-NC-SA 4.0", text=CC_NC_SA)
    with pytest.raises(build.BookCheckError, match="licence"):
        build.check_book(nc, raw)  # the book says CC BY, we recorded CC BY-NC-SA


def test_chunks_keep_metadata_and_stay_within_the_limit(tmp_path: Path):
    path = make_epub(tmp_path / "b.epub")
    passages = epub.read_passages(path, ("chapter",))
    emb = FakeEmbedder()
    chunks = build.chunk_book(book(path), passages, emb, max_tokens=12)
    assert all(emb.count_tokens(c.text) <= 12 for c in chunks)
    assert {(c.chapter, c.section) for c in chunks} == {
        ("Organizing Your Speech", None), ("Organizing Your Speech", "Transitions"),
        ("Using Evidence", "Statistics"),
    }  # fmt: skip
    assert all(c.licence == "CC BY 4.0" and c.t2_allowed for c in chunks)
    big = build.chunk_book(book(path), passages, emb, max_tokens=400)
    assert len(big) == 3  # one chunk per section: never across a section


def test_nc_sa_chunks_are_never_t2_training_text(tmp_path: Path):
    a = make_epub(tmp_path / "a.epub")
    b = make_epub(tmp_path / "b.epub", rights=CC_NC_SA)
    passages = epub.read_passages(a, ("chapter",))
    chunks = build.chunk_book(book(a), passages, FakeEmbedder(), 400) + build.chunk_book(
        book(b, licence="CC BY-NC-SA 4.0", text=CC_NC_SA, t2=False, book_id="nc"),
        epub.read_passages(b, ("chapter",)), FakeEmbedder(), 400,
    )  # fmt: skip
    kept = build.t2_training_chunks(chunks)
    assert kept and all(c.book == "test" and c.licence == "CC BY 4.0" for c in kept)


def test_build_load_and_search_with_citations(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    b = book(make_epub(raw / "b.epub"))
    out = tmp_path / "kb"
    counts = build.build(raw, out, FakeEmbedder(), settings(b))
    assert counts == {"test": 3}
    chunks, vectors = build.load(out)
    kb = retrieve.KnowledgeBase(chunks, vectors, (b,), FakeEmbedder())
    (hit, *_) = kb.search("how do I use statistics and numbers?", k=2)
    assert hit.chunk.section == "Statistics"
    assert hit.citation == (
        "Test Book, Using Evidence, Statistics. Tester. CC BY 4.0. https://example.org/book"
    )


def test_the_configured_books():
    s = build.load_settings()
    assert {b.licence for b in s.books} == {"CC BY 4.0", "CC BY-NC-SA 4.0"}
    for b in s.books:
        assert b.t2_allowed == ("NC" not in b.licence), b.id  # NC-SA never T2 text


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**20


@pytest.mark.slow
def test_e5_small_embeds_and_its_ram_use(capsys: Any):
    """Owner, 2026-11-22: measure e5-small's RAM on the laptop (printed; recorded in ADR 0020)."""
    from presenceai.knowledge.embed import E5Embedder
    from presenceai.speech.asr import ModelNotAvailable

    before = _rss_mb()
    try:
        emb = E5Embedder()
    except ModelNotAvailable:
        pytest.skip("multilingual-e5-small not downloaded (download_models.py --only e5_small)")
    loaded = _rss_mb()
    passages = ["A clear central message helps the audience remember the talk."] * 64
    vectors = emb.embed(passages, "passage")
    after = _rss_mb()
    assert vectors.shape == (64, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    q = emb.embed(["How do I make my main point memorable?"], "query")[0]
    other = emb.embed(["Bring a bottle of water to the venue."], "passage")[0]
    assert float(vectors[0] @ q) > float(other @ q)
    with capsys.disabled():
        print(f"\ne5-small RSS: +{loaded - before:.0f} MB after loading, "
              f"+{after - before:.0f} MB after embedding 64 passages")  # fmt: skip
    kb = REPO_ROOT / "data" / "texts" / "kb"
    if (kb / "chunks.jsonl").is_file():  # the owner's built knowledge base, counts only
        chunks, vecs = build.load(kb)
        assert len(chunks) == len(vecs) > 0


def test_xml_entity_declarations_are_refused():
    bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>'
    with pytest.raises(epub.UnsafeXmlError):
        epub.parse_xml(bomb)
    assert epub.parse_xml(b"<x><y/></x>").tag == "x"
