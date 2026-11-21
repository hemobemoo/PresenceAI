"""Read an EPUB with the standard library only (P2-d, ADR 0020; owner: no ebooklib, AGPL).

`zipfile` opens the archive; `xml.etree.ElementTree` reads the package files (container, OPF,
NCX or the EPUB 3 nav) through `parse_xml`, which refuses any entity declaration (no entity
expansion; ElementTree never loads external entities); `html.parser` reads the chapter files.
The text is data: nothing in it is executed or followed.

A book is returned as passages: (chapter, section, paragraph text). The chapter is the
table-of-contents label of the spine file; the section is the latest h2-h4 heading before the
paragraph. Only spine files whose id starts with one of `spine_prefixes` are read (Pressbooks:
"chapter"), so covers, front and back matter are skipped.
"""

from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

_SPACE = re.compile(r"\s+")
_BLOCKS = frozenset({"p", "li", "blockquote", "dd", "td"})
_HEADINGS = frozenset({"h2", "h3", "h4"})
_SKIP = frozenset({"script", "style", "nav", "figcaption", "sup"})


@dataclass(frozen=True)
class Passage:
    chapter: str
    section: str | None
    text: str


class UnsafeXmlError(ValueError):
    """An EPUB package file declares entities, which a book never needs."""


def parse_xml(data: bytes) -> ET.Element:
    """Parse a package file from an EPUB, refusing entity declarations (no defusedxml needed)."""
    if b"<!ENTITY" in data.upper():
        raise UnsafeXmlError("EPUB package file declares XML entities")
    return ET.fromstring(data)  # noqa: S314 - entity declarations refused above


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _clean(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


class _Chapter(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[tuple[str | None, str]] = []
        self.section: str | None = None
        self._buf: list[str] = []
        self._kind: str | None = None  # "block" or "heading"
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip += 1
        elif tag in _HEADINGS | _BLOCKS and self._kind is None:
            self._kind = "heading" if tag in _HEADINGS else "block"
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in _HEADINGS | _BLOCKS and self._kind is not None:
            if (tag in _HEADINGS) != (self._kind == "heading"):
                return
            text = _clean("".join(self._buf))
            if text and self._kind == "heading":
                self.section = text
            elif text:
                self.out.append((self.section, text))
            self._kind = None

    def handle_data(self, data: str) -> None:
        if self._kind is not None and not self._skip:
            self._buf.append(data)


def _toc_labels(
    zf: zipfile.ZipFile, opf_dir: str, manifest: dict[str, tuple[str, str, str]]
) -> dict[str, str]:
    """{file path in the archive: first table-of-contents label}, from the NCX or the nav."""
    labels: dict[str, str] = {}
    for href, media, props in manifest.values():
        path = posixpath.normpath(posixpath.join(opf_dir, href))
        base = posixpath.dirname(path)
        if media == "application/x-dtbncx+xml":
            root = parse_xml(zf.read(path))
            for point in root.iter():
                if _local(point.tag) != "navPoint":
                    continue
                text = next((e.text for e in point.iter() if _local(e.tag) == "text"), None)
                src = next((e.get("src") for e in point if _local(e.tag) == "content"), None)
                if text and src:
                    target = posixpath.normpath(posixpath.join(base, src.split("#")[0]))
                    labels.setdefault(target, _clean(text))
        elif "nav" in props.split():
            root = parse_xml(zf.read(path))
            for a in root.iter():
                if _local(a.tag) == "a" and a.get("href"):
                    target = posixpath.normpath(
                        posixpath.join(base, a.get("href", "").split("#")[0])
                    )
                    labels.setdefault(target, _clean("".join(a.itertext())))
    return labels


def rights(path: Path) -> str:
    """The book's own licence statement (dc:rights in the OPF)."""
    with zipfile.ZipFile(path) as zf:
        opf = _opf_path(zf)
        root = parse_xml(zf.read(opf))
        return " ".join(
            _clean("".join(e.itertext())) for e in root.iter() if _local(e.tag) == "rights"
        )


def _opf_path(zf: zipfile.ZipFile) -> str:
    container = parse_xml(zf.read("META-INF/container.xml"))
    for e in container.iter():
        if _local(e.tag) == "rootfile" and e.get("full-path"):
            return str(e.get("full-path"))
    raise ValueError("EPUB without a package file")


def read_passages(path: Path, spine_prefixes: tuple[str, ...]) -> list[Passage]:
    with zipfile.ZipFile(path) as zf:
        opf = _opf_path(zf)
        opf_dir = posixpath.dirname(opf)
        root = parse_xml(zf.read(opf))
        manifest = {
            e.get("id", ""): (e.get("href", ""), e.get("media-type", ""), e.get("properties", ""))
            for e in root.iter()
            if _local(e.tag) == "item"
        }
        spine = [e.get("idref", "") for e in root.iter() if _local(e.tag) == "itemref"]
        labels = _toc_labels(zf, opf_dir, manifest)
        passages: list[Passage] = []
        for idref in spine:
            if not idref.startswith(spine_prefixes) or idref not in manifest:
                continue
            path_in_zip = posixpath.normpath(posixpath.join(opf_dir, manifest[idref][0]))
            parser = _Chapter()
            parser.feed(zf.read(path_in_zip).decode("utf-8", errors="replace"))
            parser.close()
            chapter = labels.get(path_in_zip) or idref
            passages += [Passage(chapter, section, text) for section, text in parser.out]
        return passages
