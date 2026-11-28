"""Read single members of a split zip (Info-ZIP `zip -s`: .z01 … .zNN, then .zip) without
downloading the whole archive (owner E1, ADR 0028). Standard library only.

- `Parts` joins the parts into one logical byte stream; a `fetch(part, start, length)` callback
  reads from HTTP range requests (`http_fetch`) or from local files (`file_fetch`, the fallback
  after a full download).
- `index(parts)` reads the end-of-central-directory record (zip64 included) and the central
  directory, which sit in the last part, and returns every member's method, CRC32, sizes, disk
  and offset. Offsets in a split archive are relative to the start of their disk (part).
- `extract(parts, entry, dest)` streams one member (stored or deflated), checks its CRC32 against
  the index and returns its SHA-256; a mismatch deletes the file and raises.
"""

from __future__ import annotations

import hashlib
import struct
import urllib.request
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

EOCD = 0x06054B50
EOCD64 = 0x06064B50
EOCD64_LOC = 0x07064B50
CDH = 0x02014B50
LFH = 0x04034B50
CHUNK = 8 << 20
USER_AGENT = "PresenceAI-research-fetch/1.0"

Fetch = Callable[[int, int, int], bytes]  # (part index, start, length) -> bytes


class RangeNotSupportedError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    name: str
    method: int
    crc32: int
    compressed: int
    size: int
    disk: int
    offset: int


class Parts:
    def __init__(self, sizes: Sequence[int], fetch: Fetch) -> None:
        self.sizes = list(sizes)
        self.fetch = fetch
        self.starts = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.total = sum(self.sizes)

    def read_logical(self, pos: int, n: int) -> bytes:
        if pos < 0 or pos + n > self.total:
            raise ValueError(f"read {pos}+{n} outside the archive ({self.total} bytes)")
        out = bytearray()
        while n > 0:
            part = max(i for i, s in enumerate(self.starts) if s <= pos)
            within = pos - self.starts[part]
            take = min(n, self.sizes[part] - within)
            data = self.fetch(part, within, take)
            if len(data) != take:
                raise OSError(f"short read from part {part}: {len(data)} of {take} bytes")
            out += data
            pos += take
            n -= take
        return bytes(out)

    def read(self, disk: int, offset: int, n: int) -> bytes:
        return self.read_logical(self.starts[disk] + offset, n)


def http_fetch(urls: Sequence[str]) -> Fetch:
    def fetch(part: int, start: int, length: int) -> bytes:
        req = urllib.request.Request(  # noqa: S310 (https URLs from the config)
            urls[part],
            headers={"Range": f"bytes={start}-{start + length - 1}", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
            if r.status != 206:
                raise RangeNotSupportedError(f"{urls[part]}: HTTP {r.status} to a range request")
            return r.read()

    return fetch


def file_fetch(paths: Sequence[Path]) -> Fetch:
    def fetch(part: int, start: int, length: int) -> bytes:
        with paths[part].open("rb") as f:
            f.seek(start)
            return f.read(length)

    return fetch


def _zip64_extra(extra: bytes, need: list[str]) -> dict[str, int]:
    """Values from the zip64 extra field (id 0x0001), in the order the spec gives."""
    i = 0
    while i + 4 <= len(extra):
        hid, size = struct.unpack_from("<HH", extra, i)
        body = extra[i + 4 : i + 4 + size]
        if hid == 1:
            out, j = {}, 0
            for key in ("size", "compressed", "offset", "disk"):
                if key in need:
                    fmt = "<I" if key == "disk" else "<Q"
                    out[key] = struct.unpack_from(fmt, body, j)[0]
                    j += 4 if key == "disk" else 8
            return out
        i += 4 + size
    if need:
        raise ValueError(f"zip64 extra field missing for {need}")
    return {}


def index(parts: Parts, tail: int = 1 << 16) -> dict[str, Entry]:
    last = len(parts.sizes) - 1
    n = min(tail + 22, parts.sizes[last])
    buf = parts.read(last, parts.sizes[last] - n, n)
    at = buf.rfind(struct.pack("<I", EOCD))
    if at < 0:
        raise ValueError("no end-of-central-directory record in the last part")
    (_, _, cd_disk, _, total, cd_size, cd_off, _) = struct.unpack_from("<IHHHHIIH", buf, at)
    if 0xFFFF in (cd_disk, total) or 0xFFFFFFFF in (cd_size, cd_off):
        loc = at - 20
        sig, eocd64_disk, eocd64_off, _ = struct.unpack_from("<IIQI", buf, loc)
        if sig != EOCD64_LOC:
            raise ValueError("zip64 locator missing")
        rec = parts.read(eocd64_disk, eocd64_off, 56)
        if struct.unpack_from("<I", rec, 0)[0] != EOCD64:
            raise ValueError("zip64 end-of-central-directory record missing")
        (_, _, _, _, _, cd_disk, _, total, cd_size, cd_off) = struct.unpack_from(
            "<IQHHIIQQQQ", rec, 0
        )
    cd = parts.read(cd_disk, cd_off, cd_size)
    out: dict[str, Entry] = {}
    i = 0
    for _ in range(total):
        if struct.unpack_from("<I", cd, i)[0] != CDH:
            raise ValueError(f"bad central directory entry at {i}")
        # after signature, versions and flags: method time date crc csize usize nlen elen clen
        # disk internal-attr external-attr offset
        (method, _, _, crc, csize, usize, nlen, elen, clen, disk, _, _, off) = struct.unpack_from(
            "<HHHIIIHHHHHII", cd, i + 10
        )
        name = cd[i + 46 : i + 46 + nlen].decode("utf-8", errors="replace")
        extra = cd[i + 46 + nlen : i + 46 + nlen + elen]
        need = [
            k
            for k, v in (("size", usize), ("compressed", csize), ("offset", off))
            if v == 0xFFFFFFFF
        ]
        if disk == 0xFFFF:
            need.append("disk")
        z = _zip64_extra(extra, need)
        out[name] = Entry(
            name,
            method,
            crc,
            z.get("compressed", csize),
            z.get("size", usize),
            z.get("disk", disk),
            z.get("offset", off),
        )
        i += 46 + nlen + elen + clen

    return out


def extract(parts: Parts, e: Entry, dest: Path) -> dict[str, object]:
    """Stream one member to dest; check CRC32; return {bytes, sha256, crc32}."""
    head = parts.read(e.disk, e.offset, 30)
    if struct.unpack_from("<I", head, 0)[0] != LFH:
        raise ValueError(f"{e.name}: no local file header at its offset")
    nlen, elen = struct.unpack_from("<HH", head, 26)
    pos = parts.starts[e.disk] + e.offset + 30 + nlen + elen
    if e.method not in (0, 8):
        raise ValueError(f"{e.name}: compression method {e.method} not supported")
    dec = zlib.decompressobj(-15) if e.method == 8 else None
    crc, sha, n = 0, hashlib.sha256(), 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    left = e.compressed
    with dest.open("wb") as f:
        while left > 0:
            take = min(CHUNK, left)
            raw = parts.read_logical(pos, take)
            pos += take
            left -= take
            data = dec.decompress(raw) if dec else raw
            f.write(data)
            crc = zlib.crc32(data, crc)
            sha.update(data)
            n += len(data)
        if dec:
            tail = dec.flush()
            f.write(tail)
            crc = zlib.crc32(tail, crc)
            sha.update(tail)
            n += len(tail)
    if crc != e.crc32 or n != e.size:
        dest.unlink(missing_ok=True)
        raise ValueError(
            f"{e.name}: CRC32/size mismatch (got {crc:08x}/{n}, index {e.crc32:08x}/{e.size})"
        )
    return {"bytes": n, "sha256": sha.hexdigest(), "crc32": f"{crc:08x}"}
