"""AMI for T1 (ADR 0026 §3, ADR 0027): word files, spelling, segments, speakers, meeting choice.

Only the official Edinburgh files are used (DATA_SOURCES §4). This module is pure: it reads bytes
and returns data, and never downloads anything (the Colab data run does that).

Format assumptions (NXT, AMI manual annotations v1.6.2), confirmed by the data run's format probe
before any result is used:
- `words/<meeting>.<agent>.words.xml`: `<w starttime endtime [punc="true"] [trunc="true"]>`,
  plus `<vocalsound>`, `<nonvocalsound>`, `<gap>`, `<disfmarker>`, `<pause>`, `<comment>`
  (run d2's probe confirmed `trunc`, `punc` and these elements, plus `<transformerror>`);
- `corpusResources/meetings.xml`: `<meeting observation=...>` with `<speaker nxt_agent
  global_name channel>` children (channel = the `Headset-<n>` number).
Unknown elements are counted, never silently used.
"""

from __future__ import annotations

import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

PARTIAL = "-"
SCENARIO = re.compile(r"^(ES2|IS1|TS3)\d{3}[a-d]$")
# `transformerror`: found 3 times by run d2's probe; treated like `<gap>` (the transcript is not
# reliable there), so a segment holding one is dropped (ADR 0027, PROVISIONAL).
GAP_LIKE = {"gap", "transformerror"}
KNOWN = {"w", "vocalsound", "nonvocalsound", "disfmarker", "pause", "comment", *GAP_LIKE}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _float(v: str | None) -> float | None:
    if v is None or v.strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


@dataclass(frozen=True)
class Item:
    kind: str  # word | punc | gap | other (vocal sounds, markers: never in the text)
    text: str
    start: float | None
    end: float | None
    trunc: bool = False
    id: str = ""  # nite:id, used to resolve the disfluency layer's word links (E3)


NITE_ID = "{http://nite.sourceforge.net/}id"


@dataclass
class Probe:
    """What the word files actually contain (written to format_probe.json by the data run)."""

    elements: Counter[str] = field(default_factory=Counter)
    w_attributes: Counter[str] = field(default_factory=Counter)
    short_tokens: Counter[str] = field(default_factory=Counter)  # filler-like spellings
    untimed_words: int = 0


_FILLERISH = re.compile(r"^(u+h*m*|e+r+m*|a+h+|h*m+h*m*|m+-?h*m+|u+h+-?h*u+h*|m+)$")


def parse_words(xml: bytes, probe: Probe | None = None) -> list[Item]:
    root = ET.fromstring(xml)  # noqa: S314 (official AMI corpus file, checked by SHA-256)
    items: list[Item] = []
    for el in root:
        tag = _local(el.tag)
        start, end = _float(el.get("starttime")), _float(el.get("endtime"))
        if probe is not None:
            probe.elements[tag] += 1
        if tag == "w":
            text = (el.text or "").strip()
            if probe is not None:
                probe.w_attributes.update(_local(k) for k in el.attrib)
                if start is None or end is None:
                    probe.untimed_words += 1
                if _FILLERISH.match(text.casefold().strip(".,?!")):
                    probe.short_tokens[text.casefold()] += 1
            wid = el.get(NITE_ID, "")
            if el.get("punc") == "true":
                items.append(Item("punc", text, start, end, id=wid))
            elif text:
                trunc = el.get("trunc") == "true"
                items.append(Item("word", text, start, end, trunc=trunc, id=wid))
        elif tag in GAP_LIKE:
            items.append(Item("gap", "", start, end))
        else:
            items.append(Item("other", "", start, end))
    return items


def spell(word: str, trunc: bool, spelling: Mapping[str, str], backchannels: Iterable[str]) -> str:
    """AMI spelling -> T1 text spelling. Fillers mapped to the E3 lexicon; backchannels kept as
    written (never split at the hyphen); a cut-off word keeps a trailing hyphen."""
    low = word.casefold()
    if low in {b.casefold() for b in backchannels}:
        out = word
    elif low.rstrip(PARTIAL) in spelling:
        out = spelling[low.rstrip(PARTIAL)]
        if low.endswith(PARTIAL):
            out += PARTIAL
    else:
        out = word
    if trunc and not out.endswith(PARTIAL):
        out += PARTIAL
    return out


@dataclass(frozen=True)
class Segment:
    meeting: str
    agent: str
    channel: str
    start_s: float
    end_s: float
    text: str
    n_words: int
    crosstalk_share: float
    # one AMI word id per text token (E3 repetition scoring); never written to the T1 manifest,
    # so the pinned manifest hash (ADR 0027 §6) is unchanged
    word_ids: tuple[str, ...] = field(default=(), compare=False, repr=False)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def overlap_share(start: float, end: float, others: Sequence[tuple[float, float]]) -> float:
    if end <= start:
        return 0.0
    covered = sum(max(0.0, min(e, end) - max(s, start)) for s, e in others)
    return covered / (end - start)


def timed_words(items: Iterable[Item]) -> list[tuple[float, float]]:
    return [
        (i.start, i.end)
        for i in items
        if i.kind == "word" and i.start is not None and i.end is not None and i.end > i.start
    ]


def segments(
    meeting: str,
    agent: str,
    channel: str,
    items: Sequence[Item],
    others: Sequence[tuple[float, float]],
    cfg: Mapping[str, object],
    spelling: Mapping[str, str],
    backchannels: Sequence[str],
    drops: Counter[str] | None = None,
) -> list[Segment]:
    """Cut one speaker's items into segments at that speaker's pauses (ADR 0027)."""
    gap_s, max_s = float(cfg["split_gap_s"]), float(cfg["max_s"])  # type: ignore[arg-type]
    min_s, pad_s = float(cfg["min_s"]), float(cfg["pad_s"])  # type: ignore[arg-type]
    max_cross = float(cfg["crosstalk_max_share"])  # type: ignore[arg-type]
    drop_gap = bool(cfg["drop_if_gap_element"])
    others_u = _union(others)
    drops = drops if drops is not None else Counter()

    groups: list[list[Item]] = []
    cur: list[Item] = []
    first: float | None = None
    last_end: float | None = None
    for it in items:
        if it.kind == "word" and it.start is not None and it.end is not None:
            too_far = last_end is not None and it.start - last_end >= gap_s
            too_long = first is not None and it.end - first + 2 * pad_s > max_s
            if cur and (too_far or too_long):
                groups.append(cur)
                cur, first = [], None
            if first is None:
                first = it.start
            last_end = it.end
        cur.append(it)
    if cur:
        groups.append(cur)

    out: list[Segment] = []
    for g in groups:
        words = [i for i in g if i.kind == "word"]
        times = [(i.start, i.end) for i in words if i.start is not None and i.end is not None]
        if not times:
            drops["no_timed_word"] += 1
            continue
        start = max(0.0, min(s for s, _ in times) - pad_s)
        end = max(e for _, e in times) + pad_s
        if drop_gap and any(i.kind == "gap" for i in g):
            drops["gap_element"] += 1
            continue
        if end - start < min_s:
            drops["too_short"] += 1
            continue
        if end - start > max_s:
            drops["too_long"] += 1  # a single word longer than the window
            continue
        share = overlap_share(start, end, others_u)
        if share > max_cross:
            drops["crosstalk"] += 1
            continue
        parts: list[str] = []
        ids: list[str] = []
        for i in g:
            if i.kind == "word":
                parts.append(spell(i.text, i.trunc, spelling, backchannels))
                ids.append(i.id)
            elif i.kind == "punc" and parts:
                parts[-1] += i.text
        out.append(
            Segment(
                meeting,
                agent,
                channel,
                round(start, 3),
                round(end, 3),
                " ".join(parts),
                len(words),
                round(share, 4),
                tuple(ids),
            )
        )
    return out


@dataclass(frozen=True)
class Speaker:
    agent: str
    global_name: str
    channel: str


def parse_meetings(xml: bytes) -> dict[str, list[Speaker]]:
    root = ET.fromstring(xml)  # noqa: S314 (official AMI corpus file, checked by SHA-256)
    out: dict[str, list[Speaker]] = {}
    for m in root.iter():
        if _local(m.tag) != "meeting":
            continue
        obs = m.get("observation")
        if not obs:
            raise ValueError("a <meeting> has no observation attribute")
        spk: list[Speaker] = []
        for s in m:
            if _local(s.tag) != "speaker":
                continue
            agent, name, ch = s.get("nxt_agent"), s.get("global_name"), s.get("channel")
            if not (agent and name and ch is not None):
                raise ValueError(f"{obs}: a <speaker> lacks nxt_agent, global_name or channel")
            spk.append(Speaker(agent, name, ch))
        out[obs] = spk
    if not out:
        raise ValueError("meetings.xml has no <meeting> element")
    return out


def session(meeting: str) -> str:
    return meeting[:-1] if meeting[-1:] in "abcd" else meeting


def speakers_of(meetings: Mapping[str, Sequence[Speaker]], sessions: Iterable[str]) -> set[str]:
    want = set(sessions)
    return {s.global_name for m, spk in meetings.items() if session(m) in want for s in spk}


def speaker_check(
    meetings: Mapping[str, Sequence[Speaker]], dev: Sequence[str], test: Sequence[str]
) -> dict[str, object]:
    """ADR 0012: no speaker in both an E3 dev and an E3 test session (ids kept only if shared)."""
    d, t = speakers_of(meetings, dev), speakers_of(meetings, test)
    shared = sorted(d & t)
    found = {session(m) for m in meetings}
    missing = sorted((set(dev) | set(test)) - found)  # run d2 lacked these brackets (ADR 0027)
    return {
        "dev_sessions": list(dev),
        "test_sessions": list(test),
        "missing_sessions": missing,
        "dev_speakers": len(d),
        "test_speakers": len(t),
        "shared_speakers": shared,
        "passed": not shared and not missing,
    }


def training_pool(
    meetings: Mapping[str, Sequence[Speaker]],
    exclude_sessions: Sequence[str],
    exclude_shared_speakers: bool,
) -> tuple[list[str], dict[str, int]]:
    """Scenario meetings outside the E3 sessions, without any of their speakers."""
    banned = speakers_of(meetings, exclude_sessions) if exclude_shared_speakers else set()
    pool, why = [], Counter[str]()
    for m in sorted(meetings):
        if not SCENARIO.match(m):
            why["not_scenario"] += 1
        elif session(m) in set(exclude_sessions):
            why["e3_session"] += 1
        elif any(s.global_name in banned for s in meetings[m]):
            why["shares_e3_speaker"] += 1
        else:
            pool.append(m)
    return pool, dict(why)


def meeting_order(pool: Sequence[str], sites: Sequence[str], seed: int) -> list[str]:
    """Seeded shuffle within each site (sorted first), then round-robin over the sites."""
    rng = random.Random(seed)  # noqa: S311 (reproducible data order, not security)
    per = {}
    for site in sites:
        ms = sorted(m for m in pool if m.startswith(site))
        rng.shuffle(ms)
        per[site] = ms
    order: list[str] = []
    while any(per.values()):
        for site in sites:
            if per[site]:
                order.append(per[site].pop(0))
    return order


def select(
    order: Sequence[str], segs: Mapping[str, Sequence[Segment]], hours: float
) -> list[Segment]:
    """Whole meetings in order until the kept speech reaches `hours`; the last meeting is cut at
    the target, segment by segment (owner P1: hours are speech kept in the manifest)."""
    target = hours * 3600.0
    out: list[Segment] = []
    total = 0.0
    for m in order:
        for s in sorted(segs.get(m, ()), key=lambda x: (x.start_s, x.agent)):
            if total >= target:
                return out
            out.append(s)
            total += s.duration_s
    return out


# ---- E3: the AMI disfluency layer (ADR 0028) ---------------------------------------------------
#
# Run b3's probe (2026-11-28): `<dsfl>` elements (33,032) carry no type attribute; they hold
# `<nite:child href=...words.xml#id(A)..id(B)>` word links and `<nite:pointer role=... href=...>`
# links (69,773). The type is found by following the pointers into the AMI ontology files
# (`ontologies/*.xml`): a node's label is its own attribute values (except ids) plus those of its
# ancestors. A disfluency is a repetition if any label it points to contains "repet". Every role,
# link target and resolved label is counted in the probe, so the rule can be checked.

_HREF = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")
REPETITION = re.compile(r"repet", re.IGNORECASE)


@dataclass
class DisfluencyProbe:
    elements: Counter[str] = field(default_factory=Counter)
    attributes: Counter[str] = field(default_factory=Counter)
    type_values: Counter[str] = field(default_factory=Counter)
    pointer_roles: Counter[str] = field(default_factory=Counter)
    pointer_targets: Counter[str] = field(default_factory=Counter)  # file part of the href
    unresolved_links: int = 0
    unresolved_pointers: int = 0


@dataclass(frozen=True)
class Disfluency:
    type: str
    word_ids: tuple[str, ...]


def _href(el: ET.Element) -> str:
    return next((v for k, v in el.attrib.items() if _local(k) == "href"), "")


def load_ontology(xml: bytes) -> dict[str, str]:
    """id -> label for every node of one ontology file (ancestors' labels joined by '/')."""
    root = ET.fromstring(xml)  # noqa: S314 (official AMI corpus file, checked by SHA-256)
    out: dict[str, str] = {}

    def walk(el: ET.Element, trail: tuple[str, ...]) -> None:
        own = " ".join(v for k, v in el.attrib.items() if _local(k) not in {"id", "href"})
        label = (*trail, own) if own else trail
        nid = el.get(NITE_ID)
        if nid:
            out[nid] = "/".join(label) or _local(el.tag)
        for c in el:
            walk(c, label)

    walk(root, ())
    return out


def _type_of(el: ET.Element, ontology: Mapping[str, str], probe: DisfluencyProbe | None) -> str:
    for k, v in el.attrib.items():
        if _local(k) in {"type", "dsfl-type", "disfl-type", "kind"}:
            return v
    labels = []
    for c in el:
        if _local(c.tag) != "pointer":
            continue
        role, href = c.get("role", ""), _href(c)
        if probe is not None:
            probe.pointer_roles[role] += 1
            probe.pointer_targets[href.split("#", 1)[0]] += 1
        m = _HREF.search(href)
        if m and m.group(1) in ontology:
            labels.append(f"{role}={ontology[m.group(1)]}")
        elif probe is not None:
            probe.unresolved_pointers += 1
    return "; ".join(labels) if labels else _local(el.tag)


def parse_disfluency(
    xml: bytes,
    word_order: Sequence[str],
    probe: DisfluencyProbe | None = None,
    ontology: Mapping[str, str] | None = None,
) -> list[Disfluency]:
    """Every element with word links, its type (an attribute, or the ontology labels its pointers
    lead to) and the word ids it covers (ranges expanded in the words file's order)."""
    root = ET.fromstring(xml)  # noqa: S314 (official AMI corpus file, checked by SHA-256)
    onto = ontology or {}
    pos = {w: i for i, w in enumerate(word_order)}
    out: list[Disfluency] = []
    for el in root.iter():
        tag = _local(el.tag)
        if probe is not None:
            probe.elements[tag] += 1
            probe.attributes.update(_local(k) for k in el.attrib)
        links = [c for c in el if _local(c.tag) == "child"]
        if not links:
            continue
        ids: list[str] = []
        for c in links:
            m = _HREF.search(_href(c))
            if not m:
                continue
            a, b = m.group(1), m.group(2) or m.group(1)
            if a not in pos or b not in pos:
                if probe is not None:
                    probe.unresolved_links += 1
                continue
            ids.extend(word_order[pos[a] : pos[b] + 1])
        t = _type_of(el, onto, probe)
        if probe is not None:
            probe.type_values[t] += 1
        if ids:
            out.append(Disfluency(t, tuple(ids)))
    return out


def repetitions(dis: Iterable[Disfluency]) -> list[Disfluency]:
    return [d for d in dis if REPETITION.search(d.type)]
