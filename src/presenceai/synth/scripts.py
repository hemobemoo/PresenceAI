"""C1 scripts: parsing into phrases and the automated checks (ADR 0014 §2).

A script is plain English with three kinds of phrase ends: a sentence mark (`. ! ?`), a comma,
and `|`, which marks a mid-sentence split point (removed from the spoken text). Each phrase is
one TTS call; the pause inserted after it gets the matching true position.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import yaml

from presenceai.evalconfig import C1Settings
from presenceai.schemas.c1 import WORD, Boundary, Phrase, Script

ALLOWED = re.compile(r"^[A-Za-z ,.?!|]+$")
SENTENCE_MARKS = ".?!"
# Words whose pronunciation depends on meaning or part of speech; G2P can pick the wrong one.
HETERONYMS = frozenset(
    {
        "read", "live", "lives", "lead", "leads", "tear", "tears", "wind", "close", "record",
        "records", "present", "presents", "object", "objects", "content", "minute", "desert",
        "produce", "project", "projects", "refuse", "row", "bow", "bass", "dove", "wound",
        "subject", "permit", "conduct", "contract", "increase", "decrease", "export", "import",
        "progress", "rebel", "invalid", "perfect", "separate", "estimate", "alternate",
        "moderate", "appropriate", "associate", "attribute", "graduate", "polish",
        "used", "use", "uses", "excuse", "house", "does", "sow", "entrance", "console",
        "number", "resume", "advocate", "deliberate", "duplicate", "elaborate", "intimate",
        "suspect", "insult", "convert", "address", "combine", "compound", "conflict",
        "contest", "digest", "survey", "upset", "second",
    }
)  # fmt: skip


def parse_script(script_id: str, voice: str, title: str, text: str) -> Script:
    phrases: list[Phrase] = []
    buf = ""
    for ch in " ".join(text.split()):
        if ch == "|":
            if not buf.strip():
                raise ValueError(f"{script_id}: '|' must split a sentence between two words")
            phrases.append(Phrase(text=buf.strip(), boundary_after="mid_sentence"))
            buf = ""
            continue
        buf += ch
        kind: Boundary | None = (
            "sentence_boundary"
            if ch in SENTENCE_MARKS
            else "clause_boundary"
            if ch == ","
            else None
        )
        if kind is not None:
            phrases.append(Phrase(text=buf.strip(), boundary_after=kind))
            buf = ""
    if buf.strip():
        phrases.append(Phrase(text=buf.strip(), boundary_after="end"))
    if not phrases:
        raise ValueError(f"{script_id}: empty script")
    last = phrases[-1]
    phrases[-1] = Phrase(text=last.text, boundary_after="end")
    return Script(id=script_id, voice=voice, title=title, phrases=tuple(phrases), source_text=text)


def load_scripts(path: Path) -> tuple[Script, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = []
    for i, item in enumerate(data.get("scripts") or []):
        missing = {"id", "voice", "title", "text"} - set(item)
        if missing:
            raise ValueError(
                f"script {i}: missing {sorted(missing)} (needs id, voice, title, text)"
            )
        out.append(parse_script(item["id"], item["voice"], item["title"], item["text"]))
    return tuple(out)


def check_text(text: str, cfg: C1Settings) -> list[str]:
    """Problems with one script's text; empty if it passes."""
    problems: list[str] = []
    flat = " ".join(text.split())
    if not ALLOWED.match(flat):
        bad = sorted(set(re.sub(r"[A-Za-z ,.?!|]", "", flat)))
        problems.append(f"characters not allowed: {bad}")
    words = [w.lower() for w in WORD.findall(flat)]
    het = sorted({w for w in words if w in HETERONYMS})
    if het:
        problems.append(f"heteronym(s): {het}")
    # a '|' must sit inside a sentence: not at the start, not next to another phrase end
    for m in re.finditer(r"\|", flat):
        before = flat[: m.start()].rstrip()
        after = flat[m.end() :].lstrip()
        if not before or before[-1] in SENTENCE_MARKS + "," or not after or after[0] in "|,.?!":
            problems.append("bar '|' must split a sentence between two words")
            break
    if flat.count("|") < 2:
        problems.append("at least two '|' mid-sentence split points needed")
    marks = sum(flat.count(c) for c in SENTENCE_MARKS)
    if words and (marks == 0 or len(words) / marks > cfg.max_words_per_sentence_mark):
        problems.append(
            f"too few sentence marks: {len(words)} words, {marks} marks "
            f"(at most {cfg.max_words_per_sentence_mark} words per sentence mark)"
        )
    return problems


def check_set(scripts: Sequence[Script], cfg: C1Settings) -> list[str]:
    """Problems with the whole set; empty if it passes."""
    problems: list[str] = []
    ids = Counter(s.id for s in scripts)
    problems += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    lo, hi = cfg.words_per_script
    for s in scripts:
        problems += [f"{s.id}: {p}" for p in check_text(s.source_text or s.text, cfg)]
        if s.voice not in cfg.voices:
            problems.append(f"{s.id}: unknown voice {s.voice}")
        if not lo <= s.words <= hi:
            problems.append(f"{s.id}: {s.words} words, need {lo}-{hi}")
    per_voice = Counter(s.voice for s in scripts)
    for voice in cfg.voices:
        if per_voice[voice] != cfg.scripts_per_voice:
            problems.append(f"{per_voice[voice]} scripts for {voice}, need {cfg.scripts_per_voice}")
    return problems


def words_by_accent(scripts: Sequence[Script], cfg: C1Settings) -> dict[str, set[str]]:
    """Lower-case words each accent's word list must contain (fillers included)."""
    out: dict[str, set[str]] = {code: set() for code in sorted(set(cfg.voices.values()))}
    for s in scripts:
        out[cfg.voices[s.voice]] |= {w.lower() for w in WORD.findall(s.text)}
    for code in out:
        out[code] |= {t.lower() for t in cfg.fillers.tokens}
    return out
