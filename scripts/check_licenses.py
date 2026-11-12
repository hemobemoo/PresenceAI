"""Fail if any runtime dependency of PresenceAI is GPL or AGPL (AGENTS.md rule 6).

Run after adding a dependency:
    uv run python scripts/check_licenses.py

Walks the installed runtime requirements of the `presenceai` distribution
transitively (no extras, no dev groups) and reads each distribution's licence
metadata: `License-Expression`, `License` and `License ::` classifiers.

    FAIL  GPL / AGPL (exit code 1)
    WARN  LGPL, dual licences that include GPL, or no licence metadata (review by hand)
    OK    everything else, and LGPL packages in LGPL_ALLOWED (reviewed, with a reason)

Uses only the standard library.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata

ROOT_DIST = "presenceai"
OK, WARN, FAIL = "OK", "WARN", "FAIL"

# LGPL dependencies the project owner has reviewed and accepted (listed in NOTICE).
# LGPL is allowed; GPL and AGPL stay blocked whatever this list says.
LGPL_ALLOWED: dict[str, str] = {
    "soxr": (
        "LGPL-2.1-or-later, used unmodified as a separately installed library "
        "(via fastrtc -> librosa), which is compatible with Apache-2.0; accepted by the "
        "owner on 2026-11-12, see NOTICE"
    ),
}

_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_EXTRA_MARKER = re.compile(r"""\bextra\s*==""")
_GPL = re.compile(r"(?<![L])\bA?GPL|GNU (AFFERO )?GENERAL PUBLIC LICEN[CS]E")
_LGPL = re.compile(r"\bLGPL|GNU LESSER GENERAL PUBLIC|GNU LIBRARY")


@dataclass(frozen=True)
class Result:
    name: str
    version: str
    licence: str
    status: str


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def licence_texts(meta: metadata.PackageMetadata) -> list[str]:
    """Short licence strings from a distribution's metadata."""
    texts: list[str] = []
    expression = meta.get("License-Expression")
    if expression:
        texts.append(expression.strip())
    licence = meta.get("License")
    if licence and licence.strip().upper() not in {"UNKNOWN", ""}:
        first_line = licence.strip().splitlines()[0]
        texts.append(first_line[:80])
    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            texts.append(classifier.removeprefix("License :: ").strip())
    return texts


def classify(texts: Iterable[str]) -> str:
    texts = [t for t in texts if t]
    if not texts:
        return WARN
    upper = [t.upper() for t in texts]
    gpl = [t for t in upper if _GPL.search(t) and not _LGPL.search(t)]
    if gpl:
        dual = any(" OR " in t for t in gpl)
        return WARN if dual else FAIL
    if any(_LGPL.search(t) for t in upper):
        return WARN
    return OK


def runtime_requirements(meta: metadata.PackageMetadata) -> list[str]:
    names: list[str] = []
    for req in meta.get_all("Requires-Dist") or []:
        requirement, _, marker = req.partition(";")
        if marker and _EXTRA_MARKER.search(marker):
            continue  # optional extra, not part of the runtime group
        match = _REQ_NAME.match(requirement)
        if match:
            names.append(normalise(match.group(1)))
    return names


def runtime_closure(
    root: str = ROOT_DIST,
    lookup: Callable[[str], metadata.PackageMetadata] = metadata.metadata,
) -> dict[str, metadata.PackageMetadata]:
    """All installed distributions reachable from `root` (root excluded)."""
    seen: dict[str, metadata.PackageMetadata] = {}
    stack = runtime_requirements(lookup(root))
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        try:
            meta = lookup(name)
        except metadata.PackageNotFoundError:
            continue  # requirement for another platform (environment marker), not installed
        seen[name] = meta
        stack.extend(runtime_requirements(meta))
    return seen


def check(
    root: str = ROOT_DIST,
    lookup: Callable[[str], metadata.PackageMetadata] = metadata.metadata,
) -> list[Result]:
    results: list[Result] = []
    for name, meta in sorted(runtime_closure(root, lookup).items()):
        texts = licence_texts(meta)
        licence = "; ".join(texts) or "(no licence metadata)"
        status = classify(texts)
        # The allow-list only turns a pure-LGPL WARN into OK. GPL/AGPL (classified FAIL),
        # dual licences with GPL and missing metadata are never covered by it.
        lgpl_only = status == WARN and any(_LGPL.search(t.upper()) for t in texts)
        if lgpl_only and name in LGPL_ALLOWED:
            status = OK
            licence = f"{licence} [allowed: {LGPL_ALLOWED[name]}]"
        results.append(
            Result(name=name, version=meta.get("Version") or "?", licence=licence, status=status)
        )
    return results


def main() -> int:
    try:
        results = check()
    except metadata.PackageNotFoundError:
        print(f"'{ROOT_DIST}' is not installed; run `uv sync` first.")
        return 2
    width = max((len(r.name) for r in results), default=10)
    for r in results:
        print(f"{r.status:<4}  {r.name:<{width}}  {r.version:<12}  {r.licence}")
    fails = sum(r.status == FAIL for r in results)
    warns = sum(r.status == WARN for r in results)
    print(f"\n{len(results)} runtime distributions: {fails} FAIL, {warns} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
