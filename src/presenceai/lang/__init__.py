"""English and Arabic language packs: lexicons, templates, UI strings, filler lists, banned-term lists."""

from __future__ import annotations

from importlib.resources import files
from typing import Any, Literal, cast

import yaml

Language = Literal["en", "ar"]
LANGUAGES: tuple[Language, ...] = ("en", "ar")


def load_pack_file(lang: Language, name: str) -> dict[str, Any]:
    """Load ``lang/<lang>/<name>`` (a YAML mapping) from the installed package."""
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language: {lang!r}")
    text = files(__package__).joinpath(lang, name).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{lang}/{name} must contain a mapping")
    return cast("dict[str, Any]", data)


def load_ui_strings(lang: Language) -> dict[str, Any]:
    """Nested UI string table for one language."""
    return load_pack_file(lang, "ui.yaml")
