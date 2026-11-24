"""English and Arabic language packs: lexicons, templates, UI strings, filler lists, banned-term lists.

Report texts (measurement methods, "not measurable" reasons, notes, unit labels, the language
notice) live in `lang/<lang>/report.yaml` with named placeholders such as `{snr:.0f}`. Speech code
uses keys only, through `text(lang, key, **values)`; the report is written in the recording's
language (ADR 0016). Both files must have the same keys and placeholders
(`tests/unit/test_report_texts.py`).
"""

from __future__ import annotations

import string
from functools import cache
from importlib.resources import files
from typing import Any, Literal, cast

import yaml

Language = Literal["en", "ar"]
LANGUAGES: tuple[Language, ...] = ("en", "ar")
REPORT_FILE = "report.yaml"


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


def flatten(node: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Nested mapping -> {"a.b.c": leaf}. Leaves must be strings."""
    out: dict[str, str] = {}
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out |= flatten(cast("dict[str, Any]", value), path + ".")
        elif isinstance(value, str):
            out[path] = value
        else:
            raise ValueError(f"{path}: report texts must be strings")
    return out


@cache
def report_texts(lang: Language) -> dict[str, str]:
    """All report texts of one language, flattened to dotted keys."""
    return flatten(load_pack_file(lang, REPORT_FILE))


def placeholders(template: str) -> frozenset[str]:
    """Named fields in a template ("{rate:.1f} per {n}" -> {"rate", "n"})."""
    return frozenset(name for _, name, _, _ in string.Formatter().parse(template) if name)


def text(lang: Language, key: str, **values: Any) -> str:
    """The report text `key` in `lang`, with its placeholders filled in.

    Raises KeyError for an unknown key, and ValueError when `values` do not match the
    template's placeholders exactly (a missing value would otherwise show as "{snr}").
    """
    template = report_texts(lang)[key]
    expected = placeholders(template)
    if set(values) != expected:
        raise ValueError(f"{lang}:{key} needs {sorted(expected)}, got {sorted(values)}")
    return template.format(**values)


def unit_label(lang: Language, unit: str) -> str:
    """Display label of a measurement unit code (`Measurement.unit` stays a stable code)."""
    return report_texts(lang)[f"units.{unit}"]


def language_name(lang: Language, code: str) -> str:
    """Name of a detected language in `lang`; codes other than en/ar are shown as codes."""
    names = report_texts(lang)
    key = f"language_names.{code}"
    return names[key] if key in names else text(lang, "language_names.other", code=code)


@cache
def ui_texts(lang: Language) -> dict[str, str]:
    """All UI strings of one language (`lang/<lang>/ui.yaml`), flattened to dotted keys."""
    return flatten(load_ui_strings(lang))


def ui_text(lang: Language, key: str, **values: Any) -> str:
    """The UI string `key` in `lang`, placeholders filled exactly as for `text`."""
    template = ui_texts(lang)[key]
    expected = placeholders(template)
    if set(values) != expected:
        raise ValueError(f"{lang}:{key} needs {sorted(expected)}, got {sorted(values)}")
    return template.format(**values) if expected else template
