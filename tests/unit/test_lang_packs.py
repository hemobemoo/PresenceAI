from collections.abc import Iterator
from typing import Any

import pytest

from presenceai.lang import LANGUAGES, Language, load_ui_strings


def _keys(node: dict[str, Any], prefix: str = "") -> set[str]:
    out: set[str] = set()
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out |= _keys(value, path + ".")  # pyright: ignore[reportUnknownArgumentType]
        else:
            out.add(path)
    return out


def _leaves(node: dict[str, Any]) -> Iterator[Any]:
    for value in node.values():
        if isinstance(value, dict):
            yield from _leaves(value)  # pyright: ignore[reportUnknownArgumentType]
        else:
            yield value


def test_en_and_ar_have_identical_keys():
    assert _keys(load_ui_strings("en")) == _keys(load_ui_strings("ar"))


def test_required_keys_present():
    assert {"app.title", "consent.heading"} <= _keys(load_ui_strings("en"))


@pytest.mark.parametrize("lang", LANGUAGES)
def test_no_empty_strings(lang: Language):
    for value in _leaves(load_ui_strings(lang)):
        assert isinstance(value, str)
        assert value.strip()


def test_unknown_language_rejected():
    with pytest.raises(ValueError):
        load_ui_strings("fr")  # pyright: ignore[reportArgumentType]
