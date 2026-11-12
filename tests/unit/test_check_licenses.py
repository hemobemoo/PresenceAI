"""Tests for scripts/check_licenses.py with fake distribution metadata."""

import importlib.util
import sys
from email.message import Message
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_licenses.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_licenses", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_licenses"] = module
    spec.loader.exec_module(module)
    return module


cl: Any = _load_script()


def _meta(
    *,
    version: str = "1.0",
    expression: str | None = None,
    licence: str | None = None,
    classifiers: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
) -> metadata.PackageMetadata:
    msg = Message()
    msg["Version"] = version
    if expression:
        msg["License-Expression"] = expression
    if licence:
        msg["License"] = licence
    for c in classifiers:
        msg["Classifier"] = c
    for r in requires:
        msg["Requires-Dist"] = r
    return cast("metadata.PackageMetadata", msg)


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (["MIT"], "OK"),
        (["Apache-2.0"], "OK"),
        (["OSI Approved :: BSD License"], "OK"),
        (["GPL-3.0-or-later"], "FAIL"),
        (["AGPL-3.0-only"], "FAIL"),
        (["OSI Approved :: GNU General Public License v3 (GPLv3)"], "FAIL"),
        (["OSI Approved :: GNU Affero General Public License v3"], "FAIL"),
        (["LGPL-2.1-or-later"], "WARN"),
        (["OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)"], "WARN"),
        (["MIT OR GPL-2.0"], "WARN"),
        ([], "WARN"),
    ],
)
def test_classify(texts: list[str], expected: str):
    assert cl.classify(texts) == expected


def test_closure_skips_extras_and_missing_and_flags_gpl():
    fake = {
        "presenceai": _meta(
            requires=("typer>=0.1", "laya; extra == 'laya'", "winonly; sys_platform == 'win32'")
        ),
        "typer": _meta(expression="MIT", requires=("click",)),
        "click": _meta(classifiers=("License :: OSI Approved :: GNU General Public License v2",)),
        "laya": _meta(expression="GPL-3.0"),  # behind an extra: must not be checked
    }

    def lookup(name: str) -> metadata.PackageMetadata:
        if name not in fake:
            raise metadata.PackageNotFoundError(name)
        return fake[name]

    results = {r.name: r.status for r in cl.check("presenceai", lookup)}
    assert results == {"typer": "OK", "click": "FAIL"}


def test_installed_environment_has_no_gpl():
    results = cl.check()
    assert results, "presenceai runtime dependencies should be installed"
    assert not [r for r in results if r.status == "FAIL"]
