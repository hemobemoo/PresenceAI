"""Brand assets (owner, 2026-11-25): the logo and favicon come from the app's own static folder,
with no outside request, and both header images carry alt text. Same approach as
tests/safety/test_recording_network.py: the loopback-only guard is on while the pages and files
are fetched."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from presenceai.evalconfig import REPO_ROOT
from presenceai.ui.app import BRAND_DIR, BRAND_URL, create_app
from presenceai.ui.context import AppContext

FILES = ("presenceai-logo.svg", "presenceai-logo-dark.svg", "presenceai-icon.svg", "favicon.ico")


@pytest.fixture
def client(tmp_path: Path, loopback_only: list[str]) -> Any:
    from fastapi.testclient import TestClient

    ctx = AppContext.open(
        tmp_path / "home", SecretStr(Fernet.generate_key().decode()), raw_media_days=30
    )
    yield TestClient(create_app(ctx))
    ctx.close()
    assert loopback_only == []  # nothing tried to leave the machine


def _header(config: dict[str, Any]) -> str:
    values = [
        str(c.get("props", {}).get("value", ""))
        for c in config["components"]
        if c.get("type") == "html"
    ]
    return next(v for v in values if "pai-logo" in v)


@pytest.mark.parametrize("page", ["", "/ar"])
def test_both_header_images_are_local_and_have_alt_text(client: Any, page: str):
    header = _header(client.get(f"{page}/config").json())
    direction = "rtl" if page == "/ar" else "ltr"
    assert f'dir="{direction}"' in header  # the logo starts on the right on the Arabic page
    imgs = re.findall(r"<img\b[^>]*>", header)
    assert len(imgs) == 2  # light and dark
    for img in imgs:
        src = re.search(r'src="([^"]+)"', img)
        assert src and src.group(1).startswith(f"{BRAND_URL}/")  # the app's own folder
        assert 'alt="PresenceAI"' in img
        assert re.search(r'height="\d+"', img)  # fixed height: no layout jump


def test_logo_and_favicon_are_served_from_the_static_folder(client: Any):
    for name in FILES:
        r = client.get(f"{BRAND_URL}/{name}")
        assert r.status_code == 200, name
        assert r.content == (BRAND_DIR / name).read_bytes(), name
    assert "svg" in client.get(f"{BRAND_URL}/presenceai-logo.svg").headers["content-type"]
    for page in ("", "/ar"):  # both pages, served by the app itself (not Gradio's ignored option)
        fav = client.get(f"{page}/favicon.ico")
        assert fav.status_code == 200 and fav.content == (BRAND_DIR / "favicon.ico").read_bytes()


def test_the_svgs_fetch_nothing_from_outside():
    for name in FILES[:3]:
        svg = (BRAND_DIR / name).read_text("utf-8")
        assert not re.search(r"<script|<image|@import|url\(\s*['\"]?https?:", svg, re.I), name
        hrefs = re.findall(r'(?:xlink:)?href\s*=\s*"([^"]+)"', svg)
        assert all(h.startswith("#") for h in hrefs), name  # in-file references only


def test_nothing_logo_related_is_left_in_the_repository_root():
    for name in (*FILES, "preview.png", "Logo.png", "Logo.svg"):
        assert not (REPO_ROOT / name).exists(), name


@pytest.mark.parametrize("name", [f for f in FILES if f.endswith(".svg")])
def test_svg_files_carry_no_embedded_metadata(name: str):
    # owner, 2026-11-25: the C2PA content credential added on delivery was stripped; the
    # drawing is shapes and gradients only
    text = (BRAND_DIR / name).read_text("utf-8").lower()
    assert "c2pa" not in text and "<metadata" not in text
    assert len(text) < 10_000  # a 7.7 KB base64 block would push the files past this
