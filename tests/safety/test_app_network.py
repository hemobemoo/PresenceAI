"""The app phones nobody (owner, P3 plan): Gradio analytics off, no external fonts or CDN assets,
no non-loopback connection when the app is imported and its pages are built and served.

`import gradio` starts a thread that fetches messages from api.gradio.app unless
GRADIO_ANALYTICS_ENABLED is already "False" before the first import, so the import check runs in
a fresh Python process with a socket guard installed first.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

_PROBE = textwrap.dedent(
    """
    import ipaddress, os, socket, sys, time
    os.environ.pop("GRADIO_ANALYTICS_ENABLED", None)  # as on a fresh machine
    blocked = []
    real = socket.socket.connect

    def loopback(address):
        if not isinstance(address, tuple):
            return True
        host = str(address[0])
        try:
            return host == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def connect(self, address):
        if not loopback(address):
            blocked.append(str(address[0]))
            raise OSError("blocked")
        return real(self, address)

    socket.socket.connect = connect
    real_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        if host not in (None, "localhost", "127.0.0.1", "::1"):
            blocked.append(str(host))
            raise OSError("blocked")
        return real_getaddrinfo(host, *args, **kwargs)

    socket.getaddrinfo = getaddrinfo

    from pathlib import Path
    from pydantic import SecretStr
    import presenceai.ui.app as ui_app
    from presenceai.ui.context import AppContext

    ctx = AppContext.open(Path(sys.argv[1]), SecretStr(sys.argv[2]), raw_media_days=30)
    from fastapi.testclient import TestClient
    client = TestClient(ui_app.create_app(ctx))
    assert client.get("/").status_code == 200
    assert client.get("/ar/").status_code == 200
    time.sleep(2.0)  # give any background thread the chance to call out
    ctx.close()
    print("ANALYTICS=" + os.environ.get("GRADIO_ANALYTICS_ENABLED", "unset"))
    print("BLOCKED=" + ",".join(blocked))
    """
)


def test_importing_and_serving_the_app_makes_no_outside_connection(tmp_path: Path):
    key = Fernet.generate_key().decode()
    out = subprocess.run(  # noqa: S603  (our own probe, fixed arguments)
        [sys.executable, "-c", _PROBE, str(tmp_path / "home"), key],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ANALYTICS=False" in out.stdout
    assert "BLOCKED=\n" in out.stdout or out.stdout.rstrip().endswith("BLOCKED="), out.stdout


_URL = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


def _external(urls: list[str]) -> list[str]:
    bad = []
    for u in urls:
        if u.startswith(("http://", "https://", "//")):
            host = re.sub(r"^(?:https?:)?//", "", u).split("/")[0].split(":")[0]
            if host not in ("localhost", "127.0.0.1", "testserver"):
                bad.append(u)
    return bad


def test_pages_load_no_external_fonts_or_cdn_assets(tmp_path: Path):
    from fastapi.testclient import TestClient

    from presenceai.ui.app import create_app
    from presenceai.ui.context import AppContext

    ctx = AppContext.open(
        tmp_path / "home", SecretStr(Fernet.generate_key().decode()), raw_media_days=30
    )
    client: Any = TestClient(create_app(ctx))
    try:
        for path in ("/", "/ar/"):
            page = client.get(path).text
            assert _external(_URL.findall(page)) == [], path
            for asset in _URL.findall(page):
                if asset.endswith(".css") and not _external([asset]):
                    css = client.get(asset if asset.startswith("/") else f"{path}{asset}").text
                    assert "fonts.googleapis" not in css and "fonts.gstatic" not in css
            assert "fonts.googleapis" not in page and "fonts.gstatic" not in page
            config = client.get(f"{path.rstrip('/')}/config").json()
            assert "googleapis" not in str(config.get("stylesheets", ""))
            assert config.get("analytics_enabled") is False
    finally:
        ctx.close()


@pytest.mark.parametrize("module", ["presenceai.ui", "presenceai.ui.app"])
def test_the_ui_package_turns_analytics_off_before_gradio_loads(module: str):
    code = (
        "import os, sys; os.environ.pop('GRADIO_ANALYTICS_ENABLED', None); "
        f"import {module}; "
        "e = os.environ; "
        "print(e.get('GRADIO_ANALYTICS_ENABLED'), e.get('HF_HUB_DISABLE_TELEMETRY'))"
    )
    out = subprocess.run(  # noqa: S603  (our own code, fixed arguments)
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert out.returncode == 0, out.stderr[-1000:]
    assert out.stdout.split() == ["False", "1"]


def test_outside_tags_are_stripped_and_local_ones_kept():
    from presenceai.ui.offline import strip_outside_assets

    html = (
        '<link rel="preconnect" href="https://fonts.googleapis.com" />'
        '<link\n rel="preconnect"\n href="https://fonts.gstatic.com" crossorigin="anonymous" />'
        '<script src="https://cdnjs.cloudflare.com/x.js" async></script>'
        '<link rel="stylesheet" href="./assets/index.css" />'
        '<script type="module" src="./assets/index.js"></script>'
        '<link rel="icon" href="http://127.0.0.1:7860/favicon.ico" />'
    )
    out = strip_outside_assets(html)
    assert "googleapis" not in out and "gstatic" not in out and "cdnjs" not in out
    assert "./assets/index.css" in out and "./assets/index.js" in out and "127.0.0.1" in out
