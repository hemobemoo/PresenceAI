"""Remove outside assets from Gradio's page (owner, P3 plan: no external fonts or CDN assets).

Gradio 5.23's `templates/frontend/index.html` hard-codes `<link rel="preconnect">` to Google
Fonts and a `<script>` from cdnjs.cloudflare.com (iframe-resizer, needed only when the app is
embedded in another page's iframe, as on Hugging Face Spaces). A browser opening the page would
contact both. This middleware deletes those tags from every HTML response; the theme already
uses system fonts only (`ui/app.py`). tests/safety/test_app_network.py fails if any `src` or
`href` in a served page points outside the machine, so a Gradio upgrade that adds another one is
caught.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_OUTSIDE = r"""(?:https?:)?//(?!(?:localhost|127\.0\.0\.1)[:/"'])"""
_TAGS = re.compile(
    rf"""<link\b[^>]*\bhref\s*=\s*["']{_OUTSIDE}[^>]*>"""
    rf"""|<script\b[^>]*\bsrc\s*=\s*["']{_OUTSIDE}[^>]*>\s*</script>""",
    re.IGNORECASE | re.DOTALL,
)


def strip_outside_assets(html: str) -> str:
    return _TAGS.sub("", html)


class LocalAssetsOnly(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        if not response.headers.get("content-type", "").startswith("text/html"):
            return response
        body = b""
        async for chunk in response.body_iterator:  # type: ignore[attr-defined]
            body += chunk if isinstance(chunk, bytes) else chunk.encode()
        html = strip_outside_assets(body.decode("utf-8"))
        headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
        return Response(html, status_code=response.status_code, headers=headers)
