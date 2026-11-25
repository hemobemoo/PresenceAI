"""Keep the browser on this computer (owner, P3 plan: no outside connection; ADR 0024 correction).

Three layers, all in Python on the server:

1. **Outside tags removed from every page.** Gradio 5.23's `templates/frontend/index.html`
   hard-codes `<link rel="preconnect">` to Google Fonts and a `<script>` from cdnjs (iframe
   resizer, only for embedding). They are deleted from every HTML response.
2. **Known outside fetches rewritten in served scripts.** FastRTC 0.0.34's microphone component
   plays a sound fetched from Hugging Face on every page load (`new Audio("https://huggingface.co
   /…/pop-sounds.mp3")`). Every request revealed the user's IP address and the page-load time to
   Hugging Face (nothing else). The URL is rewritten to `/local/silent.wav`, served by the app.
3. **A Content-Security-Policy on every page** allows scripts, styles, fonts, images, media and
   connections from this server only ('self', plus inline code, data: and blob: that Gradio
   needs), so any outside fetch a future Gradio or FastRTC adds is blocked by the browser.
   WebRTC itself is not governed by CSP; it uses no STUN/TURN server (ADR 0024 item 2).

tests/safety/test_app_network.py checks the served pages and scripts; tests/ui/
test_browser_journey.py runs a full S2 session in a real browser and fails on any request to a
host other than this computer.
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

SILENT_URL = "/local/silent.wav"
# Outside URLs that served scripts fetch at run time, and their local replacements.
POP_SOUND = "https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/pop-sounds.mp3"
REWRITES: dict[str, str] = {POP_SOUND: SILENT_URL}
CSP = "; ".join(
    (
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "media-src 'self' data: blob:",
        "font-src 'self' data:",
        "connect-src 'self' data: blob:",
        "worker-src 'self' blob:",
        "frame-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    )
)


def strip_outside_assets(html: str) -> str:
    return _TAGS.sub("", html)


def rewrite_outside_fetches(script: str) -> str:
    for outside, local in REWRITES.items():
        script = script.replace(outside, local)
    return script


def silent_wav() -> bytes:
    """0.1 s of silence: what the microphone component plays instead of the outside sound."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


class LocalAssetsOnly(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        kind = response.headers.get("content-type", "")
        is_html = kind.startswith("text/html")
        is_script = "javascript" in kind
        if not (is_html or is_script) or response.headers.get("content-encoding"):
            if is_html:
                response.headers["Content-Security-Policy"] = CSP
            return response
        body = b""
        async for chunk in response.body_iterator:  # type: ignore[attr-defined]
            body += chunk if isinstance(chunk, bytes) else chunk.encode()
        text = body.decode("utf-8")
        text = strip_outside_assets(text) if is_html else rewrite_outside_fetches(text)
        headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
        if is_html:
            headers["Content-Security-Policy"] = CSP
        return Response(text, status_code=response.status_code, headers=headers)
