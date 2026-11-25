"""The PresenceAI web app (P3-c): `uv run presenceai` serves it on http://127.0.0.1:7860.

- English at `/`, Arabic at `/ar/` (right-to-left text); each page links to the other.
- Tabs: Consent, Practice (disabled until "use the app" is saved; record or upload, then the
  report, P3-d),
  My data, About (with the note on why a part of the report can be "not measurable").
- No outside connection: analytics are off before Gradio is imported (`presenceai.ui`), the
  theme uses system fonts only (no Google Fonts), Gradio's own outside `<link>`/`<script>` tags
  are removed from every page (`ui/offline.py`), the server listens on 127.0.0.1 only, and no
  share link is made. tests/safety/test_app_network.py checks all of it.
- Gradio's upload folder is `PRESENCEAI_HOME/tmp/gradio`, emptied at every start (ADR 0023).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import presenceai.ui  # noqa: F401  (analytics off before gradio is imported)
from presenceai.lang import Language, ui_text
from presenceai.pipeline.app_run import Analyzer
from presenceai.settings import Settings
from presenceai.ui.context import AppContext
from presenceai.ui.offline import SILENT_URL, LocalAssetsOnly, silent_wav
from presenceai.ui.pages import consent, data
from presenceai.ui.pages import practice as practice_page

HOST = "127.0.0.1"
PORT = 7860
SYSTEM_FONTS = ("Segoe UI", "Tahoma", "Arial", "sans-serif")
MONO_FONTS = ("Consolas", "monospace")


def theme() -> Any:
    """System fonts only. A plain string would become a `LocalFont` that requests
    /static/fonts/<name>/*.woff2 (404s); the base `Font` emits no stylesheet at all."""
    from gradio.themes import Default
    from gradio.themes.utils.fonts import Font

    return Default(
        font=[Font(name) for name in SYSTEM_FONTS], font_mono=[Font(n) for n in MONO_FONTS]
    )


BRAND_DIR = Path(__file__).parent / "static" / "brand"
BRAND_URL = "/brand"  # served from BRAND_DIR by this app; never an outside URL
LOGO_HEIGHT_PX = 40
# The dark logo shows only under Gradio's dark theme (the `.dark` class it sets); the light one
# otherwise. A fixed height, so the header never pushes the page down.
HEADER_CSS = f"""
.pai-logo {{ display: flex; }}  /* starts at the page's start edge: right on the Arabic page */
.pai-logo img {{ height: {LOGO_HEIGHT_PX}px; width: auto; display: block; }}
.pai-logo .pai-logo-dark {{ display: none; }}
.dark .pai-logo .pai-logo-light {{ display: none; }}
.dark .pai-logo .pai-logo-dark {{ display: block; }}
"""


def header_html(lang: Language) -> str:
    """The logo, light and dark, from the local static folder; alt text from lang/."""
    alt = ui_text(lang, "app.logo_alt")
    direction = "rtl" if lang == "ar" else "ltr"
    return (
        f'<div class="pai-logo" dir="{direction}">'
        f'<img class="pai-logo-light" src="{BRAND_URL}/presenceai-logo.svg" alt="{alt}"'
        f' height="{LOGO_HEIGHT_PX}">'
        f'<img class="pai-logo-dark" src="{BRAND_URL}/presenceai-logo-dark.svg" alt="{alt}"'
        f' height="{LOGO_HEIGHT_PX}">'
        "</div>"
    )


def build_blocks(ctx: AppContext, lang: Language, analyzer: Analyzer | None = None) -> Any:
    import gradio as gr

    rtl = lang == "ar"
    with gr.Blocks(
        theme=theme(), analytics_enabled=False, title=ui_text(lang, "app.title"), css=HEADER_CSS
    ) as blocks:
        other = ui_text(lang, "app.other_language")
        path = ui_text(lang, "app.other_language_path")
        tagline = ui_text(lang, "app.tagline")
        gr.HTML(header_html(lang))
        gr.Markdown(f"{tagline}\n\n[{other}]({path})", rtl=rtl)
        with gr.Tabs():
            with gr.Tab(ui_text(lang, "nav.consent")):
                gr.Markdown(ui_text(lang, "practice.locked"), rtl=rtl)
                wire = consent.build(ctx, lang)
            practice_on = consent.practice_open(ctx)
            with gr.Tab(ui_text(lang, "nav.practice"), interactive=practice_on) as practice:
                practice_page.build(ctx, lang, analyzer or default_analyzer())
            wire(practice, blocks)
            with gr.Tab(ui_text(lang, "nav.data")):
                wire_data = data.build(ctx, lang)
            wire_data(blocks)
            with gr.Tab(ui_text(lang, "nav.about")):
                gr.Markdown(f"## {ui_text(lang, 'about.heading')}", rtl=rtl)
                gr.Markdown(ui_text(lang, "about.body"), rtl=rtl)
    return blocks


def default_analyzer() -> Analyzer:
    """Config only, never the `.env` file: the analysis needs no secret."""
    return Analyzer(Settings(_env_file=None))  # pyright: ignore[reportCallIssue]


def create_app(ctx: AppContext, analyzer: Analyzer | None = None) -> Any:
    """FastAPI app with the English page at `/` and the Arabic page at `/ar`. One analyser
    for both pages, so only one analysis runs at a time (ADR 0024)."""
    analyzer = analyzer or default_analyzer()
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    # outside tags removed, FastRTC's Hugging Face sound rewritten, CSP on every page
    app.add_middleware(LocalAssetsOnly)
    app.mount(BRAND_URL, StaticFiles(directory=BRAND_DIR), name="brand")  # before Gradio's "/"
    favicon = BRAND_DIR / "favicon.ico"

    # Gradio ignores `favicon_path` for a mounted app (it warns at every start), so the favicon
    # is served here, for both pages, from the local static folder.
    def serve_favicon() -> FileResponse:
        return FileResponse(favicon, media_type="image/x-icon")

    for route in ("/favicon.ico", "/ar/favicon.ico"):
        app.add_api_route(route, serve_favicon, include_in_schema=False)

    # the microphone component's sound, served here instead of from Hugging Face (ui/offline.py)
    def serve_silence() -> Response:
        return Response(silent_wav(), media_type="audio/wav")

    app.add_api_route(SILENT_URL, serve_silence, include_in_schema=False)
    app = gr.mount_gradio_app(app, build_blocks(ctx, "ar", analyzer), path="/ar", show_api=False)
    return gr.mount_gradio_app(app, build_blocks(ctx, "en", analyzer), path="", show_api=False)


def ignore_connection_resets(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """Drop Windows' harmless WinError 10054 (the browser closed a connection; asyncio's Proactor
    loop logs a traceback for each), so real errors stay visible (owner, round-1 test: 633 of
    them buried the log). Everything else goes to asyncio's default handler."""
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
        return
    loop.default_exception_handler(context)


def configure_logging() -> None:
    """One line per analysis start and end (ADR 0024), never any user text."""
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("presenceai").setLevel(logging.INFO)


URL = f"http://{HOST}:{PORT}/"


def ready_message() -> str:
    """The line printed once the server listens: Ctrl+click opens the app (owner, 2026-11-25)."""
    return ui_text("en", "app.running", url=URL)


def _server(app: Any) -> Any:
    import uvicorn

    class Server(uvicorn.Server):
        async def startup(self, sockets: Any = None) -> None:
            await super().startup(sockets)
            if self.started:  # only once it really listens, so the link always works
                print(ready_message(), flush=True)

    return Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))


def main() -> None:
    """Start the app on 127.0.0.1:7860 (the `presenceai` command with no arguments)."""
    from presenceai.settings import get_settings

    configure_logging()
    settings = get_settings()
    ctx = AppContext.open(
        settings.presenceai_home,
        settings.media_key,
        raw_media_days=settings.retention.raw_media_days,
    )
    os.environ["GRADIO_TEMP_DIR"] = str(ctx.layout.tmp_dir / "gradio")
    try:
        app = create_app(ctx, Analyzer(settings))
        server = _server(app)
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(ignore_connection_resets)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
        if not server.started:  # the port was taken: one clear line instead of tracebacks
            print(ui_text("en", "app.port_busy", url=URL), flush=True)
    finally:
        ctx.close()
