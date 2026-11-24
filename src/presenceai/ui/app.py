"""The PresenceAI web app (P3-c): `uv run presenceai` serves it on http://127.0.0.1:7860.

- English at `/`, Arabic at `/ar/` (right-to-left text); each page links to the other.
- Tabs: Consent, Practice (disabled until "use the app" is saved; recording arrives in P3-d),
  My data, About (with the note on why a part of the report can be "not measurable").
- No outside connection: analytics are off before Gradio is imported (`presenceai.ui`), the
  theme uses system fonts only (no Google Fonts), Gradio's own outside `<link>`/`<script>` tags
  are removed from every page (`ui/offline.py`), the server listens on 127.0.0.1 only, and no
  share link is made. tests/safety/test_app_network.py checks all of it.
- Gradio's upload folder is `PRESENCEAI_HOME/tmp/gradio`, emptied at every start (ADR 0023).
"""

from __future__ import annotations

import os
from typing import Any

import presenceai.ui  # noqa: F401  (analytics off before gradio is imported)
from presenceai.lang import Language, ui_text
from presenceai.ui.context import AppContext
from presenceai.ui.offline import LocalAssetsOnly
from presenceai.ui.pages import consent, data

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


def build_blocks(ctx: AppContext, lang: Language) -> Any:
    import gradio as gr

    rtl = lang == "ar"
    with gr.Blocks(
        theme=theme(), analytics_enabled=False, title=ui_text(lang, "app.title")
    ) as blocks:
        other = ui_text(lang, "app.other_language")
        path = ui_text(lang, "app.other_language_path")
        title, tagline = ui_text(lang, "app.title"), ui_text(lang, "app.tagline")
        gr.Markdown(f"# {title}\n\n{tagline}\n\n[{other}]({path})", rtl=rtl)
        with gr.Tabs():
            with gr.Tab(ui_text(lang, "nav.consent")):
                gr.Markdown(ui_text(lang, "practice.locked"), rtl=rtl)
                wire = consent.build(ctx, lang)
            practice_on = consent.practice_open(ctx)
            with gr.Tab(ui_text(lang, "nav.practice"), interactive=practice_on) as practice:
                gr.Markdown(ui_text(lang, "practice.not_yet"), rtl=rtl)
                if not ctx.key_ok:
                    gr.Markdown(ui_text(lang, "practice.key_missing"), rtl=rtl)
            wire(practice)
            with gr.Tab(ui_text(lang, "nav.data")):
                data.build(ctx, lang)
            with gr.Tab(ui_text(lang, "nav.about")):
                gr.Markdown(f"## {ui_text(lang, 'about.heading')}", rtl=rtl)
                gr.Markdown(ui_text(lang, "about.body"), rtl=rtl)
    return blocks


def create_app(ctx: AppContext) -> Any:
    """FastAPI app with the English page at `/` and the Arabic page at `/ar`."""
    import gradio as gr
    from fastapi import FastAPI

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(LocalAssetsOnly)  # Gradio's page links Google Fonts and cdnjs
    app = gr.mount_gradio_app(app, build_blocks(ctx, "ar"), path="/ar", show_api=False)
    return gr.mount_gradio_app(app, build_blocks(ctx, "en"), path="", show_api=False)


def main() -> None:
    """Start the app on 127.0.0.1:7860 (the `presenceai` command with no arguments)."""
    from presenceai.settings import get_settings

    settings = get_settings()
    ctx = AppContext.open(
        settings.presenceai_home,
        settings.media_key,
        raw_media_days=settings.retention.raw_media_days,
    )
    os.environ["GRADIO_TEMP_DIR"] = str(ctx.layout.tmp_dir / "gradio")
    import uvicorn

    try:
        uvicorn.run(create_app(ctx), host=HOST, port=PORT, log_level="warning")
    finally:
        ctx.close()
