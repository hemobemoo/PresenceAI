"""Consent page (FINAL_11 §3-§4): switches (a)-(d), all off until saved. Practice stays
disabled until (a) "use the app" is saved. (b) camera and (d) hosted AI are stored but not used
before P6 and P7; the page says so."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from presenceai.lang import Language, ui_text
from presenceai.schemas.storage import Consent
from presenceai.ui.context import AppContext


def practice_open(ctx: AppContext) -> bool:
    return ctx.consent().use_app


def save(
    ctx: AppContext, lang: Language, consent: Consent, *, now: datetime | None = None
) -> tuple[str, bool]:
    """Store the switches; returns the message and whether practice is now open."""
    ctx.db.set_consent(consent, now=now or datetime.now(UTC))
    is_open = consent.use_app
    return ui_text(lang, "consent.saved_open" if is_open else "consent.saved_locked"), is_open


def build(ctx: AppContext, lang: Language) -> Any:
    """The page's components, inside the caller's Tab. Returns `wire(practice_tab, blocks)`,
    called once the Practice tab exists: saving opens or locks it, and every page load reads the
    saved consent again (round-1 test: a reload showed Practice locked after consent was saved,
    because the state was read once at app start)."""
    import gradio as gr

    rtl = lang == "ar"
    current = ctx.consent()
    gr.Markdown(f"## {ui_text(lang, 'consent.heading')}", rtl=rtl)
    gr.Markdown(ui_text(lang, "consent.subheading"), rtl=rtl)
    use_app = gr.Checkbox(value=current.use_app, label=ui_text(lang, "consent.switches.use_app"))
    camera = gr.Checkbox(
        value=current.camera,
        label=ui_text(lang, "consent.switches.camera"),
        info=ui_text(lang, "consent.camera_note"),
    )
    keep = gr.Checkbox(
        value=current.keep_recordings,
        label=ui_text(lang, "consent.switches.keep_recordings"),
        info=ui_text(lang, "consent.keep_note", days=ctx.raw_media_days),
    )
    hosted = gr.Checkbox(
        value=current.hosted_ai,
        label=ui_text(lang, "consent.switches.hosted_ai"),
        info=ui_text(lang, "consent.hosted_ai_note"),
    )
    button = gr.Button(ui_text(lang, "consent.save"), variant="primary")
    status = gr.Markdown(rtl=rtl)

    def on_save(a: bool, b: bool, c: bool, d: bool) -> tuple[str, Any]:
        msg, is_open = save(ctx, lang, Consent(use_app=a, camera=b, keep_recordings=c, hosted_ai=d))
        return msg, gr.Tab(interactive=is_open)

    def on_load() -> tuple[Any, ...]:
        c = ctx.consent()
        return (
            c.use_app, c.camera, c.keep_recordings, c.hosted_ai,
            gr.Tab(interactive=c.use_app),
        )  # fmt: skip

    def wire(practice_tab: Any, blocks: Any) -> None:
        button.click(on_save, [use_app, camera, keep, hosted], [status, practice_tab])
        outputs = [use_app, camera, keep, hosted, practice_tab]
        blocks.load(on_load, outputs=outputs, queue=False, show_progress="hidden")

    return wire
