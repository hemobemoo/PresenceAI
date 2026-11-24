"""My data page (FINAL_11 §3-§4, ADR 0023): the sessions on this computer, export one or all,
delete one or everything. Deleting everything needs a ticked confirmation."""

from __future__ import annotations

from typing import Any

from presenceai.lang import Language, ui_text
from presenceai.schemas.storage import SessionRecord
from presenceai.storage import media, rights
from presenceai.ui.context import AppContext


def _recording(lang: Language, s: SessionRecord) -> str:
    if s.media_deleted_at is not None:
        return ui_text(lang, "data.recording.deleted")
    if s.media_expires_at is None:
        return ui_text(lang, "data.recording.after_analysis")
    return ui_text(lang, "data.recording.kept_until", date=s.media_expires_at.date().isoformat())


def _scenario(lang: Language, scenario: str) -> str:
    try:
        return ui_text(lang, f"data.scenarios.{scenario}")
    except KeyError:
        return scenario


def rows(ctx: AppContext, lang: Language) -> list[list[str]]:
    return [
        [
            s.created_at.strftime("%Y-%m-%d %H:%M"),
            _scenario(lang, s.scenario),
            ui_text(lang, f"data.languages.{s.language}"),
            ui_text(lang, f"data.status.{s.status}"),
            _recording(lang, s),
        ]
        for s in ctx.db.list_sessions()
    ]


def choices(ctx: AppContext, lang: Language) -> list[tuple[str, str]]:
    """(label, session id) pairs for the session picker."""
    return [
        (f"{s.created_at:%Y-%m-%d %H:%M} · {_scenario(lang, s.scenario)}", s.id)
        for s in ctx.db.list_sessions()
    ]


def export_session(ctx: AppContext, lang: Language, sid: str | None) -> tuple[str | None, str]:
    if not sid:
        return None, ui_text(lang, "data.choose_session")
    if ctx.store is None:
        return None, ui_text(lang, "data.export_needs_key")
    try:
        path = rights.export_session(ctx.db, ctx.store, sid)
    except media.MediaUnreadable:
        return None, ui_text(lang, "data.export_needs_key")
    return str(path), ui_text(lang, "data.export_ready")


def export_all(ctx: AppContext, lang: Language) -> tuple[str | None, str]:
    if ctx.store is None:
        return None, ui_text(lang, "data.export_needs_key")
    try:
        path = rights.export_all(ctx.db, ctx.store)
    except media.MediaUnreadable:
        return None, ui_text(lang, "data.export_needs_key")
    return str(path), ui_text(lang, "data.export_ready")


def delete_session(ctx: AppContext, lang: Language, sid: str | None) -> str:
    if not sid:
        return ui_text(lang, "data.choose_session")
    rights.delete_session(ctx.db, sid)
    return ui_text(lang, "data.deleted_one")


def delete_all(ctx: AppContext, lang: Language, *, confirmed: bool) -> str:
    if not confirmed:
        return ui_text(lang, "data.confirm_needed")
    rights.delete_everything(ctx.db)
    ctx.reset()
    return ui_text(lang, "data.deleted_all")


def build(ctx: AppContext, lang: Language) -> None:
    import gradio as gr

    rtl = lang == "ar"
    headers = [
        ui_text(lang, f"data.columns.{c}")
        for c in ("date", "scenario", "language", "status", "recording")
    ]
    gr.Markdown(f"## {ui_text(lang, 'data.heading')}", rtl=rtl)
    gr.Markdown(ui_text(lang, "data.intro"), rtl=rtl)
    table = gr.Dataframe(value=rows(ctx, lang), headers=headers, interactive=False)
    picker = gr.Dropdown(choices=choices(ctx, lang), label=ui_text(lang, "data.session"))
    with gr.Row():
        export_one = gr.Button(ui_text(lang, "data.export_session"))
        delete_one = gr.Button(ui_text(lang, "data.delete_session"), variant="stop")
    with gr.Row():
        export_every = gr.Button(ui_text(lang, "data.export_all"))
        confirm = gr.Checkbox(value=False, label=ui_text(lang, "data.confirm_delete_all"))
        delete_every = gr.Button(ui_text(lang, "data.delete_all"), variant="stop")
    status = gr.Markdown(rtl=rtl)
    download = gr.File(interactive=False)

    def refreshed(msg: str) -> tuple[str, Any, Any]:
        return (
            msg,
            gr.Dataframe(value=rows(ctx, lang)),
            gr.Dropdown(choices=choices(ctx, lang), value=None),
        )

    def on_export(sid: str | None) -> tuple[str, str | None]:
        path, msg = export_session(ctx, lang, sid)
        return msg, path

    def on_export_all() -> tuple[str, str | None]:
        path, msg = export_all(ctx, lang)
        return msg, path

    export_one.click(on_export, [picker], [status, download])
    export_every.click(on_export_all, [], [status, download])
    delete_one.click(
        lambda sid: refreshed(delete_session(ctx, lang, sid)), [picker], [status, table, picker]
    )
    delete_every.click(
        lambda ok: refreshed(delete_all(ctx, lang, confirmed=ok)),
        [confirm],
        [status, table, picker],
    )
