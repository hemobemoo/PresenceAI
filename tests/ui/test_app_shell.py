"""The app shell (P3-c): strings, consent gating, My data, the served pages. No browser, no models.

Handlers are plain functions (`ui/pages/*.py`) tested directly; the Gradio pages are built and
served through FastAPI's test client to check what the browser would load.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from presenceai.lang import LANGUAGES, flatten, load_ui_strings, placeholders, ui_text
from presenceai.schemas.storage import Consent
from presenceai.ui.context import AppContext
from presenceai.ui.pages import consent, data

T0 = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def ctx(tmp_path: Path):
    c = AppContext.open(
        tmp_path / "home", SecretStr(Fernet.generate_key().decode()), raw_media_days=30, now=T0
    )
    yield c
    c.close()


@pytest.fixture
def keyless(tmp_path: Path):
    c = AppContext.open(tmp_path / "home2", None, raw_media_days=30, now=T0)
    yield c
    c.close()


# --- strings ----------------------------------------------------------------------------------


def test_ui_strings_match_in_both_languages_and_have_no_study_switches():
    en, ar = (flatten(load_ui_strings(lang)) for lang in LANGUAGES)
    assert set(en) == set(ar)
    for key in en:
        assert placeholders(en[key]) == placeholders(ar[key]), key
    assert not any("study" in k for k in en)  # not applicable: no participant study (ADR 0010)
    assert re.search(r"[؀-ۿ]", ar["about.body"])
    assert ui_text("en", "consent.keep_note", days=30).startswith("Recordings are kept")


# --- consent gates practice -------------------------------------------------------------------


def test_practice_is_locked_until_use_app_is_saved(ctx: AppContext):
    assert ctx.consent() == Consent()
    assert not consent.practice_open(ctx)
    msg, is_open = consent.save(ctx, "en", Consent(camera=True), now=T0)
    assert not is_open and "stays off" in msg
    msg, is_open = consent.save(ctx, "en", Consent(use_app=True, hosted_ai=True), now=T0)
    assert is_open and consent.practice_open(ctx)
    assert ctx.consent().hosted_ai  # saved; nothing uses it before P7


# --- My data ----------------------------------------------------------------------------------


def _sessions(ctx: AppContext) -> tuple[str, str]:
    kept = ctx.db.new_session("s1_presentation", "en", "app", now=T0, keep_days=30)
    later = T0 + timedelta(minutes=1)
    unkept = ctx.db.new_session("s2_project_defense", "ar", "upload", now=later, keep_days=None)
    assert ctx.store is not None
    ctx.store.write_bytes(kept.id, "talk.wav", b"x")
    ctx.store.write_assessment(kept.id, b"{}")
    return kept.id, unkept.id


def test_sessions_table_in_the_chosen_language(ctx: AppContext):
    assert data.rows(ctx, "en") == []
    kept, unkept = _sessions(ctx)
    rows = data.rows(ctx, "en")
    assert rows[0][1:] == ["Short presentation", "English", "recorded", "kept until 2026-12-24"]
    assert rows[1][1:] == ["Project defense", "Arabic", "recorded", "deleted after analysis"]
    assert [v for _, v in data.choices(ctx, "en")] == [kept, unkept]
    ar = data.rows(ctx, "ar")
    assert ar[0][1] == "عرض قصير"


def test_export_and_delete_one_session(ctx: AppContext):
    kept, _ = _sessions(ctx)
    path, msg = data.export_session(ctx, "en", kept)
    assert path is not None and Path(path).exists() and "ready" in msg
    _, msg = data.export_session(ctx, "en", None)
    assert msg == ui_text("en", "data.choose_session")
    msg = data.delete_session(ctx, "en", kept)
    assert msg == ui_text("en", "data.deleted_one")
    assert [v for _, v in data.choices(ctx, "en")] != [] and kept not in dict(
        (v, k) for k, v in data.choices(ctx, "en")
    )
    assert not ctx.layout.session_dir(kept).exists()


def test_delete_everything_needs_the_box_ticked(ctx: AppContext):
    _sessions(ctx)
    consent.save(ctx, "en", Consent(use_app=True), now=T0)
    assert data.delete_all(ctx, "en", confirmed=False) == ui_text("en", "data.confirm_needed")
    assert len(data.rows(ctx, "en")) == 2
    assert data.delete_all(ctx, "en", confirmed=True) == ui_text("en", "data.deleted_all")
    assert data.rows(ctx, "en") == [] and ctx.consent() == Consent()  # starts again, all off
    assert not consent.practice_open(ctx)


def test_without_the_key_export_says_so_and_delete_still_works(keyless: AppContext):
    assert keyless.store is None and not keyless.key_ok
    s = keyless.db.new_session("s1_presentation", "en", "app", now=T0, keep_days=None)
    path, msg = data.export_all(keyless, "en")
    assert path is None and msg == ui_text("en", "data.export_needs_key")
    assert data.delete_session(keyless, "en", s.id) == ui_text("en", "data.deleted_one")


def test_retention_runs_when_the_app_opens(tmp_path: Path):
    key = SecretStr(Fernet.generate_key().decode())
    c = AppContext.open(tmp_path / "h", key, raw_media_days=30, now=T0)
    s = c.db.new_session("s1_presentation", "en", "app", now=T0, keep_days=None)
    assert c.store is not None
    c.store.write_bytes(s.id, "talk.wav", b"x")
    c.close()
    c = AppContext.open(tmp_path / "h", key, raw_media_days=30, now=T0 + timedelta(hours=1))
    assert c.store is not None and c.store.media_names(s.id) == ()  # unkept: deleted at start
    c.close()


# --- the served pages -------------------------------------------------------------------------


@pytest.fixture
def client(ctx: AppContext) -> Any:
    from fastapi.testclient import TestClient

    from presenceai.ui.app import create_app

    return TestClient(create_app(ctx))


def test_english_and_arabic_pages_are_served(client: Any):
    en = client.get("/")
    ar = client.get("/ar/")
    assert en.status_code == 200 and ar.status_code == 200
    assert "PresenceAI" in en.text


def test_blocks_have_analytics_off_and_system_fonts(ctx: AppContext):
    from presenceai.ui.app import build_blocks

    for lang in LANGUAGES:
        blocks = build_blocks(ctx, lang)
        assert blocks.analytics_enabled is False
        css = blocks.theme._get_theme_css()
        assert "fonts.googleapis" not in css and "@import" not in css
        assert "@font-face" not in css and "/static/fonts/" not in css  # no font files at all


def test_presenceai_with_no_command_starts_the_app(monkeypatch: pytest.MonkeyPatch):
    from typer.testing import CliRunner

    from presenceai import cli
    from presenceai.ui import app as ui_app

    started: list[bool] = []
    monkeypatch.setattr(ui_app, "main", lambda: started.append(True))
    result = CliRunner().invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert started == [True]
    started.clear()
    assert CliRunner().invoke(cli.app, ["version"]).exit_code == 0
    assert started == []  # a command never starts the app
