---
description: Phase P0 — scaffold the PresenceAI repository from the plan, then verify it
---

You are starting phase P0 of PresenceAI. Read `AGENTS.md`, then `docs/plan/FINAL_00_README_DECISIONS.md`, `docs/plan/FINAL_08_FOLDER_STRUCTURE.md`, `docs/plan/FINAL_03_TECH_STACK.md` and section 1–2 of `docs/plan/FINAL_12_ROADMAP_AND_THESIS.md`.

Goal of this session: a clean, tested skeleton that matches FINAL_08. No product features, no model or dataset downloads, no spikes yet.

Work in this order, and show me a plan first (plan mode). Wait for my approval before writing files.

1. **Project metadata.** `uv init --package --name presenceai --python 3.12` if `pyproject.toml` does not exist; set licence Apache-2.0; add the `[project.scripts] presenceai = "presenceai.cli:app"` entry; write `.python-version`.
2. **Folder tree.** Create every directory in FINAL_08 §1. Python packages get an `__init__.py` with a one-line module docstring stating the package's job from the plan. Non-code folders (`data/`, `models/`, `docs/adr/`, `docs/cards/`, `docs/study/`, `docs/thesis/`, `resources/`, `scenarios/_library/`, `exercises/`, `prompts/`, `training/*`, `deploy/*`, `experiments/configs`, `experiments/results`) get a short `README.md` saying what goes there and which FINAL file defines it. `data/README.md` is the dataset register header only (no data).
3. **Dependencies.** Add only what the skeleton needs now with `uv add`: `pydantic`, `pydantic-settings`, `typer`, `pyyaml`. Dev group: `uv add --dev ruff pyright pytest pytest-cov hypothesis pre-commit`. Do not add Gradio, Whisper, MediaPipe or ML packages yet — they come with their slices.
4. **Settings and schemas stubs.** `src/presenceai/settings.py` (pydantic-settings reading `.env` and `configs/default.yaml`, secrets as `SecretStr`, never printed). `src/presenceai/schemas/assessment.py` with `Evidence`, `Measurement` (value, unit, confidence, method, evidence, language, live) exactly as FINAL_04 describes. `src/presenceai/cli.py` with a Typer app and a `version` command.
5. **Config files.** `configs/default.yaml`, `configs/engine.yaml`, `configs/live.yaml`, `configs/models.yaml`, `configs/profiles/{lite,standard,space}.yaml`, `configs/runtime/{local,space,offline}.yaml` with the keys named in the plan and `null` or TODO values where a human decision is pending. Add `docs/adr/0000-template.md` and `docs/adr/0001-record-architecture-decisions.md`.
6. **Language packs.** `src/presenceai/lang/en/ui.yaml` and `src/presenceai/lang/ar/ui.yaml` with the same keys (app title, consent headings) and a test that both files have identical keys.
7. **Safety tests first.** In `tests/safety/`: (a) the five allowed expression labels constant exists and nothing else is allowed; (b) a banned-inference term list exists in EN and AR and a helper `contains_banned_term()` passes basic cases; (c) a test that `settings` never includes secret values in `repr()`.
8. **Tooling.** Ruff + Pyright config in `pyproject.toml`; pytest markers `slow`, `needs_models`, `needs_network`; `.pre-commit-config.yaml` (ruff, ruff-format, end-of-file, detect-private-key); `.github/workflows/ci.yml` running ruff, pyright and `pytest -m "not slow and not needs_models and not needs_network"` on Python 3.12.
9. **Scripts.** `scripts/check_licenses.py` (reads installed distributions' licence metadata and fails on GPL/AGPL in the runtime group). Keep the existing `scripts/check_setup.py` unchanged.
10. **Verify.** Run `uv sync`, `uv run ruff check .`, `uv run pyright`, `uv run pytest -m "not slow"`, `uv run presenceai version`, `uv run python scripts/check_licenses.py`. Fix failures.
11. **Commit.** One commit per numbered step above or a small group of steps, with clear messages. Do not push; I will review and push.

At the end, give me: the tree (two levels), test results, anything you could not do and why, and the proposed next slice (spike S1 or S4 from FINAL_12).
