# PresenceAI — rules for coding agents (Claude Code and Cline)

The plan lives in `docs/plan/FINAL_00` … `FINAL_12`. Read `FINAL_00_README_DECISIONS.md` first, then only the file(s) relevant to the current task. If this file and a plan file disagree, stop and ask.

## Non-negotiables
1. Python 3.12 only. Do not hand-write .js/.ts/.html/.css files. UI = Gradio + FastRTC.
2. $0 stack. Allowed network calls at runtime: OpenRouter (only from `src/presenceai/coach/providers/openrouter.py`, text only, only when the user enabled AI wording) and Supabase (only from `src/presenceai/storage/media.py`, Stage B only). Model downloads happen only in `scripts/download_models.py`.
3. Never add emotion, mood, personality, honesty, intelligence, confidence or hireability inference. `vision/expressions.py` may emit only: `smile_visible`, `brows_raised`, `brows_lowered`, `eyes_closed`, `mouth_moving_silent`. No face recognition. Expression events never feed scores, the skill profile or LLM prompts.
4. Every feedback statement carries evidence (transcript span and/or timestamp) and a confidence. LLM claims whose quote is not found in the transcript are dropped.
5. Raw media is encrypted at write time (Fernet, key from `MEDIA_KEY`). Never log, print or upload raw media or any key.
6. No GPL/AGPL runtime dependency. Run `uv run python scripts/check_licenses.py` after adding a dependency.
7. Thresholds, model ids and targets live in `configs/`. You may propose values; the human decides and an ADR in `docs/adr/` records it. Never tune a threshold to make a test pass.
8. Never fabricate data, metrics, citations or results. Experiments write results; you may summarise them.
9. Never read, print or modify `.env`, `~/.presenceai/`, `data/study/` or any credential file. Refer to secrets only by variable name.

## How to work
- One vertical slice per session (see `docs/plan/FINAL_12_ROADMAP_AND_THESIS.md`). Propose a plan, wait for approval, then implement.
- Test first. `uv run pytest -m "not slow"` must pass before you say a task is done.
- `uv run ruff check --fix . && uv run ruff format .` before committing. Small commits with clear messages.
- Use `uv add <pkg>` for dependencies; never edit versions by hand.
- Before using Gradio, FastRTC, MediaPipe, faster-whisper, Unsloth or Laya APIs, check current docs with the Context7 MCP server.
- Shared data types only in `src/presenceai/schemas/`. No ad-hoc dicts passed between modules.
- `ui/` and `live/` may import any package; no package imports `ui/`.
- English and Arabic UI strings live in `src/presenceai/lang/`; no hard-coded user-facing text.
- Tests must run without models, camera, microphone or network (use fakes in `tests/fixtures/`).
- When a requirement is ambiguous, ask instead of guessing.

## Wording for any user-facing text
Observable behaviour + number + location.
Good: "Speech rate rose from 148 to 191 wpm during the follow-up (01:12–01:40)."
Never: "You seemed nervous / confident / happy."

## Commands
```
uv sync
uv run presenceai                      # app at http://127.0.0.1:7860 (once it exists)
uv run pytest -m "not slow"
uv run python scripts/check_setup.py   # environment check, prints no secret values
```
