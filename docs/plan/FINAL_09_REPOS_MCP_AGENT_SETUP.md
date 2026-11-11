# FINAL 09 — Starred Repos, MCP Servers, Coding-Agent Setup

✓ = checked on the repo/docs page on 11 Nov 2026 · ~ = run `--help` or check the README if it fails.

## 1. Your starred repos — final verdict

### Used in the product or training

| # | Repo | Role | Licence |
|---|---|---|---|
| 56 | duckdb/duckdb | Analytics, experiment tables, optional vector search | MIT |
| 21 | NandhaKishorM/laya | Typed yes/no/choice/score rubric detector | Apache-2.0 ✓ |
| 51 | unslothai/unsloth | T1 Whisper and T2 Coach-SLM fine-tuning; Studio for local runs | ~ |
| 37 | supabase/supabase | Postgres + storage for the optional online pilot (hosted free tier) | ~ |
| 2 | jamiepine/voicebox | Not used directly; Kokoro (which it bundles) is used for optional panel voice | MIT ✓ |

### Optional / research

| # | Repo | Role | Note |
|---|---|---|---|
| 11 | Arize-ai/phoenix | Trace LLM calls while developing T2 and prompts | Elastic License 2.0 ✓ (not OSI) |
| 53 | google-research/tabfm | Small-N baseline for "features → human rating" in E10 analysis | Weights non-commercial ✓ |
| 22 | PYPE-AI-MAIN/cue-v1 | Turn detection for future interruptible panels | Needs GPU, early-stage ✓ |
| 1 | opendatalab/MinerU | Parse open-textbook PDFs for the knowledge base if Pressbooks EPUB export is messy | Apache-2.0 + extra terms ✓ |
| 52 | reflex-dev/xy | Thesis figures | Alpha ✓ |
| 33, 8 | hyperframes, video-shotcraft | Competition demo video | — |

### Coding-agent tooling

| # | Repo | Use |
|---|---|---|
| 13 | upstash/context7 | MCP: current docs for Gradio, FastRTC, MediaPipe, faster-whisper, Unsloth |
| 12 | github/github-mcp-server | MCP: issues/PRs/CI |
| 14 | microsoft/playwright-mcp | MCP: agent clicks through the Gradio app; screenshots |
| 48 | github/spec-kit | Spec → plan → tasks → implement workflow (MIT ✓) |
| 15 | addyosmani/agent-skills | Pick 4–5 skills: TDD, incremental implementation, debugging, code review, ADRs (MIT ✓) |
| 65 | multica-ai/andrej-karpathy-skills | Merge useful rules into AGENTS.md |
| 55 | Graphify-Labs/graphify | Codebase graph once the repo is large; code parsing local and keyless ✓ |
| 42 | headroomlabs-ai/headroom | Token compression proxy; set `HEADROOM_BEACON=off` ✓ |
| 47 | ripienaar/free-for-dev | Reference list of free tiers (used for FINAL_02 §5) |

### Not used

| # | Repo | Reason |
|---|---|---|
| 38 | zauberzeug/nicegui | Replaced by Gradio + FastRTC: no browser mic/webcam recording without custom JS ✓ |
| 62 | ultralytics | AGPL-3.0 ~; MediaPipe covers pose |
| 43 | decolua/9router | Routes agents to commercial models through third parties; unclear terms |
| 29 | p-e-w/heretic | Removes model safety behaviour |
| 5 | NVIDIA/auto-ontology | Needs NVIDIA API/NIMs + Postgres ✓ |
| 32 | archestra-ai/OpenAPPA | Guards agent tool calls; the product LLM has no tools ✓ |
| 45 | google-research/timesfm | Too few sessions per user for forecasting |
| 67 | huggingface/ml-intern | Archived |
| others | — | Different problem domain |

## 2. MCP servers

### Always on

| Server | Why for this project | Checked |
|---|---|---|
| Context7 | Gradio/FastRTC/MediaPipe APIs changed in 2026; agent must read current docs | ~ |
| Serena | Symbol-level navigation and edits | ✓ |
| GitHub | Roadmap issues, PRs, CI | ~ |
| Playwright | UI checks of the Gradio pages (camera/mic can be faked with Chromium flags ~) | ~ |
| DuckDB (`mcp-server-motherduck`, local file, read-only) | Query experiment results | ✓ |

### Per phase

| Server | Phase | Checked |
|---|---|---|
| Hugging Face (`https://huggingface.co/mcp`, snippet from your HF MCP settings) | Models, datasets, Spaces, papers | ✓ |
| arXiv (`uvx arxiv-mcp-server`) | Related work | ✓ |
| Jupyter (`uvx jupyter-mcp-server@latest`) | Experiments notebooks | ✓ |
| Supabase MCP (official, read-only mode) | Stage B schema and data checks | ~ |
| Fetch (`uvx mcp-server-fetch`) | Reading public pages | ~ |

Never connect an MCP server to `data/study/` or `~/.presenceai/sessions/`.

## 3. `AGENTS.md` (shared by both agents)

```markdown
# PresenceAI — rules for coding agents
Plan: docs/plan/FINAL_00..12. Read FINAL_00 first, then only the file for the task.

## Non-negotiables
1. Python 3.12 only. No .js/.ts/.html files written by hand. UI = Gradio + FastRTC.
2. $0 stack. Allowed network calls at runtime: OpenRouter (coach/providers/openrouter.py only,
   text only, only if the user enabled AI wording) and Supabase (storage/media.py, Stage B only).
3. Never add emotion, mood, personality, honesty, intelligence, confidence or hireability
   inference. vision/expressions.py may emit only: smile_visible, brows_raised, brows_lowered,
   eyes_closed, mouth_moving_silent. No face recognition. Expressions never feed scores.
4. Every feedback statement carries evidence (span or timestamp) and a confidence.
   LLM claims without a verified quote are dropped.
5. Raw media is encrypted at write time. Never log, print or upload raw media or the media key.
6. No GPL/AGPL runtime dependency (run scripts/check_licenses.py).
7. Thresholds, model ids and targets live in configs/. You may propose values; the human decides
   and records an ADR. Never tune a threshold to make a test pass.
8. Never fabricate data, metrics or results. Experiments write results; you may summarise them.

## How to work
- One vertical slice per session. Test first. `uv run pytest -m "not slow"` before "done".
- Check current library APIs with Context7 before using Gradio, FastRTC, MediaPipe, Unsloth.
- Shared types only in src/presenceai/schemas/.
- English and Arabic strings live in src/presenceai/lang/; no hard-coded UI text.
- Ask when a requirement is ambiguous.

## Wording for any user-facing text
Observable + number + location. "Speech rate rose from 148 to 191 wpm (01:12–01:40)."
Never: "You seemed nervous / confident / happy."
```

## 4. Claude Code

`.mcp.json` (commit; secrets via env):

```json
{
  "mcpServers": {
    "context7":   { "command": "npx", "args": ["-y", "@upstash/context7-mcp"] },
    "playwright": { "command": "npx", "args": ["@playwright/mcp@latest"] },
    "github":     { "type": "http", "url": "https://api.githubcopilot.com/mcp/",
                    "headers": { "Authorization": "Bearer ${GITHUB_PAT}" } },
    "duckdb":     { "command": "uvx", "args": ["mcp-server-motherduck", "--db-path", "/ABS/PATH/experiments/results/analytics.duckdb"] },
    "arxiv":      { "type": "stdio", "command": "uvx", "args": ["arxiv-mcp-server"] },
    "jupyter":    { "command": "uvx", "args": ["jupyter-mcp-server@latest"],
                    "env": { "JUPYTER_URL": "http://localhost:8888", "JUPYTER_TOKEN": "${JUPYTER_TOKEN}", "ALLOW_IMG_OUTPUT": "true" } },
    "fetch":      { "command": "uvx", "args": ["mcp-server-fetch"] }
  }
}
```

Per machine:

```bash
uv tool install -p 3.13 serena-agent && serena init && serena setup claude-code      # ✓
/plugin marketplace add addyosmani/agent-skills                                       # ✓ inside Claude Code
/plugin install agent-skills@addy-agent-skills                                        # ✓
uv tool install specify-cli && specify init . --integration <key>                      # ✓ CLI, ~ key (see specify init --help)
```

Hugging Face MCP: paste the snippet generated at huggingface.co/settings/mcp. ✓

Slash commands in `.claude/commands/`: `/slice <name>` (read FINAL_12, list exit criteria, propose files, wait), `/verify` (ruff, pyright, tests, safety tests, licence check), `/ui-check` (Playwright walk through consent → practice with fixture upload → report → progress), `/experiment <n>` (run, then summarise via DuckDB; no code edits).

## 5. Cline

Cline panel → MCP Servers → Configure → `cline_mcp_settings.json` ✓:

```json
{
  "mcpServers": {
    "context7":   { "command": "npx", "args": ["-y", "@upstash/context7-mcp"], "disabled": false, "autoApprove": [] },
    "serena":     { "command": "serena", "args": ["start-mcp-server", "--context", "ide", "--project", "/ABS/PATH/presenceai"], "disabled": false, "autoApprove": [] },
    "playwright": { "command": "npx", "args": ["@playwright/mcp@latest"], "disabled": false, "autoApprove": [] },
    "github":     { "type": "streamableHttp", "url": "https://api.githubcopilot.com/mcp/",
                    "headers": { "Authorization": "Bearer YOUR_FINE_GRAINED_PAT" }, "disabled": false, "autoApprove": [] },
    "duckdb":     { "command": "uvx", "args": ["mcp-server-motherduck", "--db-path", "/ABS/PATH/experiments/results/analytics.duckdb"], "disabled": false, "autoApprove": [] },
    "arxiv":      { "command": "uvx", "args": ["arxiv-mcp-server"], "disabled": true, "autoApprove": [] },
    "jupyter":    { "command": "uvx", "args": ["jupyter-mcp-server@latest"],
                    "env": { "JUPYTER_URL": "http://localhost:8888", "JUPYTER_TOKEN": "YOUR_TOKEN", "ALLOW_IMG_OUTPUT": "true" },
                    "disabled": true, "autoApprove": [] }
  }
}
```

Skills: `npx skills add addyosmani/agent-skills --skill test-driven-development` ✓. If Cline uses OpenRouter free models, give it its own key so its usage is visible separately from the app's.

## 6. Enable by phase

| Phase (FINAL_12) | Servers |
|---|---|
| P0 setup, spikes | Context7, GitHub, Hugging Face |
| P1–P3 speech, text, rubric, UI | Context7, Serena, GitHub, Playwright |
| P4 training | Hugging Face, Jupyter |
| P5 engine, simulator | Serena, DuckDB, Jupyter |
| P6 vision, live loop | Context7, Playwright |
| P7 online pilot | Supabase (read-only), Hugging Face |
| P8–P9 evaluation, writing | DuckDB, Jupyter, arXiv, Fetch |
