# FINAL 13 — Setup Guide: Everything on Your Side Before the First Claude Code Session

Goal: at the end of this guide every account exists, every key is created and stored in the right place, every tool is installed, both coding agents are configured, and `scripts/check_setup.py` reports no failures. Then one command, `/kickoff`, makes Claude Code generate the folder structure and run phase P0.

Data sources are **not** part of this guide. Nothing here downloads datasets or models.

Time needed: about 2–3 hours. Do the steps in order; each one ends with a check.

✓ = command checked against the tool's docs on 11 Nov 2026 · ~ = confirm on screen; menus move.

---

## 0. What you will have at the end

| Item | Where it lives |
|---|---|
| Private GitHub repo `presenceai` with the starter kit committed | GitHub + your disk |
| App secrets: OpenRouter key, Hugging Face read token, media encryption key | `presenceai/.env` (git-ignored) |
| Agent secrets: GitHub token, Context7 key | Your shell environment (user variables) |
| Claude Code logged in, project MCP servers connected, Serena + Hugging Face MCP at user scope, agent-skills plugin | Your machine |
| Cline in VS Code with its own OpenRouter key and MCP servers | VS Code |
| Kaggle account with phone verified and API token saved | `~/.kaggle/` |

### Secrets register (fill this in a password manager, not in a file in the repo)

| Secret | Created in step | Stored in | Used by |
|---|---|---|---|
| OpenRouter key `presenceai-app` | 3.3 | `.env` → `OPENROUTER_API_KEY` | The app |
| OpenRouter key `cline-dev` | 3.3 | Cline settings | Cline |
| Hugging Face token `presenceai-read` | 3.4 | `.env` → `HF_TOKEN` | Model downloads (later) |
| Hugging Face token `presenceai-mcp` | 3.4 | Claude Code / Cline MCP config | HF MCP server |
| GitHub fine-grained PAT `presenceai-agents` | 5.3 | Shell env → `GITHUB_PAT` | GitHub MCP server |
| Context7 key | 3.5 | Shell env → `CONTEXT7_API_KEY` | Context7 MCP server |
| Kaggle API token | 3.6 | `~/.kaggle/access_token` | Training (P4) |
| `MEDIA_KEY` (Fernet) | 6.2 | `.env` + password manager backup | Encrypting recordings |

If you lose `MEDIA_KEY`, encrypted recordings cannot be recovered. If any other key leaks, delete it on the provider's site and create a new one.

---

## 1. Choose your platform path

| Your computer | Recommended path |
|---|---|
| **Windows 10/11** | Use **WSL2 + Ubuntu** for the project (several audio/ML packages build more easily on Linux). Browser, VS Code and Unsloth Studio stay on Windows. Keep the repo inside the Linux file system (`~/projects/presenceai`), not under `/mnt/c/`. |
| Windows, native (alternative) | Works too: use PowerShell commands below. Expect occasional build issues with `llama-cpp-python` later. |
| macOS | Native, with Homebrew. |
| Linux | Native. |

Windows + WSL2 setup (PowerShell as Administrator):

```powershell
wsl --install -d Ubuntu
```

Restart, open "Ubuntu" from the Start menu, create your Linux user. From here on, **use the macOS/Linux commands inside Ubuntu**, except where a step says "on Windows". Install the VS Code extension "WSL" so VS Code opens the Linux folder.

Check: `wsl -l -v` (PowerShell) shows Ubuntu with VERSION 2.

---

## 2. Install the tools

### 2.1 Package manager basics

| Platform | Command |
|---|---|
| macOS | Install Homebrew from brew.sh if missing |
| Ubuntu / WSL | `sudo apt update && sudo apt install -y git curl unzip build-essential ffmpeg` |
| Windows native | `winget` is built in (PowerShell) |

### 2.2 Git, ffmpeg, Node.js LTS, GitHub CLI, VS Code

| Tool | Why | macOS | Ubuntu / WSL | Windows native |
|---|---|---|---|---|
| Git | Version control | `brew install git` | (installed in 2.1) | `winget install Git.Git` ~ |
| ffmpeg | Audio/video conversion for Gradio and Whisper | `brew install ffmpeg` | (installed in 2.1) | `winget install Gyan.FFmpeg` ~ |
| Node.js LTS | Only to launch `npx` MCP servers (Playwright); not used by the product | `brew install node` | Install via nvm: see nodejs.org "Download" page, choose LTS ~ | `winget install OpenJS.NodeJS.LTS` ~ |
| GitHub CLI `gh` | Create repo, auth git, PRs | `brew install gh` | Follow cli.github.com → Linux install ~ | `winget install GitHub.cli` ~ |
| VS Code | Editor for Cline | `brew install --cask visual-studio-code` | Install on **Windows**, then the "WSL" extension | `winget install Microsoft.VisualStudioCode` ~ |

Git identity (once):

```bash
git config --global user.name  "Eyad Milhem"
git config --global user.email "<the email on your GitHub account>"
git config --global init.defaultBranch main
```

### 2.3 uv and Python 3.12 ✓

```bash
# macOS / Linux / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```powershell
# Windows native
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal, then:

```bash
uv python install 3.12
uv --version
```

### 2.4 Claude Code ✓

Requires a Claude Pro, Max, Team, Enterprise or Console account; the free plan does not include Claude Code.

```bash
# macOS / Linux / WSL
curl -fsSL https://claude.ai/install.sh | bash
```
```powershell
# Windows native
irm https://claude.ai/install.ps1 | iex
```

Then:

```bash
claude --version
claude doctor
```

On Windows native, Git for Windows is recommended so Claude Code can use Git Bash.

### 2.5 Agent helper tools ✓

```bash
uv tool install -p 3.13 serena-agent
serena init
uv tool install specify-cli
```

### 2.6 VS Code extensions

Install from the Extensions panel: **Cline**, **Python** (Microsoft), **Ruff**, and on Windows+WSL also **WSL**.

**Check for step 2:** `git --version`, `uv --version`, `node --version`, `npx --version`, `gh --version`, `ffmpeg -version`, `claude --version`, `serena --help`, `specify --help` all print something.

---

## 3. Create accounts and keys

### 3.1 GitHub

1. Sign in (or create an account) at github.com and verify your email.
2. Optional but useful: apply for the **GitHub Student Developer Pack** with your university email (education.github.com).
3. `gh auth login` → GitHub.com → HTTPS → "Login with a web browser". This lets `git push` work without tokens.

The repo and the fine-grained token come in step 5, because the token is scoped to the repo.

### 3.2 Claude

Make sure your Claude plan includes Claude Code (2.4). Login happens in step 7.

### 3.3 OpenRouter (you already have an account and credits)

1. **Two separate keys** at openrouter.ai → Keys → Create key:
   - `presenceai-app` — for the app. Set a **credit limit** (e.g. $1) so a bug cannot spend your balance on paid models; free models cost $0.
   - `cline-dev` — for Cline, with its own small limit. Keeping them separate lets you see agent usage apart from app usage on the activity page.
   - If offered, set an expiry date for each key and note it in your password manager.
2. **Privacy settings** (openrouter.ai/settings/privacy): in the **free models** section, **allow** free endpoints that may log or train on prompts. Why: the more capable free models are served only by such providers, and they cost nothing (decision of 2026-11-13, `docs/adr/0003-openrouter-model-and-privacy.md`). The protections stay: the app sends text only, with personal names redacted, no media, no ids and no `user` field, and only when the participant has accepted consent switch (d); otherwise it uses templates. The app also sends `provider.data_collection: "allow"` with each request. If this setting blocks those endpoints, requests fail with "No endpoints found matching your data policy".
3. Leave your existing key alone or delete it if it was used elsewhere.

### 3.4 Hugging Face

1. Account at huggingface.co, **verify your email**. (The optional online pilot needs an account older than 30 days, so do this now even though the pilot is weeks away.)
2. Settings → Access Tokens → New token ✓:
   - `presenceai-read` — role **Read**. Goes in `.env` as `HF_TOKEN`.
   - `presenceai-mcp` — role **Read**. Used only by the HF MCP server.
   - Create the write token later (P4), when you first push a trained model.
3. While logged in, open `https://huggingface.co/google/embeddinggemma-300m` and accept the Gemma terms (needed if the app ever downloads it; you already have a GGUF copy).
4. Settings → MCP (`https://huggingface.co/settings/mcp`): choose Claude Code and keep the page open; you will paste the generated command in step 7.3.

### 3.5 Context7 ✓

1. Sign in at context7.com → Dashboard → Settings → create an API key (free; optional but gives higher limits).
2. It goes into your shell environment as `CONTEXT7_API_KEY` in step 6.

### 3.6 Kaggle (for training later; set up now because verification can take time)

1. Account at kaggle.com.
2. Settings → **Phone verification** (needed for GPU and internet in notebooks) ~.
3. Settings → API → **Generate New Token** ~. Save it:

```bash
mkdir -p ~/.kaggle
# paste the token into this file with an editor, then:
chmod 600 ~/.kaggle/access_token
```

Windows native: `C:\Users\<you>\.kaggle\access_token`. Older tools use the legacy `kaggle.json` from "Create Legacy API Key"; you do not need it now.

4. Open any notebook, Settings → Accelerator → GPU, run one cell (`!nvidia-smi`) to confirm GPU access, then stop the session.

### 3.7 Not now

Supabase, Cloudflare, Azure for Students, Google Colab: only needed in P4/P7. Skip.

---

## 4. Unpack the starter kit

```bash
mkdir -p ~/projects && cd ~/projects
unzip ~/Downloads/presenceai-starter-kit.zip      # on WSL: unzip /mnt/c/Users/<you>/Downloads/presenceai-starter-kit.zip
mv presenceai-starter presenceai
cd presenceai
ls -a     # AGENTS.md CLAUDE.md .mcp.json .env.example .gitignore .claude .clinerules docs scripts README.md
```

What the kit contains:

| File | Purpose |
|---|---|
| `AGENTS.md` | Rules for both agents (non-negotiables, way of working) |
| `CLAUDE.md` | Imports AGENTS.md for Claude Code |
| `.clinerules/00-project.md` | Points Cline at AGENTS.md |
| `.mcp.json` | Project MCP servers for Claude Code: Context7, GitHub, Playwright, Fetch; secrets come from your shell environment |
| `.claude/settings.json` | Blocks the agent from reading `.env`, credentials and study data; asks before `git push` and model downloads |
| `.claude/commands/` | `/kickoff`, `/slice`, `/verify`, `/ui-check` |
| `.env.example` | Template for `.env` |
| `.gitignore` | Keeps secrets, data, models and media out of git |
| `docs/plan/` | FINAL_00–13 |
| `scripts/check_setup.py` | Verifies this whole guide; prints no secret values |

---

## 5. Create the GitHub repository and the agent token

### 5.1 First commit and push

```bash
cd ~/projects/presenceai
git init
git add -A
git status            # confirm .env is NOT listed (it does not exist yet anyway)
git commit -m "Starter kit: agent rules, MCP config, plan, setup checker"
gh repo create presenceai --private --source=. --remote=origin --push
```

Private now, public at the end with the thesis.

### 5.2 Repository settings (github.com → presenceai → Settings)

- Security → enable **Secret scanning** and **Push protection** if offered for your plan ~.
- Branches → add a rule for `main` later, once CI exists (after `/kickoff`).

### 5.3 Fine-grained token for the GitHub MCP server

github.com → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token ~:

| Field | Value |
|---|---|
| Name | `presenceai-agents` |
| Expiration | 90 days |
| Repository access | Only select repositories → `presenceai` |
| Contents | Read and write |
| Issues | Read and write |
| Pull requests | Read and write |
| Actions | Read-only |
| Metadata | Read-only (automatic) |
| Everything else | No access |

Copy it once into your password manager; it goes into the shell environment in step 6.

---

## 6. Put the secrets in place

### 6.1 Agent secrets → shell environment

These are read by Claude Code when it expands `${GITHUB_PAT}` and `${CONTEXT7_API_KEY}` in `.mcp.json` ✓.

macOS (zsh) / Linux / WSL (bash): open `~/.zshrc` or `~/.bashrc` in an editor (not with `echo`, so the token is not saved in shell history) and add:

```bash
export GITHUB_PAT="github_pat_..."
export CONTEXT7_API_KEY="ctx7..."
```

Then `source ~/.bashrc` (or open a new terminal).

Windows native: Start → "Edit environment variables for your account" → New… for each name/value. Open a new terminal afterwards.

### 6.2 App secrets → `.env`

```bash
cd ~/projects/presenceai
cp .env.example .env
uv run --with cryptography python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Open `.env` in an editor and fill:

- `OPENROUTER_API_KEY` = the `presenceai-app` key
- `MEDIA_KEY` = the 44-character value just printed (copy it to your password manager too)
- `HF_TOKEN` = `presenceai-read`
- Leave `OPENROUTER_MODEL` empty; spike S4 chooses it.

On macOS/Linux/WSL: `chmod 600 .env`.

### 6.3 Check

```bash
uv run python scripts/check_setup.py --online
```

Every line should be `OK` except `OPENROUTER_MODEL` (warning until S4) and any recommended tool you chose to skip. The OpenRouter line also shows today's free-request count.

---

## 7. Configure Claude Code

### 7.1 Log in

```bash
cd ~/projects/presenceai
claude
```

Follow the browser login. Claude Code asks whether to trust this folder and whether to use the project's `.mcp.json` servers: approve both. Exit with `/exit`.

### 7.2 Serena (user scope) ✓

```bash
serena setup claude-code
```

If it starts slowly later, add `export MCP_TIMEOUT=60000` to your shell profile.

### 7.3 Hugging Face MCP (user scope)

Paste the command generated on `huggingface.co/settings/mcp` for Claude Code (step 3.4). Use the `presenceai-mcp` token if it asks for one.

### 7.4 Check all servers ✓

```bash
claude mcp list
```

Expected: `context7`, `github`, `playwright`, `fetch`, `serena`, `hf` (name may differ) → `✔ Connected`. Playwright may download a browser on first use. If `github` or `context7` fail, the shell variable is not visible to Claude Code: open a new terminal and retry.

### 7.5 Skills plugin ✓ (inside Claude Code)

```
/plugin marketplace add addyosmani/agent-skills
/plugin install agent-skills@addy-agent-skills
```

### 7.6 spec-kit (optional now, recommended before P1)

```bash
specify integration list          # find the key for Claude Code
specify init --help               # find the option for "current directory"
specify init <option-for-current-dir> --integration <claude-key>
```

If it proposes to overwrite `CLAUDE.md` or `AGENTS.md`, decline and keep the kit's versions. Commit what it adds.

---

## 8. Configure Cline

1. VS Code → open `~/projects/presenceai` (on Windows: "WSL: Open Folder in WSL").
2. Cline panel → Settings → **API Provider**: OpenRouter, paste the `cline-dev` key, choose a free model (ids end in `:free`). If your Cline version lists a "Claude Code" provider, you can use it instead to work under your Claude plan ~.
3. Cline panel → **MCP Servers** → Configure → paste this into `cline_mcp_settings.json` ✓ (format):

```json
{
  "mcpServers": {
    "context7":   { "type": "streamableHttp", "url": "https://mcp.context7.com/mcp",
                    "headers": { "Authorization": "Bearer YOUR_CONTEXT7_KEY" }, "disabled": false, "autoApprove": [] },
    "github":     { "type": "streamableHttp", "url": "https://api.githubcopilot.com/mcp/",
                    "headers": { "Authorization": "Bearer YOUR_GITHUB_PAT" }, "disabled": false, "autoApprove": [] },
    "serena":     { "command": "serena", "args": ["start-mcp-server", "--context", "ide", "--project", "/ABSOLUTE/PATH/TO/presenceai"],
                    "disabled": false, "autoApprove": [] },
    "playwright": { "command": "npx", "args": ["@playwright/mcp@latest"], "disabled": false, "autoApprove": [] }
  }
}
```

   This file is outside the repo, so pasting the keys here does not commit them. Replace the path with the output of `pwd` in the repo.
4. Cline reads `.clinerules/` automatically ~. Keep auto-approve **off** for file edits and commands.
5. Optional: `npx skills add addyosmani/agent-skills --skill test-driven-development` ✓.

Check: in the MCP Servers panel each server shows a green status.

---

## 9. Training tools (check only)

- **Unsloth Studio** (you already have it): open it once to confirm it starts. If your machine has no NVIDIA GPU, training will use Kaggle instead; nothing else to do now.
- **Kaggle**: done in 3.6.

---

## 10. Final check before the first session

```bash
cd ~/projects/presenceai
uv run python scripts/check_setup.py --online
claude mcp list
git status          # clean
```

All `OK`, all servers connected, working tree clean → ready.

---

## 11. First Claude Code session: generate the structure and execute P0

```bash
cd ~/projects/presenceai
claude
```

Inside Claude Code:

```
/kickoff
```

What happens:

1. Claude reads AGENTS.md and the plan files, then shows a **plan** for P0 (folder tree from FINAL_08, `pyproject.toml`, settings and schema stubs, configs, EN/AR language files, safety tests, CI, licence checker). Nothing is written yet.
2. You review it. Ask for changes or approve.
3. It creates the structure, adds only skeleton dependencies, runs ruff, pyright and pytest, and makes small commits. It does **not** push and does **not** download models or data.
4. You review the commits (`git log --stat`, or the diff in VS Code), then push:

```bash
git push
```

5. On GitHub, check that the CI workflow ran green; then add the branch rule for `main` (5.2).

Next sessions, one per spike (FINAL_12 §1):

```
/slice S4     # choose and pin the OpenRouter free model
/slice S1     # live mic + webcam streaming in Gradio/FastRTC
/slice S3     # Unsloth Whisper notebook on Kaggle
```

Start each in a fresh session. Use `/verify` before you push.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `claude mcp list` shows `github` failing | `echo ${GITHUB_PAT:+set}` prints `set`? If not, the variable is not in this terminal; reopen it. Check the token has access to the `presenceai` repo |
| `context7` failing | Same as above for `CONTEXT7_API_KEY`; or remove the header line to use it without a key (lower limits) |
| OpenRouter check returns 401 | Wrong key in `.env`, or the key was disabled/expired |
| OpenRouter free model returns "No endpoints found matching your data policy" | Your OpenRouter privacy setting still blocks free endpoints that may log or train on prompts. Allow them as described in step 3.3 (item 2), then retry |
| Hugging Face check returns 401 | Token deleted or pasted with spaces |
| `serena` not found | Open a new terminal; `uv tool update-shell` adds uv tools to PATH |
| Claude Code cannot see files on WSL | You opened `/mnt/c/...`; move the repo to `~/projects` |
| Agent tries to read `.env` | Blocked by `.claude/settings.json`; it should refer to variable names only |

## 13. Sources

- Claude Code setup: https://code.claude.com/docs/en/setup
- Claude Code MCP: https://code.claude.com/docs/en/mcp
- uv install: https://docs.astral.sh/uv/getting-started/installation/
- GitHub MCP server: https://github.com/github/github-mcp-server
- Context7: https://github.com/upstash/context7
- Hugging Face tokens: https://huggingface.co/docs/hub/security-tokens
- OpenRouter privacy: https://openrouter.ai/docs/features/privacy-and-logging
- Serena clients: https://oraios.github.io/serena/02-usage/030_clients.html
- Cline MCP: https://docs.cline.bot/mcp/configuring-mcp-servers
