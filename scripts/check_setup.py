"""PresenceAI setup checker.

Run from the repository root:
    uv run python scripts/check_setup.py            # local checks only
    uv run python scripts/check_setup.py --online   # also tests API keys against their services

Uses only the Python standard library. Never prints secret values: only whether
each one is present and whether the service accepted it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def record(status: str, item: str, detail: str = "") -> None:
    results.append((status, item, detail))


def tool_version(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=30)
        text = (out.stdout or out.stderr).strip().splitlines()
        return text[0] if text else "installed"
    except (OSError, subprocess.SubprocessError):
        return "installed (version check failed)"


def check_tools() -> None:
    required = {
        "git": ["git", "--version"],
        "uv": ["uv", "--version"],
        "node": ["node", "--version"],
        "npx": ["npx", "--version"],
        "claude": ["claude", "--version"],
        "ffmpeg": ["ffmpeg", "-version"],
    }
    recommended = {
        "gh": ["gh", "--version"],
        "serena": ["serena", "--help"],
        "specify": ["specify", "--help"],
        "code (VS Code)": ["code", "--version"],
    }
    for name, cmd in required.items():
        v = tool_version(cmd)
        record(OK if v else FAIL, f"tool: {name}", v or "not found on PATH")
    for name, cmd in recommended.items():
        v = tool_version(cmd)
        record(OK if v else WARN, f"tool: {name}", (v or "not found on PATH")[:60])

    py = sys.version_info
    record(OK if (py.major, py.minor) == (3, 12) else WARN, "python running this script",
           f"{py.major}.{py.minor}.{py.micro} (project targets 3.12; uv will provide it)")
    record(OK, "platform", f"{platform.system()} {platform.release()} {platform.machine()}")


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def check_files_and_env() -> dict[str, str]:
    for rel in ["AGENTS.md", "CLAUDE.md", ".mcp.json", ".gitignore", ".env.example",
                ".clinerules/00-project.md", ".claude/settings.json",
                "docs/plan/FINAL_00_README_DECISIONS.md"]:
        p = ROOT / rel
        record(OK if p.exists() else FAIL, f"file: {rel}", "" if p.exists() else "missing")

    gi = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").exists() else ""
    record(OK if "\n.env\n" in f"\n{gi}\n" else FAIL, ".env is git-ignored")

    env = load_dotenv(ROOT / ".env")
    record(OK if (ROOT / ".env").exists() else FAIL, "file: .env", "" if env else "missing or empty")
    for key in ["OPENROUTER_API_KEY", "MEDIA_KEY", "HF_TOKEN"]:
        record(OK if env.get(key) else FAIL, f".env: {key}", "set" if env.get(key) else "empty")
    for key in ["OPENROUTER_MODEL"]:
        record(OK if env.get(key) else WARN, f".env: {key}",
               "set" if env.get(key) else "empty (chosen in spike S4)")

    mk = env.get("MEDIA_KEY", "")
    if mk:
        record(OK if len(mk) == 44 and mk.endswith("=") else FAIL, "MEDIA_KEY format",
               "looks like a Fernet key" if len(mk) == 44 else "not a Fernet key (44 chars)")

    for key in ["GITHUB_PAT", "CONTEXT7_API_KEY"]:
        present = bool(os.environ.get(key))
        record(OK if present else FAIL, f"shell env: {key}",
               "set (used by .mcp.json)" if present else "not set in this shell")
    return env


def http_get(url: str, token: str, extra: dict[str, str] | None = None) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "presenceai-check"}
    headers.update(extra or {})
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        return e.code, {}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return 0, {}


def check_online(env: dict[str, str]) -> None:
    if env.get("OPENROUTER_API_KEY"):
        code, body = http_get("https://openrouter.ai/api/v1/key", env["OPENROUTER_API_KEY"])
        data = body.get("data", {}) if isinstance(body, dict) else {}
        detail = f"HTTP {code}"
        free = data.get("free_model_daily_requests") if isinstance(data, dict) else None
        if isinstance(free, dict):
            detail += f"; free requests today: used {free.get('used')} of {free.get('limit')}"
        record(OK if code == 200 else FAIL, "OpenRouter key accepted", detail)

    if env.get("HF_TOKEN"):
        code, body = http_get("https://huggingface.co/api/whoami-v2", env["HF_TOKEN"])
        name = body.get("name") if isinstance(body, dict) else None
        record(OK if code == 200 else FAIL, "Hugging Face token accepted",
               f"HTTP {code}" + (f"; user {name}" if name else ""))

    pat = os.environ.get("GITHUB_PAT")
    if pat:
        code, body = http_get("https://api.github.com/user", pat,
                              {"Accept": "application/vnd.github+json"})
        login = body.get("login") if isinstance(body, dict) else None
        record(OK if code == 200 else FAIL, "GitHub PAT accepted",
               f"HTTP {code}" + (f"; user {login}" if login else ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true", help="test keys against the services")
    args = parser.parse_args()

    check_tools()
    env = check_files_and_env()
    if args.online:
        check_online(env)

    width = max(len(item) for _, item, _ in results)
    for status, item, detail in results:
        print(f"[{status}] {item.ljust(width)}  {detail}")
    fails = sum(1 for s, _, _ in results if s == FAIL)
    warns = sum(1 for s, _, _ in results if s == WARN)
    print(f"\n{fails} failing, {warns} warnings.")
    print("Ready for /kickoff." if fails == 0 else "Fix the FAIL lines before starting Claude Code.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
