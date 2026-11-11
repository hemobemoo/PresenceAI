# PresenceAI — Cline rules

Follow `AGENTS.md` in the repository root exactly. It is the single source of rules for all coding agents on this project.

The five rules that matter most:
1. Python 3.12 only; UI is Gradio + FastRTC; no hand-written JS/TS/HTML.
2. $0 stack; the only runtime network calls are OpenRouter (text only) and Supabase (Stage B), from the two files named in AGENTS.md.
3. No emotion, personality or "confidence" inference; expression labels limited to the five listed in AGENTS.md.
4. Every feedback statement needs evidence and a confidence.
5. Never read or print `.env`, credentials, `data/study/` or `~/.presenceai/`.

Plan files: `docs/plan/FINAL_00` … `FINAL_12`.
