# FINAL 02 — Architecture and Deployment

## 1. The hybrid split (decided)

Rule used: put a job in the cloud only if (a) the cloud does it for $0 and (b) the laptop cannot do it well. Everything else runs where the app runs.

| Job | Where | Why |
|---|---|---|
| Browser UI, mic and webcam capture | User's browser (Gradio components, no hand-written JS) | Required for a web UI |
| Live loop: meters, gaze zone, framing, live prompts | Same machine as the app server (your laptop in Stage A) | Needs < 300 ms latency; streaming video to a remote free tier is fragile |
| ASR, VAD, audio features | App server (laptop CPU; ZeroGPU in Stage B) | Whisper `small`/`turbo` int8 runs on a laptop CPU; free cloud CPU hosting for Gradio no longer exists on HF |
| Text analysis, Laya, embeddings, rubric, engine | App server | Light enough for CPU |
| Language model (follow-ups, wording, rubric judge) | **Cloud: OpenRouter free models** | Your decision; no local memory use; 1,000 requests/day is enough |
| Coach-SLM (fine-tuned small model, T2) | App server, GGUF on CPU | Research comparison and offline fallback |
| Model training (T1, T2, T3) | **Cloud: Kaggle / Colab free GPUs** (or Unsloth Studio locally if you have an NVIDIA GPU) | No local GPU assumed |
| Model and dataset hosting | **Cloud: Hugging Face Hub** (free) | Versioned, citable, private repos allowed |
| Online access for invited users (optional) | **Cloud: Hugging Face ZeroGPU Gradio Space** | Only free compute-capable Gradio host on HF for free accounts |
| Pilot database and encrypted media (optional) | **Cloud: Supabase free** (Postgres 500 MB, Storage 1 GB) | Needed only when the app runs in the Space |
| Code, CI | **Cloud: GitHub + Actions** | Free for public repos |

## 2. Component diagram

```
 Browser (Gradio page, EN/AR)                      App server (Python 3.12)
 ┌──────────────────────────────┐   WebRTC     ┌──────────────────────────────────────────────┐
 │ FastRTC WebRTC component     │─────────────▶│ live/  frame + audio handlers                 │
 │  mic + webcam stream         │◀─────────────│   meters, gaze zone, framing, live prompts    │
 │ meters, prompt banner        │ AdditionalOut│   writes encrypted chunks + landmarks          │
 │ report, timeline, dashboard  │   HTTP       ├──────────────────────────────────────────────┤
 └──────────────────────────────┘◀────────────▶│ pipeline/ orchestrator (after recording)       │
                                               │  speech → text → vision → rubric → engine     │
                                               │  → coach (LLM via provider) → report          │
                                               ├──────────────────────────────────────────────┤
                                               │ storage/  SQLite or Postgres (SQLModel)        │
                                               │           Parquet features, DuckDB analytics   │
                                               │           Fernet-encrypted media, retention job│
                                               └───────────┬───────────────┬──────────────────┘
                                                           │ HTTPS (text)  │ HTTPS (optional)
                                             OpenRouter free models   Supabase (Stage B only)
```

## 3. The two loops

### 3.1 Live loop (while speaking)

| Step | Detail |
|---|---|
| Transport | FastRTC `WebRTC` component in a `gr.Blocks` page, audio + video send to server ~ (confirm the combined audio-video mode in spike S1; fallback is two Gradio streaming components: `gr.Audio(streaming=True)` and `gr.Image(sources=["webcam"], streaming=True)`) |
| Video | Downscale to 640×360, process every 2nd–3rd frame (≈ 10 fps) with MediaPipe Pose + Face Landmarker in VIDEO mode |
| Audio | 20 ms frames → RMS level, voicing (Silero VAD), syllable-nucleus rate over a 5 s window (no ASR needed, works in Arabic too) |
| Outputs | Level meter, pace meter vs the user's own target band, pause timer, gaze-zone light, framing guide, time remaining |
| Live prompts | Rule-based; at most one prompt every 20 s; shown for 3 s; only the categories the engine activated for this exercise (`FINAL_05` §6). Wording is observable: "Pace above your target", "Face partly out of frame", "Looking away from camera for 8 s" |
| Recording | Audio written to an encrypted WAV stream; video frames optionally to an encrypted low-res MP4; landmarks to Parquet |
| Budget | Target ≤ 300 ms from event to on-screen prompt, ≥ 8 fps landmarks on the laptop (measured in E8) |

### 3.2 Analysis loop (after "Stop")

`speech.analyze` (faster-whisper + VAD, word timestamps) → `text.analyze` (structure, supporting material, language) → `vision.analyze` (full-resolution pass over stored landmarks) → `rubric.score` (rules + Laya + LLM judge, fused with confidence) → `engine.update` and `engine.next` → `coach.write` (OpenRouter, Coach-SLM or templates) → report. Target under 60 s for a 3-minute talk on the laptop profile.

## 4. Deployment stages (zero-cost order)

| Stage | When | What | Cost |
|---|---|---|---|
| **A. Local web app** | Weeks 1–18 (always available) | `uv run presenceai` starts Gradio on `http://127.0.0.1:7860`; participants come in person and use your laptop | $0 |
| **A′. Local network demo** | Demo day | Same app on your laptop; if a second device must connect, use the LAN address with HTTPS via Gradio's own options ~ | $0 |
| **B. Invited online pilot (optional)** | Weeks 13–16 | Private Gradio Space on ZeroGPU: CPU work (live loop, text, engine) on the Space's CPU, Whisper inside `@spaces.GPU`; WebRTC relayed through Cloudflare TURN using your HF token (10 GB/month free); Supabase free for pilot data | $0; GPU time is charged to each visitor's own daily ZeroGPU quota |
| C. Public deployment | Future work | Not in scope | — |

Stage B constraints you must design for:

- ZeroGPU requires a Gradio Space and an account older than 30 days with a verified email; at most 2 such Spaces on a free account. ✓
- Each visitor has only a few GPU-minutes per day; one 3-minute analysis with `turbo` should use well under a minute ~. Measure in spike S2; if too tight, run Whisper `small` on the Space CPU instead.
- Space disk is not persistent, so pilot data must go to Supabase immediately; the Space keeps nothing between restarts.
- Whether a free account may keep a ZeroGPU Space private is not stated in the docs I read ~; if not, protect it with an invitation code and keep it unlisted.

## 5. Fallback hosts (if Stage B is blocked)

| Option | What you get | Catch | Status |
|---|---|---|---|
| Azure for Students | $100 credit for 12 months, renewable while a student, no card | Credit, not free forever; App Service/Container Apps would consume it | ✓ |
| Oracle Always Free | Arm VM, now 2 OCPU / 12 GB | Card needed for identity verification; capacity can be scarce | ✓ |
| Lightning AI / Modal free credits | Small monthly GPU credit | Credit-based; terms change | ~ |
| Run Stage A only | Everything works | No remote participants | — |

Not used: Render/Streamlit Cloud free tiers (≈ 0.5–1 GB RAM, too small for Whisper and MediaPipe) ~.

## 6. Configuration-driven runtime

One codebase, three runtime profiles in `configs/runtime/`:

| Setting | `local` (default) | `space` | `offline` |
|---|---|---|---|
| DB | SQLite | Supabase Postgres | SQLite |
| Media store | Local encrypted folder | Supabase Storage (encrypted before upload) | Local encrypted folder |
| ASR device | CPU int8 | ZeroGPU (`@spaces.GPU`) | CPU int8 |
| LLM provider | OpenRouter → Coach-SLM → templates | OpenRouter → templates | Coach-SLM → templates |
| TURN | none | Cloudflare via `HF_TOKEN` | none |

Secrets (`OPENROUTER_API_KEY`, `HF_TOKEN`, `SUPABASE_URL`, `SUPABASE_KEY`, `MEDIA_KEY`) come from `.env` locally and from Space secrets in Stage B; never from config files.

## 7. Storage layout

| Data | Local profile | Space profile | Retention |
|---|---|---|---|
| Profile, consent, sessions, assessments, skill state, decisions, disputes, ratings | `app.sqlite` | Supabase Postgres, pseudonymous ids only | Until user deletes |
| Word, pause, rate, landmark tables | `sessions/<id>/features/*.parquet` | Postgres JSONB or Parquet in Storage | Until user deletes |
| Raw audio/video | `sessions/<id>/media/*.enc` (Fernet) | Supabase Storage, encrypted client-side before upload, ≤ 50 MB per file | 30 days default; study: until rating ends, max 60 days |
| LLM calls | `llm_calls.jsonl` + cache | Postgres table | Same as session |

A retention job runs at every app start (and daily via GitHub Actions cron in Stage B) and deletes expired media.
