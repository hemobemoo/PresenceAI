# FINAL 08 — Folder Structure

```
presenceai/
├── README.md  LICENSE (Apache-2.0)  NOTICE  CITATION.cff
├── AGENTS.md                      # rules for Cline and Claude Code (FINAL_09 §3)
├── CLAUDE.md                      # "@AGENTS.md"
├── .clinerules/00-project.md      # points to AGENTS.md
├── .mcp.json                      # Claude Code project MCP servers (no secrets)
├── .claude/commands/              # /slice /verify /ui-check /experiment
├── .specify/                      # spec-kit constitution, specs, plans, tasks
├── pyproject.toml  uv.lock  .python-version (3.12)
├── .env.example                   # OPENROUTER_API_KEY HF_TOKEN SUPABASE_URL SUPABASE_KEY MEDIA_KEY
├── .gitignore                     # .env data/ models/ *.wav *.mp4 *.enc *.sqlite *.duckdb
├── .github/workflows/
│   ├── ci.yml                     # ruff, pyright, fast tests, licence check
│   └── retention.yml              # daily cron for Stage B media deletion
│
├── configs/
│   ├── default.yaml               # thresholds, feature switches
│   ├── models.yaml                # model ids, digests, OpenRouter pinned + fallbacks
│   ├── engine.yaml                # q, r0, beta, sigma_max, targets, promotion rules
│   ├── live.yaml                  # fps, windows, prompt rate limit
│   ├── profiles/{lite,standard,space}.yaml
│   └── runtime/{local,space,offline}.yaml
│
├── scenarios/
│   ├── s1_presentation.yaml       # primary, VALUE-adapted rubric
│   ├── s2_project_defense.yaml    # secondary, with follow-ups
│   └── _library/                  # stubs: interview, criticism, disagreement, bad_news, ...
├── exercises/                     # one YAML per skill, EN + AR prompts, sources
├── prompts/                       # LLM prompt templates (.md), versioned
│
├── src/presenceai/
│   ├── __init__.py  __main__.py  cli.py  settings.py
│   ├── schemas/                   # media, features, assessment, engine, study, live
│   ├── live/                      # real-time loop
│   │   ├── stream.py              # FastRTC handlers (audio+video)
│   │   ├── audio_meters.py        # RMS, VAD, syllable-rate
│   │   ├── vision_meters.py       # framing, head pose, gaze zone
│   │   ├── prompts.py             # rule engine + rate limiter
│   │   └── recorder.py            # encrypted chunk writer, event log
│   ├── speech/                    # asr, vad, rate, pauses, fillers, repetitions, loudness, pitch, analyze
│   ├── text/                      # segment, central_message, organization, hook_conclusion,
│   │                              # supporting_material, language, hedging, relevance, analyze
│   ├── vision/                    # landmarks, framing, head_pose, gaze (T3 inference),
│   │                              # expressions (labels only), posture, gestures, analyze
│   ├── lang/                      # en/, ar/: lexicons, templates, ui strings, filler lists
│   ├── rubric/                    # loader, detectors/{rule,laya,llm_judge}.py, fusion, evidence, score
│   ├── engine/                    # skills, profile (Kalman), difficulty, library,
│   │                              # policies/{adaptive,fixed,practice_only,random,oracle}.py,
│   │                              # live_prompt_policy.py, explain.py, simulator.py
│   ├── coach/                     # providers/{openrouter,coach_slm,templates}.py, chain.py,
│   │                              # feedback.py, followups.py, guardrails.py, cache.py, tts.py
│   ├── knowledge/                 # kb build, retrieval, citation formatting
│   ├── pipeline/                  # orchestrator, benchmark (profile picker)
│   ├── storage/                   # db (SQLModel), features_store, analytics (DuckDB),
│   │                              # media (Fernet, local + Supabase), retention, rights (export/delete)
│   ├── study/                     # assignment (randomisation), rating queue, export
│   └── ui/                        # Gradio app
│       ├── app.py                 # gr.Blocks, tabs, theme, language switch
│       ├── pages/                 # consent, goal, calibrate, practice, report, exercise,
│       │                          # progress, data, rater, about
│       └── components/            # timeline plot, transcript highlight, rubric cards, skill bands
│
├── training/                      # runs on Kaggle/Colab/Unsloth Studio
│   ├── t1_whisper_verbatim/       # prepare_data.py, train.ipynb, eval.py, convert_ct2.sh, MODEL_CARD.md
│   ├── t2_coach_slm/              # build_pairs.py, filter_pairs.py, train.ipynb, export_gguf.ipynb, eval.py
│   └── t3_gaze_zone/              # protocol.md, extract_features.py, train.py, eval.py
│
├── deploy/
│   ├── space/                     # app.py entry, requirements.txt, README.md (Space card)
│   └── supabase/                  # schema.sql, storage policy notes
│
├── tests/
│   ├── fixtures/                  # tiny WAVs, landmark files, transcripts
│   ├── unit/  integration/  ui/  property/
│   └── safety/                    # banned terms, no-emotion labels, delete works,
│                                  # nothing leaves except allowed hosts, retention works
│
├── experiments/                   # E1–E12 scripts, configs/, results/ (small files committed)
├── notebooks/                     # exploration only
├── scripts/                       # download_models, fetch_public_data, build_knowledge_base,
│                                  # synth_speech, synth_text, check_licenses
├── data/                          # git-ignored; README.md register committed
├── models/                        # git-ignored; MANIFEST.json committed
├── resources/                     # rubrics (CC BY-NC-SA), ted_examples.yaml, teaching_links.yaml
├── demo/                          # script.md, fixtures (consented before/after)
└── docs/
    ├── plan/                      # these FINAL files
    ├── adr/                       # one page per threshold/design decision
    ├── cards/                     # model cards, data card, privacy & ethics, limitations
    ├── study/                     # protocol, consent (EN/AR), rater guide, analysis plan
    └── thesis/                    # chapters, figures, references.bib, weekly log
```

Runtime data (outside the repo), `PRESENCEAI_HOME` default `~/.presenceai/`:

```
app.sqlite  analytics.duckdb  llm_calls.jsonl  llm_cache/
sessions/<id>/events.jsonl  features/*.parquet  assessment.json  media/*.enc
```

## Rules the layout enforces

- `ui/` and `live/` may import packages; no package imports `ui/`.
- `schemas/` imports nothing from the project.
- `vision/expressions.py` may only emit the five allowed labels (test in `tests/safety/`).
- Only `coach/providers/openrouter.py` and `storage/media.py` (Supabase, Stage B) may open network connections; a socket-blocking test enforces it.
- Tests run without models, camera or network (fakes in `tests/fixtures/`).

## `pyproject.toml` groups

```toml
[project]
name = "presenceai"
requires-python = ">=3.12,<3.13"
license = "Apache-2.0"
dependencies = [
  "gradio", "fastrtc[vad]", "pydantic>=2", "pydantic-settings", "typer", "pyyaml",
  "numpy", "scipy", "pandas", "pyarrow", "duckdb", "sqlmodel", "cryptography",
  "soundfile", "librosa", "faster-whisper", "silero-vad",
  "mediapipe", "opencv-python-headless", "scikit-learn", "skops",
  "spacy", "textstat", "wordfreq", "rapidfuzz",
  "openai", "llama-cpp-python", "plotly",
]
[project.optional-dependencies]
laya     = ["laya"]
e5       = ["sentence-transformers"]
tts      = ["kokoro"]
space    = ["spaces", "supabase", "psycopg[binary]"]
diarize  = ["pyannote.audio"]
[dependency-groups]
dev         = ["ruff", "pyright", "pytest", "pytest-cov", "hypothesis", "pre-commit"]
experiments = ["jiwer", "statsmodels", "krippendorff", "matplotlib", "datasets", "jupyterlab"]
annotate    = ["label-studio"]
[project.scripts]
presenceai = "presenceai.cli:app"
```

Add each with `uv add`, commit `uv.lock`. Training dependencies (Unsloth, TRL) live only in the `training/` notebooks, not in the app environment.

## Commands

```bash
uv sync --extra laya
uv run python scripts/download_models.py
cp .env.example .env                 # paste OPENROUTER_API_KEY
uv run presenceai                    # http://127.0.0.1:7860
uv run presenceai analyze tests/fixtures/sample_en.wav
uv run presenceai simulate --policy adaptive --learners 2000 --seed 1
uv run pytest -m "not slow"
```
