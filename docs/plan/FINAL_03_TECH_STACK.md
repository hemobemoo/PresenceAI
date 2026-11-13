# FINAL 03 — Tech Stack

Python 3.12, one `uv`-managed project. No JavaScript or TypeScript is written by you; Gradio and FastRTC supply the browser side.

✓ = checked on the project's page on 11 Nov 2026 · ~ = confirm before citing.

## 1. Web UI and app

| Need | Choice | Licence | Checked | Notes |
|---|---|---|---|---|
| Web UI | **Gradio** (`gr.Blocks`, tabs, `gr.Plot`, `gr.HighlightedText`, `gr.Video`) | Apache-2.0 | ~ | Runs locally and in HF Spaces; RTL text supported on text components ~ |
| Live audio/video | **FastRTC** (`WebRTC` component, `Stream`, `ReplyOnPause`, `AdditionalOutputs`) | MIT | ✓ | `pip install fastrtc`; Cloudflare TURN helper with HF token, 10 GB/month free |
| API, routes | FastAPI (under Gradio) | MIT | ~ | Export endpoints, health check |
| Schemas, settings | Pydantic v2, pydantic-settings | MIT | ~ | — |
| CLI | Typer | MIT | ~ | `analyze`, `simulate`, `train-prep`, `study`, `retention` |
| Charts | Plotly | MIT | ~ | Timeline, skill bands |
| i18n | YAML string tables (`en.yaml`, `ar.yaml`) | — | — | Gradio's own i18n helper can be used if your Gradio version has it ~ |

Replaced: NiceGUI (no browser mic/webcam recorder without custom JS).

## 2. Speech

| Need | Choice | Licence | Checked |
|---|---|---|---|
| ASR | **faster-whisper** (`small`, `turbo`, or your fine-tuned T1 model converted to CTranslate2) | MIT | ✓ |
| VAD | Silero VAD | MIT | ~ |
| Audio features | librosa, NumPy, SciPy (own syllable-nucleus detector; no Praat/parselmouth, which is GPL) | ISC/BSD | ~ |
| WER | jiwer | Apache-2.0 | ~ |
| Diarization (optional) | pyannote.audio + `speaker-diarization-community-1` | MIT / CC-BY-4.0 | ✓ |
| Verbatim comparator (experiments only) | CrisperWhisper 2.0 | MIT code, non-commercial weights | ✓ |

## 3. Vision

| Need | Choice | Licence | Checked |
|---|---|---|---|
| Pose | MediaPipe Pose Landmarker (lite/full) | Apache-2.0 | ✓ |
| Face, iris, head pose, expressions | MediaPipe Face Landmarker: 478 landmarks, 52 blendshapes, transformation matrix | Apache-2.0 | ✓ |
| Frames | OpenCV (`opencv-python-headless`) | Apache-2.0 | ~ |
| Gaze-zone classifier (T3) | scikit-learn (logistic regression / gradient boosting) on landmark features | BSD | ~ |

## 4. Text, knowledge base and decisions

| Need | Choice | Licence | Checked |
|---|---|---|---|
| NLP | spaCy `en_core_web_sm`; Arabic: rule-based tokenisation with `camel-tools` ~ or simple regex | MIT / MIT ~ | ~ |
| Embeddings (default) | Your **EmbeddingGemma GGUF** via `llama-cpp-python` | Gemma Terms of Use (free, not OSI) | ✓ |
| Embeddings (open alternative) | `intfloat/multilingual-e5-small` via sentence-transformers | MIT ~ | ~ |
| Vector search | DuckDB `vss` extension or NumPy cosine over a small matrix | MIT | ~ |
| Typed rubric decisions | **Laya** (`pip install laya`) | Apache-2.0 | ✓ |
| Calibration, small models | scikit-learn | BSD | ~ |
| Readability, word frequency, fuzzy match | textstat, wordfreq, rapidfuzz | MIT / Apache-2.0 / MIT | ~ |

## 5. Language models

| Role | Choice | Cost | Checked |
|---|---|---|---|
| Default LLM | **OpenRouter** free model: pinned id + fallback(s) from `configs/models.yaml` (S4, ADR 0003: NVIDIA pinned, NVIDIA fallback, Recursive emergency fallback), called with the `openai` SDK | $0; 20 req/min, 1,000/day | ✓ |
| Teacher for T2 data | Same, restricted to a model whose licence allows training on its outputs (e.g. Apache-2.0/MIT open-weight models) | $0 | ~ |
| Coach-SLM (T2) | Fine-tuned small instruct model (start with Qwen3.5 0.8B or 2B; licence to confirm) served as GGUF with `llama-cpp-python` | $0 | ~ |
| Fallback | Deterministic templates in EN and AR | — | — |

## 6. Training

| Need | Choice | Licence | Checked |
|---|---|---|---|
| Fine-tuning | **Unsloth** (notebooks or Unsloth Studio) with TRL/PEFT LoRA | Apache-2.0 ~ | ✓ supports Whisper STT |
| Compute | Kaggle Notebooks (~30 GPU-h/week), Google Colab free | $0 | ~ |
| Experiment tracking | CSV/Parquet + DuckDB; optional Weights & Biases free tier is not used | — | — |
| Conversion | `ct2-transformers-converter` (CTranslate2) for T1; Unsloth GGUF export for T2 | MIT / Apache-2.0 | ~ |

## 7. Storage and services

| Need | Choice | Licence / cost | Checked |
|---|---|---|---|
| ORM | SQLModel (SQLite locally, Postgres in Space) | MIT | ~ |
| Analytics | **DuckDB** + pyarrow | MIT / Apache-2.0 | ~ |
| Encryption | cryptography (Fernet) | Apache-2.0/BSD | ~ |
| Pilot DB/storage | **Supabase** free: 500 MB Postgres, 1 GB storage, 50 MB max file, pauses after 1 week idle | $0 | ✓ |
| Model/dataset hosting | Hugging Face Hub | $0 | ~ |
| Online pilot | Hugging Face ZeroGPU Gradio Space | $0 | ✓ |
| TTS for panel questions (optional) | Kokoro-82M (English only for MVP; Arabic questions shown as text) | Apache-2.0 | ✓ |

## 8. Statistics, annotation, engineering

| Need | Choice | Licence |
|---|---|---|
| Statistics | SciPy, statsmodels, own ICC function (avoid pingouin, which is GPL-3.0 ~) | BSD |
| Inter-rater reliability | `krippendorff` | BSD |
| Annotation | Label Studio (audio spans, video frames) | Apache-2.0 |
| Lint, types, tests | Ruff, Pyright, pytest, Hypothesis | MIT / MPL-2.0 |
| CI | GitHub Actions | $0 public repos |

## 9. Hardware profiles

| Profile | Machine | ASR | Live vision | LLM |
|---|---|---|---|---|
| `lite` | 8 GB RAM, CPU | `small` int8 (or T1-small) | Pose lite + Face, 8 fps | OpenRouter → templates |
| `standard` | 16 GB RAM, CPU | `turbo` int8 (or T1-turbo) | Pose full + Face, 10–12 fps | OpenRouter → Coach-SLM → templates |
| `space` | ZeroGPU Space | `turbo` on GPU | Pose lite + Face, 8 fps on Space CPU | OpenRouter → templates |

## 10. Licence position

Project code: Apache-2.0. Rubric and knowledge-base derivatives of CC BY-NC-SA sources are shared under CC BY-NC-SA 4.0 in `resources/` with attribution. No GPL/AGPL runtime dependency. Not OSI but used: EmbeddingGemma (Gemma Terms), OpenRouter and Supabase (hosted services). Research-only, never shipped: CrisperWhisper weights, PodcastFillers, Speak & Improve, TED material.
