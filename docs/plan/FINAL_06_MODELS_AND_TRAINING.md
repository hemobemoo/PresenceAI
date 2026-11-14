# FINAL 06 — Models and Training

✓ = checked on the provider's page on 11 Nov 2026 · ~ = confirm when you run it.

## 1. Pretrained models used as-is

| Model | Role | Get it | Licence | Checked |
|---|---|---|---|---|
| Whisper `small`, `turbo` (via faster-whisper) | Baseline ASR (EN, AR) | Auto-download on first `WhisperModel("small")` | MIT | ✓ |
| Silero VAD | Voice activity | `pip install silero-vad` | MIT | ~ |
| MediaPipe Face Landmarker | 478 landmarks, 52 blendshapes, head pose | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task` | Apache-2.0 | ✓ |
| MediaPipe Pose Landmarker (full) | Body landmarks | `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task` | Apache-2.0 | ~ URL pattern |
| Laya | Typed rubric decisions | `pip install laya` (downloads checkpoints on first use, ~2.3 GB for all three) | Apache-2.0 | ✓ |
| EmbeddingGemma GGUF (yours) | Embeddings, knowledge-base search | Already on disk; or `huggingface-cli download unsloth/embeddinggemma-300m-GGUF` ~ | Gemma Terms | ✓ licence |
| multilingual-e5-small | Open-licence embedding alternative | `intfloat/multilingual-e5-small` on HF | MIT ~ | ~ |
| spaCy `en_core_web_sm` | English NLP | `uv run python -m spacy download en_core_web_sm` | MIT | ~ |
| Kokoro-82M (optional) | Reads panel questions aloud (EN) | `pip install kokoro soundfile` + system `espeak-ng` | Apache-2.0 | ✓ |
| OpenRouter free model | LLM (hosted) | API key in `.env` | Per model | ✓ limits |
| CrisperWhisper 2.0 (experiments only) | Upper-bound comparator for fillers | `pip install "crisperwhisper[transformers]"` | Non-commercial weights | ✓ |

`scripts/download_models.py` fetches all local models, writes SHA-256, size and licence to `models/MANIFEST.json`.

## 2. Where training runs

| Option | Use when | How | Cost |
|---|---|---|---|
| **Kaggle Notebooks** (default) | No NVIDIA GPU at home | Upload the notebook from `training/`, enable GPU (T4×2 or P100), attach your private dataset | $0, ~30 GPU-h/week ~ |
| Google Colab free | Kaggle quota used up | Same notebook; Unsloth's Whisper notebook opens directly in Colab ✓ | $0, availability varies |
| **Unsloth Studio** (local) | Your machine has a supported GPU | macOS/Linux/WSL: `curl -fsSL https://unsloth.ai/install.sh \| sh` then `unsloth studio -H 0.0.0.0 -p 8888`; Windows: `irm https://unsloth.ai/install.ps1 \| iex` ✓ | $0 |

Rules: every training run is a notebook in `training/` with fixed seed, logs its data manifest hash, and pushes the result to a **private** Hugging Face repo. Raw study recordings never go to Kaggle or Colab; only public data, synthetic data and your own pilot recordings (with your own consent) do.

## 3. T1 — Verbatim-Whisper (speech fine-tune, primary training result)

**Goal.** Standard Whisper often cleans fillers and repetitions out of transcripts, which hides exactly what PresenceAI coaches. T1 fine-tunes Whisper to transcribe verbatim so filler and repetition counts become trustworthy.

| Item | Decision |
|---|---|
| Base models | `openai/whisper-small` (runs on any laptop CPU) and `openai/whisper-large-v3-turbo` (quality; GPU/Space) |
| Method | LoRA with Unsloth, following the official Whisper STT notebook: `https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Whisper.ipynb` ✓ |
| Training data | DisfluencySpeech train split (Apache-2.0, transcripts that keep "uh"/"um"; note it is reportedly single-speaker ~), AMI headset subset with verbatim transcripts (CC BY 4.0 ~), your pilot recordings hand-corrected to verbatim (≈ 1–2 h) |
| Held-out evaluation | DisfluencySpeech test; PodcastFillers test (research-only, evaluation never training); 20 pilot/study clips; LibriSpeech test-clean (regression check) |
| Target metrics | Filler F1 ↑, repetition recall ↑, WER on clean speech not worse than +1 point absolute |
| Budget | whisper-small: ~1–2 GPU-h; turbo: ~3–5 GPU-h per run ~ |
| Export | Merge LoRA → `ct2-transformers-converter --model <merged_dir> --output_dir whisper-small-verbatim-ct2 --quantization int8 --copy_files tokenizer.json preprocessor_config.json` ~ → load with faster-whisper |
| Publish | Private HF model repo during the project; public at the end with model card (base MIT, data Apache-2.0 / CC BY 4.0 attributions) |
| Arabic | Future work: needs verbatim Arabic transcripts with fillers, which public corpora do not provide |

## 4. T3 — Gaze-zone classifier (small vision model)

| Item | Decision |
|---|---|
| Labels | `at_camera` (lens or screen area: toward the audience), `down_notes`, `away` (+ `unknown` by rule). `at_screen` merged into `at_camera` after S5 (ADR 0004); a separate `at_screen` is future work |
| Features | Iris-to-eye-corner ratios (both eyes), eyelid aperture, head yaw/pitch/roll, face size |
| Data | Prompted-target recordings: you and 4–6 volunteers look at on-screen targets in a fixed order (labels known by protocol, no manual annotation), across the robustness conditions (lighting, glasses, camera height, distance) |
| Model | scikit-learn: logistic regression baseline, HistGradientBoosting main; per-user offset from the 10 s calibration |
| Evaluation | Leave-one-person-out macro-F1 over the three labels; per-condition breakdown and excluded (`unknown`) share per zone (E7) |
| Runs on | Laptop CPU in seconds; saved with `skops` ~ |

## 5. T2 — Coach-SLM (small language model fine-tune)

**Goal.** Test whether a small local model can turn a structured assessment into feedback, exercise wording and follow-up questions as faithfully as a hosted model, without network or quota.

| Item | Decision |
|---|---|
| Task | Input: `Assessment` JSON (findings, evidence spans, numbers, target skill, language). Output: JSON with 1–3 feedback statements (each with quote/time span), exercise wording, 2 follow-up questions |
| Base model | Smallest instruct model that passes a 30-item pilot: try Qwen3.5 0.8B and 2B first (confirm licence), Gemma 3 1B as alternative (Gemma Terms) ~ |
| Data | ~3,000 pairs. Inputs from the pipeline run on the synthetic text set, pilot recordings and public speech data. Outputs from the OpenRouter teacher, **kept only if** quotes verify, numbers match the input, no banned terms, valid JSON. Plus ~200 pairs you write or edit by hand as gold |
| Grounding text | Exercise wording draws on CC BY 4.0 text only (*Fundamentals of Public Speaking*, Lumen) so the trained adapter carries no share-alike or non-commercial obligation; CC BY-NC-SA texts are used for retrieval, not for training ~ (licence interpretation: confirm with supervisor) |
| Teacher quota | Batch 5 items per request → ~600 requests, spread over non-study days |
| Teacher terms | Use a teacher model whose licence permits training on outputs; check OpenRouter's and the provider's terms before generating ~ |
| Method | QLoRA r=16 with Unsloth on Kaggle, 2–3 epochs, max length 2,048 |
| Export | Unsloth GGUF export (q4_k_m) → `llama-cpp-python` on the laptop |
| Arabic wording | Teacher prompts for Arabic items use the code-switching rule in `lang/ar/coach_prompt.yaml` (FINAL_04 §5, ADR 0003). Arabic pairs are **kept only if** no quote contains Latin-script text missing from the transcript (`latin_outside_transcript`); hand-written gold pairs follow the same rule |
| Feedback points | Teacher prompts in both languages add `feedback_point_rule` from `lang/<lang>/coach_prompt.yaml` (ADR 0003): each point says what was observed (with evidence), why it matters for the user, and what they gain by practising it. Pairs are **kept only if** every item has a non-empty observation, evidence quote and suggestion; whether the benefit is stated is checked by hand on the gold set and a sample of teacher pairs (no automatic check), and rated in E11 |
| Evaluation | E11: faithfulness (quote found, numbers correct), guardrail violations, JSON validity, blind rater actionability vs teacher and templates, Arabic code-switching rule followed (rater criterion), explains the benefit to the user (rater criterion), latency on laptop |
| Status in product | Provider option `coach_slm`; default stays OpenRouter unless E11 shows parity |

## 6. Provider chain at runtime

```
openrouter (pinned free model) ──429/timeout──▶ openrouter fallback(s), in order ──▶ coach_slm (if installed) ──▶ templates
```

Model ids come from `configs/models.yaml` (ADR 0003): pinned `nvidia/nemotron-3-ultra-550b-a55b:free`, fallback `nvidia/nemotron-3.5-super-vl:free` (both NVIDIA), then the emergency fallback `recursive/flint-v1.0:free` from a different provider (passed the S4 re-run). The chain runs only when consent switch (d) is on; otherwise templates are used (FINAL_11 §3).

Every generated string passes the same guardrails regardless of provider (`FINAL_11` §2). The prompt builder adds `feedback_point_rule` (observation with evidence, why it matters, what the user gains) in both languages, and for Arabic also `language_rule` (code-switching for English terms in the wording; quotes copied exactly), both from `lang/<lang>/coach_prompt.yaml`. Arabic quotes are rejected when `latin_outside_transcript` is non-empty (P7: prompt builder and its tests do not exist yet). Cache by prompt hash; log model id, provider, latency.

## 7. Model cards

`docs/cards/MODEL_CARD_T1.md`, `_T2.md`, `_T3.md`, plus one card listing the pretrained models in §1 with measured accuracy from `FINAL_10`.
