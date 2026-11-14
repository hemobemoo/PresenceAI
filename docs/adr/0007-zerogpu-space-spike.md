# 0007 — faster-whisper on a ZeroGPU Space (spike S2)

- Status: **accepted** by the owner on 2026-11-14, after CI passed on `6162727`. The owner fixed
  the pass rule on 2026-11-14, before any run. Results: pass (run `20261114-z1`); see "Results" and
  "Decision".
- Date: 2026-11-14
- Decider: project owner
- Plan reference: docs/plan/FINAL_12_ROADMAP_AND_THESIS.md §1 (S2); FINAL_02 §4 (Stage B);
  FINAL_03 (`space` profile: `turbo` on GPU)
- Config keys affected: `experiments/configs/s2_zerogpu.yaml` (spike only)
- Evidence: `experiments/results/s2/`, written by `experiments/s2_zerogpu/`

## Context

Stage B, the optional invited online pilot, would run the app in a Hugging Face ZeroGPU Gradio
Space. There, each visitor's GPU time is charged to that visitor's own daily quota.

S2 asks: *can a private Gradio ZeroGPU Space run faster-whisper on a 3-minute clip within a free
visitor's quota?* (FINAL_12). The fallback is to run Whisper `small` on the Space CPU, or to drop
Stage B.

**Facts checked on 2026-11-14 (huggingface.co/docs/hub/spaces-zerogpu).**
- A free account gets 5 GPU-minutes a day.
- `@spaces.GPU` reserves 60 s by default.
- The default GPU is half an NVIDIA RTX Pro 6000 Blackwell.
- Supported versions: Python 3.12.12; PyTorch 2.8–2.13.
- ZeroGPU works only with the Gradio SDK.
- ZeroGPU emulates CUDA for PyTorch outside `@spaces.GPU`. faster-whisper runs on CTranslate2,
  which uses CUDA directly, so its model is loaded inside the GPU function.

**Owner's account.** A free account, so its own quota equals a free visitor's.

**CTranslate2 4.9.0 needs cuBLAS for CUDA 12 and cuDNN 9** (faster-whisper README). The Space
therefore uses `torch==2.9.1`, a CUDA 12 build that ships both libraries. The probe records the
versions actually installed.

## Setup (owner decisions, 2026-11-14)

**Model.** `openai/whisper-large-v3-turbo` (MIT), revision `41f01f3`.
- Converted once on the laptop with ctranslate2 4.9.0 and transformers 4.57.6
  (`--quantization float16`); not converted at Space start.
- Uploaded to a private Hugging Face model repo, from which the Space downloads it at a pinned
  revision.
- The same repo holds S3's whisper-small int8 base export, for the CPU fallback measurement.
- SHA-256 of every file: `experiments/results/s2/model_manifest.csv`.
- `deepdml` conversions are not used.

**Space.** Private, Gradio SDK 5.23.1, Python 3.12, ZeroGPU.
- The repo id is read from the Space variable `S2_MODEL_REPO`.
- A read-only token for that one repo is the Space secret `HF_TOKEN`.
- Audio is passed as an array (ADR 0006); the Space carries a copy of the shared loader.

**Clip.** 182.65 s made of 27 whole public DisfluencySpeech test clips (revision `b7da294`, seed
3407), plus a 10 s probe clip. SHA-256 values are in `clip_info.json`.

**Decoding.** As in S3's laptop check: English, beam 5, temperature 0, no VAD, no conditioning on
previous text. GPU compute type float16.

**Order.**
1. **Probe:** CTranslate2 must see a CUDA device inside `@spaces.GPU` and transcribe the probe
   clip. If it cannot, the spike stops and the owner chooses the next step. There is no switch to
   the `transformers` route without the owner.
2. **Measured calls:** one cold call, then three warm calls, on the 3-minute clip.
3. **CPU fallback:** whisper-small int8 on the Space CPU. This is for information and uses no GPU
   quota.

**Quota.**
- If the daily quota runs out, the run stops and continues after the reset; `measure.py` resumes
  without repeating recorded calls.
- Nothing is upgraded or paid for.
- If Hugging Face shows the remaining quota, the owner records it before and after the run.

**Network.** The `gradio_client` calls to the private Space, from `experiments/s2_zerogpu/` only,
are an experiment-only exception to AGENTS.md rule 2 (owner, 2026-11-14). `HF_TOKEN` is used by
name only and never printed.

**Downloads outside `scripts/download_models.py`.** Two public files were downloaded on the laptop
while preparing S2. The owner approved them after the fact on 2026-11-14; the AGENTS exception was
not widened, and any future download outside `scripts/download_models.py` needs the owner first.
Each file's SHA-256 matched the Hub's LFS hash.

| File | Repo and revision | Bytes | SHA-256 | Used for |
|---|---|---|---|---|
| `model.safetensors` | `openai/whisper-large-v3-turbo` @ `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | 1,617,824,864 | `542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1` | the one-time CTranslate2 conversion |
| `data/test-00000-of-00001.parquet` | `amaai-lab/DisfluencySpeech` (dataset) @ `b7da294fe3a70dd96df6640893f2a5dfc2c87638` | 70,868,626 | `e78837be393206fc0ae6a0cefb4ec8c942c3e26401640eb8af67cf0dac21b4b0` | the 3-minute and probe clips |

**Setup check (2026-11-14, read-only, with the upload token).**
- The model repo and the Space both exist and are private.
- The Space SDK is Gradio, and the requested hardware is `zero-a10g` (ZeroGPU).
- The Space has the secret `HF_TOKEN` (name only) and the variable `S2_MODEL_REPO`, which matches
  the model repo's ID exactly.
- The upload token is fine-grained, with write access to these two repos only. On the first
  check it saw neither repo; the owner added them to its permissions.
- **Quota before the run:** the owner's billing page showed "Zero GPU 0/5 min".

## Pass rule (owner, 2026-11-14, fixed before the run)

All three must hold:

1. **Non-empty transcript.** Every measured call returns a non-empty transcript of the 3-minute
   clip.
2. **Median under 60 GPU-seconds.** The median of the three warm calls' time inside
   `@spaces.GPU` is below 60 s. That time includes loading the model.
3. **Within the reservation.** No call exceeds the 60 s reservation (`@spaces.GPU(duration=60)`).

**Reported for information, not part of the rule:**
- the cold call;
- load and transcription times;
- client round-trip time (including queue wait);
- whether the model stayed loaded between calls;
- WER against the clip's reference;
- the whisper-small CPU fallback on the same clip;
- remaining quota, if Hugging Face shows it.

**If the rule fails:** the numbers are reported, and the plan's fallback (Whisper `small` on the
Space CPU, or dropping Stage B) goes to the owner.

## Results (run `20261114-z1`, 2026-11-14)

Sources: `experiments/results/s2/20261114-z1_probe.json`, `_calls.csv`, `_decision.csv` and
`_cpu.json`.

**Space.**
- Built from the Space repo commit `766fd30`, with the model repo pinned at `7f416da`. All 10
  model files on the Hub matched the manifest: SHA-256 for both `model.bin` files, size for the
  others.
- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, as the MIG slice `2g.48gb`.
- Packages: torch 2.9.1 (CUDA 12.8, cuDNN 9.10.2); `nvidia-cublas-cu12` 12.8.4.1;
  `nvidia-cudnn-cu12` 9.10.2.21; ctranslate2 4.9.0; faster-whisper 1.2.1; av 19.0.2;
  gradio 5.23.1; spaces 0.51.3.
- The model download at Space start took 4.2 s.

**Probe** (10 s clip).
- CTranslate2 saw 1 CUDA device inside `@spaces.GPU`, with float16 among its compute types.
- The model loaded in 1.18 s, and the clip transcribed correctly in 10.74 s.
- GPU time: 11.92 s.
- The slow transcription of this first clip did not recur (see below). One explanation is one-off
  CUDA kernel compilation on first use with this new GPU generation. This is a hypothesis, not
  something the probe measured.

**Measured calls** (182.65 s clip).

| Call | GPU-s (inside `@spaces.GPU`) | Model load | Transcription | Model already loaded | Client round trip |
|---|---|---|---|---|---|
| cold | 4.682 | 1.268 | 3.390 | no | 12.381 |
| warm 1 | 4.706 | 1.248 | 3.452 | no | 9.530 |
| warm 2 | 3.008 | 0 | 3.006 | yes | 6.613 |
| warm 3 | 3.133 | 0 | 3.131 | yes | 6.506 |

**Pass rule.**

| Condition | Value | Holds |
|---|---|---|
| Every transcript non-empty | 4 of 4 | yes |
| Median warm GPU-s < 60 | 3.133 | yes |
| No call over the 60 s reservation | largest 4.706 | yes |

**For information.**
- **Transcription speed:** 0.017–0.019 × real time.
- **WER against the clip reference:** 8.02 on all four calls (base turbo).
- **Model staying loaded:** in 2 of the 3 warm calls the model was still loaded from the previous
  call. Those two calls did not include a model load. The cold call (4.68 s, full load of 1.27 s)
  and warm call 1 (4.71 s, load of 1.25 s) are therefore the stronger evidence that one analysis
  stays under 60 GPU-seconds with the model load included.
- **Total GPU time measured inside the function:** 27.45 s (probe 11.92, calls 15.53).
- **CPU fallback:** whisper-small int8 on the Space CPU took 8.21 s to transcribe (model load
  0.38 s), WER 9.00. The Space reported 192 CPUs, which is the host's count; the Space's own CPU
  share is not known.
- **Quota cross-check:** the owner's billing page showed "Zero GPU 0/5 min" before the run and
  "0.5/5 min" after it. The page rounds to half-minutes.
  - The reading agrees with the 27.45 GPU-seconds measured inside the function (0.46 min).
  - Had ZeroGPU charged the 60 s reservation for each of the five GPU calls (probe and four
    calls), it would have shown about 5 min.
  - So ZeroGPU appears to charge actual use, not the reservation. At half-minute resolution, any
    overhead per call cannot be measured more precisely than this.

## Decision

**Pass** under the rule fixed before the run (owner, 2026-11-14). A private ZeroGPU Space ran
faster-whisper large-v3-turbo on a 3-minute clip in 3.1–4.7 GPU-seconds per analysis, far inside
the 60 s reservation and a free account's 5-minute daily quota. Stage B can keep Whisper on
ZeroGPU, and the CPU fallback is not needed for this question.

Limits:
- **One clip, one day.** These are four calls on one public clip, one speaker, measured on one
  day. Queue waits and GPU availability vary.
- **Measured inside the function.** GPU-seconds were timed inside `@spaces.GPU`. The owner's
  quota reading (0.5 of 5 min after the run, rounded to half-minutes) agrees with that measurement.
  At that resolution, overhead per call cannot be ruled out.
- **Reservation can block visitors.** A visitor can only start a call if their remaining quota
  covers the reservation. With `duration=60`, the last minute of a visitor's quota cannot be used.
  A shorter reservation, which ZeroGPU also queues with higher priority, is a later design
  decision for Stage B; it was not changed here.
- **Not Stage B as designed.** This is not the full Stage B pipeline: no live loop, no TURN relay,
  no Supabase, and audio was uploaded as a file.

Accepted by the owner on 2026-11-14. After the run, the Space was paused with the upload token and
kept private for Stage B. The private model repo was kept.
