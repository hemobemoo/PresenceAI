# 0007 — faster-whisper on a ZeroGPU Space (spike S2)

- Status: **proposed**. The owner fixed the pass rule on 2026-11-14, before any run. Results are
  pending.
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

## Results

Pending.
