# Spike S2: faster-whisper on a Hugging Face ZeroGPU Space

## 1. Aim

Stage B of the project is an optional invited online pilot. It would run the application in a
Hugging Face Space on ZeroGPU, where GPU time is charged to each visitor's own daily quota.

Spike S2 examined whether a private Gradio ZeroGPU Space could run faster-whisper on a three-minute
recording within a free visitor's quota. The roadmap (`docs/plan/FINAL_12_ROADMAP_AND_THESIS.md`,
§1) set the pass condition: one analysis must use less than 60 GPU-seconds. The fallback was to run
Whisper `small` on the Space's CPU, or to drop Stage B.

## 2. Method

**Platform facts.** The ZeroGPU documentation was checked on 14 November 2026:
- a free account receives 5 GPU-minutes per day;
- the `@spaces.GPU` decorator reserves 60 seconds per call by default;
- ZeroGPU works only with the Gradio SDK.

The project owner's account was a free account, so its quota equalled a free visitor's.

**Model.** The `space` profile in the plan uses Whisper large-v3-turbo on the GPU.
- `openai/whisper-large-v3-turbo` (MIT licence, revision `41f01f3`) was converted once on the
  laptop to the CTranslate2 format with float16 weights.
- The converted model was uploaded to a private model repository and downloaded by the Space at a
  pinned revision.
- The SHA-256 of every model file was recorded.

**Software.** faster-whisper uses CTranslate2, which version 4.9.0 runs on CUDA 12 with cuDNN 9.
The Space therefore used PyTorch 2.9.1, a CUDA 12 build that installs these libraries. Following
ADR 0006, audio was decoded by the project's shared loader and passed to faster-whisper as an
array.

**Test audio.** The clip was 182.65 seconds long. It was made of 27 whole clips from the public
DisfluencySpeech test split (revision `b7da294`), chosen with seed 3407 and joined in order. A
10-second probe clip was cut from its start. Decoding used English, beam size 5 and temperature 0,
with no voice-activity filter.

**Procedure.**
1. **Probe:** a short call checked that CTranslate2 could see the GPU inside `@spaces.GPU`. If it
   had not, the spike would have stopped for a decision by the owner.
2. **Measured calls:** one cold call and three warm calls, each transcribing the full clip. The
   time inside the GPU function, including model loading, was measured in each call.
3. **CPU fallback:** Whisper `small` (int8) was then run on the Space's CPU for comparison.

A script on the laptop called the private Space through `gradio_client`. This was an
experiment-only exception to the project's network rule, approved by the owner. The access token
was referred to by name only.

## 3. Pass conditions

The owner set the rule on 14 November 2026, before any run (ADR 0007). All three conditions had to
hold:

1. Every measured call returned a non-empty transcript of the three-minute clip.
2. The median GPU time of the three warm calls, including model loading, was below 60 seconds.
3. No call exceeded the 60-second reservation.

The cold call, the CPU fallback, the word error rate (WER) and the remaining quota were reported
for information only.

## 4. Results

**Probe** (`20261114-z1_probe.json`).
- CTranslate2 saw one CUDA device: a 48 GB partition of an NVIDIA RTX PRO 6000 Blackwell.
- The installed libraries were PyTorch 2.9.1 for CUDA 12.8, cuBLAS 12.8.4 and cuDNN 9.10.2.
- The probe clip was transcribed correctly, using 11.92 GPU-seconds. Most of this was the
  transcription itself (10.74 s), which did not recur in later calls. This is consistent with a
  one-off start-up cost on first use, but that explanation was not measured.

**Table 1.** Calls on the 182.65-second clip, run `20261114-z1` (`20261114-z1_calls.csv`).

| Call | GPU time | Model loading | Transcription | Model already loaded | Round trip from laptop |
|---|---|---|---|---|---|
| Cold | 4.68 s | 1.27 s | 3.39 s | no | 12.38 s |
| Warm 1 | 4.71 s | 1.25 s | 3.45 s | no | 9.53 s |
| Warm 2 | 3.01 s | 0 s | 3.01 s | yes | 6.61 s |
| Warm 3 | 3.13 s | 0 s | 3.13 s | yes | 6.51 s |

**Table 2.** Pass rule (`20261114-z1_decision.csv`).

| Condition | Value | Holds |
|---|---|---|
| Non-empty transcripts | 4 of 4 | yes |
| Median warm GPU time below 60 s | 3.13 s | yes |
| No call above the 60 s reservation | largest 4.71 s | yes |

**Further observations.**
- **Speed:** transcription ran at about 0.02 times real time.
- **WER:** 8.02 against the clip's reference text in all four calls.
- **Model reuse:** in two of the three warm calls the model was still loaded from the previous
  call, so those calls did not include loading it. The cold call (4.68 seconds, with a full model
  load of 1.27 seconds) and the first warm call (4.71 seconds, with a load) are therefore the
  stronger evidence that one analysis stays under 60 GPU-seconds with loading included.
- **Total GPU time** measured inside the function was 27.45 seconds, probe included.
- **CPU fallback** (`20261114-z1_cpu.json`): Whisper `small` on the Space's CPU transcribed the
  clip in 8.21 seconds, with a WER of 9.00.
- **Quota:** the owner's billing page showed "Zero GPU 0/5 min" before the run and "0.5/5 min"
  after it; the page rounds to half-minutes. This agreed with the 27.45 seconds measured. If
  ZeroGPU had charged the 60-second reservation for each of the five GPU calls, the page would
  have shown about 5 minutes. ZeroGPU therefore appeared to charge actual use rather than the
  reservation.

## 5. Decision

Under the rule fixed before the run, the spike passed (decider: project owner, 14 November 2026,
ADR 0007, accepted the same day). One three-minute analysis used between 3.0 and 4.7
GPU-seconds, far below the 60-second reservation and the five-minute daily quota of a free
account. Stage B can therefore run Whisper on ZeroGPU, and the CPU fallback is not required to
answer this question.

## 6. Limitations

- **Small sample.** Four calls on one public clip with one speaker were measured on a single day.
  Queue times and GPU availability on a shared service vary.
- **Measurement point.** GPU time was measured inside the decorated function. The owner's quota
  reading agreed with it, but at half-minute resolution any small overhead per call could not be
  measured.
- **Reservation and quota.** A visitor can only start a call while their remaining quota covers the
  60-second reservation. Choosing a shorter reservation for Stage B is left as a design decision.
- **Not the full Stage B.** The spike covered transcription only. The live loop, the network relay
  and the pilot database of Stage B were not tested.
- **Single CPU measurement.** The CPU fallback was measured once, and the Space's share of the
  reported 192 processors is not known.

## Sources

- Results: `experiments/results/s2/`:
  - clip files: `clip_info.json`, `clip_manifest.csv` and `clip_reference.txt`;
  - model: `model_manifest.csv`;
  - run `20261114-z1`: `_probe.json`, `_calls.csv`, `_decision.csv` and `_cpu.json`.
- Settings: `experiments/configs/s2_zerogpu.yaml`.
- Code: `experiments/s2_zerogpu/`, with unit tests in `tests/unit/test_s2_zerogpu.py`.
- Decision records: `docs/adr/0007-zerogpu-space-spike.md` and
  `docs/adr/0006-audio-to-faster-whisper-as-array.md`.
