# 0006 — Audio reaches faster-whisper only as an array

- Status: **accepted**. The owner decided on 2026-11-14 to always pass an array and not to pin
  PyAV. This change implements that decision.
- Date: 2026-11-14
- Decider: project owner
- Plan reference: docs/plan/FINAL_03_TECH_STACK.md (ASR: faster-whisper); FINAL_06 §3 (T1);
  ADR 0005 ("Laptop check")
- Config keys affected: none

## Context

faster-whisper 1.2.1 decodes file paths itself with PyAV and passes `metadata_errors` to
`av.open`. PyAV 19.0.2, the version this project resolves, no longer accepts that argument, so any
`model.transcribe("file.wav")` raises `TypeError`. Spike S3 hit this on 2026-11-14 (ADR 0005). It
avoided the error by decoding the clips itself and passing faster-whisper an array, which is a
documented input.

## Options considered

1. **Pin PyAV below 19.** This keeps file paths working, but it holds a security-relevant media
   library back and breaks again whenever another package needs a newer PyAV. Rejected by the
   owner.
2. **Always decode in our own code and pass an array.** This works with any PyAV version and puts
   decoding and resampling in one place we can test. Chosen.

## Decision

- `src/presenceai/speech/audio.py` is the only way audio reaches faster-whisper:
  - `load_audio(path)` returns float32 samples, 16 kHz, mono, in [-1, 1]. It decodes with PyAV
    and resamples with PyAV's `AudioResampler`; stereo is down-mixed by FFmpeg's standard rule,
    as faster-whisper's own decoder does.
  - `transcribe(model, path, **options)` calls `model.transcribe(load_audio(path), ...)`. No other
    module calls a faster-whisper `.transcribe(`.
- PyAV is a direct dependency (`av>=19.0.2`). This is a lower bound only, with no upper pin.
- CI guard: `tests/unit/test_audio_loader.py` scans `src/`, `experiments/` and `scripts/`. It
  fails if any file other than the loader calls `.transcribe(` (except `audio.transcribe(...)`),
  or if any file uses faster-whisper's `decode_audio`.

## Consequences

- `experiments/s3_unsloth_whisper/check_local.py` now uses the shared function. Re-running it on
  run `20261114-c3` gave the same 10 transcripts and the same decision (pass; WER 9.42 base,
  4.04 fine-tuned).
- New ASR code (`speech/asr.py`, the S2 Space) must call `presenceai.speech.audio.transcribe`. The
  guard test fails otherwise.
- The guard matches on the method name. Another library's `.transcribe(` would also be flagged and
  would need an explicit exception in the test.
- Revisit when faster-whisper drops `metadata_errors`. Passing arrays can stay either way.
