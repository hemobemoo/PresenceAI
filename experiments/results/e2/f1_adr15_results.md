# E2 after ADR 0015 (words outside speech removed), run f1, Colab T4

Config `35df562d1acd331a`, commit 6d0eada, `--device cuda`, int8 whisper_small. Transcripts came
from the CUDA cache written by the earlier runs (config `be073a2a540d4a3b`), so only VAD, the trim
step and the scoring ran again. The order, approved by the owner on 2026-11-18, was: development
part, then one clean test run, then 20 dB and 10 dB noise. The test results are marked
`dirty: true` because the development results were written into the Colab checkout before the
test runs and were not yet committed; the code was exactly 6d0eada.

| Run | wpm error | Pause F1 0.5 / 1.0 s | Duration error | Before ADR 0015 (wpm error) |
|---|---|---|---|---|
| Development part, clean | 0.98 % | 0.973 / 0.986 | 0.080 s | 1.01 % |
| **Test part, clean (pass/fail): passed** | **0.94 %** | **0.970 / 0.982** | **0.076 s** | 0.97 % |
| Test part, 20 dB noise | 0.93 % | 0.960 / 0.975 | 0.090 s | 0.98 % |
| Test part, 10 dB noise | 1.02 % | 0.959 / 0.973 | 0.087 s | 1.02 % |

The ADR 0009 targets are unchanged: wpm error ≤ 5 %, pause F1 ≥ 0.90, duration error ≤ 0.10 s.
The pause measures are the same as before, because the trim step does not change VAD.

## Words removed, clip by clip

Read from `words_outside_speech` in the `_clips.csv` files. The words themselves were read from
the cached transcripts on Colab, for description only; nothing was tuned on the test part.

| Run | Clip | Removed | Kind | wpm error before → after |
|---|---|---|---|---|
| dev | s10_r0 | "you. Thank you." after the last speech | invented | 5.54 → 2.83 % |
| test clean | s17_r0 | "watching, and I'll see you in the next video." after the last speech | invented | 10.10 → 1.98 % |
| test clean | s05_r4 | **"When"** at 0.00–0.22 s; speech starts at 0.51 s | **real first word** | 0.74 → 1.63 % |
| snr20 | s17_r0 | 14 words ("… thank you very much for your time, and I'll see you in the next video.") | invented | 13.71 → 1.08 % |
| snr20 | s05_r4, s37_r4, s39_r4 | **"When", "A", "We've"** at 0.00–0.22 s | **real first words** | 0.74 → 1.63, 8.86 → 9.76, 0.40 → 0.49 % |
| snr20 | s50_r4 | "There" at 0.00–0.00 s, p 0.02 (script starts "With so many") | unclear | 0.77 → 0.13 % |
| snr10 | s05_r4 | **"When"** at 0.00–0.20 s | **real first word** | 0.92 → 1.82 % |
| snr10 | s55_r0 | "You" at 59.20–59.30 s, p 0.04, after the last speech | invented | 0.21 → 0.67 % |

- **The end of the recording works as intended.** Every word removed after the last speech is
  invented.
- **The start of the recording removes real words.** Whisper sometimes squeezes the real first
  word into 0.00–0.22 s while speech starts at about 0.5 s. That word then ends more than 0.2 s
  before the first onset and is removed. This did not happen on the development part, either in
  the CPU transcripts behind ADR 0015 or in these CUDA transcripts. It was first seen here, on
  the test part.
- s55_r0 at 10 dB: the error rose although the removed word was invented. The count was already
  below the truth, so removing one more word moved it further away.
