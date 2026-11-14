# Spike S3: fine-tuning Whisper with Unsloth on a free GPU

## 1. Aim

Spike S3 tested the pipeline that the speech-model task T1 depends on. A small Whisper model was
fine-tuned with Unsloth on a free cloud GPU, exported to the CTranslate2 format, and run with
faster-whisper on the project laptop's CPU. The spike was meant to show only that this pipeline
worked without damaging the model. Whether fine-tuning improves the transcription of filler words
was left to experiment E3.

If the pipeline failed, the plan provided a fallback: plain Hugging Face `Seq2SeqTrainer` with
PEFT (`docs/plan/FINAL_06_MODELS_AND_TRAINING.md`, §3).

## 2. Method

**Model and data.**
- **Base model:** `unsloth/whisper-small`, a mirror of `openai/whisper-small`, pinned to
  revision `023f354`.
- **Data:** the public DisfluencySpeech dataset (`amaai-lab/DisfluencySpeech`, revision
  `b7da294`), a single speaker. The target text was `transcript_a`, which keeps verbal
  disfluencies such as "uh" and "um".
- **Clips:** 200 training, 50 validation and 10 test clips, drawn from the dataset's own splits
  with shuffle seed 3407.

**Training.** The training cells of Unsloth's Whisper notebook (`unslothai/notebooks`, commit
`5f42bb9`) were adapted into a script.
- **LoRA:** rank 64 on the attention query and value projections.
- **Schedule:** 100 steps at an effective batch size of 4 and learning rate 1×10⁻⁴.
- **Validation loss:** at step 0 and every 10 steps.
- **Seeds:** 3407 for the data, LoRA and trainer.
- Every change from the notebook was listed in the script and in each run's information file.

**Export.** After training, the LoRA weights were merged into the model at 16-bit precision. Both
the base model and the fine-tuned model were converted to CTranslate2 with int8 weights. The 10
test clips were written as 16 kHz WAV files.

**Platform.**
- The spike was planned for Kaggle.
  - The first Kaggle run (`20261114-v1`) failed at import because of a library conflict in
    Kaggle's image.
  - The second waited about two hours for a GPU, never started, and was cancelled.
- On 14 November 2026 the owner moved the spike to Google Colab's free T4 GPU, the backup named in
  the plan (FINAL_06 §2).
- The Colab notebook was driven through Google's official Colab MCP server, pinned to commit
  `b9ab389`. No storage was mounted and no secret was used.
- Before each run, the script in Colab was checked against the build from a clean repository
  commit by SHA-256.

**Transfer.** The outputs were zipped in Colab and downloaded through the browser. Before any file
was used, its SHA-256 was checked against the table that the run wrote into its information file.
The Colab runtime was then deleted.

**Laptop check.** On the owner's laptop CPU, faster-whisper 1.2.1 transcribed the 10 test clips
with both models:
- int8 computation, beam size 5, English, temperature 0;
- no voice-activity filter and no conditioning on previous text.

The word error rate (WER) was pooled over the 10 clips: total edits divided by total reference
words. The texts were lower-cased, and punctuation and hyphens were removed, while apostrophes and
filler words were kept.

## 3. Pass conditions

The owner fixed the rule on 14 November 2026, before any run (ADR 0005). All three conditions had
to hold:

1. **The loss falls.**
   - The final validation loss was lower than at step 0.
   - The mean of the last 10 logged training losses was lower than the mean of the first 10.
2. **The model converts and loads.** The fine-tuned CTranslate2 model loaded in faster-whisper on
   the laptop CPU and returned a non-empty transcript for all 10 test clips.
3. **The model is not damaged.** The fine-tuned model's WER on the 10 clips was at most 10
   percentage points higher than the base model's.

The 10-point limit was a feasibility limit only: it checked that the pipeline did not break the
model. It was not the T1 target. That target, set in FINAL_06 §3, is a WER on clean speech no more
than 1 point higher than the base model on LibriSpeech test-clean, together with higher filler F1
and repetition recall. It is measured in experiment E3 (FINAL_10).

Filler counts and CPU time per clip were reported for information only.

## 4. Results

**Runs.** Three Colab runs were made with the same settings and seeds (Table 1). Training behaved
almost identically in all three. The first two failed at the export step. Each failure was fixed
and the run repeated with the owner's approval.

**Table 1.** Colab runs on 14 November 2026 (`experiments/results/s3/<run>_run_info.json` and
`<run>_train_log.csv`).

| Run | Validation loss, step 0 → 100 | Training time | Outcome |
|---|---|---|---|
| `20261114-c1` | 2.0898 → 0.5560 | 116 s | Export failed: the merged model had no configuration file |
| `20261114-c2` | 2.0898 → 0.5560 | 97 s | Base-model export failed: no `tokenizer.json` in the base model files |
| `20261114-c3` | 2.0898 → 0.5559 | 99 s | Both models exported |

**Export findings.**
- Unsloth's merge, called as `save_pretrained_merged(save_method=None)`, wrote the weights and
  the tokenizer files but no `config.json` or `generation_config.json`. These were copied from the
  pinned base model.
- The pinned base model contained no `tokenizer.json`, so the base export received the fine-tuned
  export's file.
- In run c2's runtime, the two tokenizers were checked and found identical in vocabulary
  (51,865 tokens) and in the token ids they produced. Both fixes were recorded in the run
  information file of `20261114-c3`.

**Laptop finding.** faster-whisper 1.2.1 could not decode audio files with the installed PyAV
19.0.2, because it passed an argument that this PyAV version no longer accepts. The laptop check
therefore read the WAV clips itself and passed the audio to faster-whisper as an array.

**Run c3.**
- **Hardware and software:** Tesla T4 (15,360 MiB), CUDA 13.0, Python 3.13.16, torch 2.14.1,
  transformers 4.57.6, Unsloth 2026.11.4 (`20261114-c3_run_info.json`).
- **Training:** 98.6 s, with at most 0.934 GB of GPU memory reserved.
- **Transfer:** the downloaded archive held 22 files (424,694,737 bytes), and all 22 matched their
  checksums.

**Table 2.** Pass rule, run `20261114-c3` (`20261114-c3_decision.csv`).

| Condition | Value | Holds |
|---|---|---|
| Validation loss, step 0 → end | 2.0898 → 0.5559 | yes |
| Mean training loss, first 10 → last 10 steps | 1.8103 → 0.5655 | yes |
| Fine-tuned model loads; non-empty transcripts | 10 of 10 | yes |
| WER, fine-tuned − base (limit +10 points) | 4.04 − 9.42 = −5.38 | yes |

**Table 3.** Information only, run `20261114-c3`: 10 clips, 64.56 s of audio
(`20261114-c3_local.csv`, `20261114-c3_local_info.json`). The laptop had an Intel64 Family 6
Model 154 processor with 12 logical CPUs.

| Measure | Reference | Base | Fine-tuned |
|---|---|---|---|
| Filler words (uh, um, er, erm, ah, hm, hmm, mm) | 8 | 1 | 8 |
| Mean CPU time per clip, including decoding | — | 3.868 s | 3.587 s |

**Transcripts.** The base model omitted 7 of the 8 reference fillers. The fine-tuned model kept
all 8. Some of the WER difference came from writing style rather than recognition: in
`test_09.wav` the fine-tuned model wrote "fifteen dollars", as the reference does, where the base
model wrote "$15". The fine-tuned model also made one error the base model did not: "caught"
instead of "called" in `test_06.wav`.

## 5. Decision

Under the rule fixed before the run, the spike passed (decider: project owner, 14 November 2026,
ADR 0005, accepted the same day). The Unsloth pipeline trained on a free GPU, exported to
CTranslate2 and ran with faster-whisper on the laptop CPU, and the fine-tuned model was not worse
than the base model on the test clips. T1 keeps the Unsloth route, and the fallback is not
needed.

T1 carries forward:
- the two export steps (configuration files for the merged model, and the base tokenizer);
- the PyAV incompatibility, which affects any later use of faster-whisper with file paths. The
  owner decided on 14 November 2026 that audio is always passed to faster-whisper as an array
  through one shared loader, without pinning PyAV. This is implemented in a separate follow-up
  change.

## 6. Limitations

- **Not evidence of generalisation.** The WER gain (9.42 to 4.04) and the kept fillers were
  measured on one speaker. The dataset has a single speaker, and the test clips came from that
  speaker. Part of the gain came from writing style rather than recognition, such as "fifteen
  dollars" instead of "$15". The result therefore says nothing about other speakers or about the
  T1 target.
- **Feasibility limit only.** Passing the 10-point limit showed that the pipeline did not damage
  the model. It does not show that T1's target of at most 1 point on LibriSpeech test-clean is
  met; experiment E3 tests that.
- **Small sample.** Ten clips and eight reference fillers are too few to estimate how well
  fillers are recognised. That is the purpose of experiment E3.
- **Shared runtime.** The three Colab runs shared one runtime, so runs c2 and c3 started from
  packages that c1 had installed.
- **Short training.** Training was short (100 steps on 200 clips) by design. The spike tested the
  pipeline, not the best achievable model.
- **One laptop.** CPU times were measured once, on one laptop, and include decoding the WAV
  files.

## Sources

- Results: `experiments/results/s3/`:
  - Kaggle run 1: `20261114-v1_run_info.json`;
  - Colab runs c1 and c2: `20261114-c1_run_info.json`, `20261114-c1_train_log.csv`,
    `20261114-c2_run_info.json` and `20261114-c2_train_log.csv`;
  - Colab run c3: `20261114-c3_run_info.json`, `_train_log.csv`, `_models_manifest.csv`,
    `_local.csv`, `_local_info.json` and `_decision.csv`.
- Settings: `experiments/configs/s3_unsloth_whisper.yaml`.
- Code: `experiments/s3_unsloth_whisper/`, with unit tests in `tests/unit/test_s3_spike.py`.
- Decision record: `docs/adr/0005-unsloth-whisper-kaggle.md`.
