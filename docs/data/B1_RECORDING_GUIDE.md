# B1 recording guide (owner only)

B1 is the pilot set: **20 clips of your own speech, 10 English and 10 Arabic, 2–3 minutes
each** (ADR 0010, ADR 0016). It is the only real-speech check for several decisions that were
made on synthetic speech (C1):

| Check | What B1 shows |
|---|---|
| E2 Arabic (FINAL_10 §2) | Rate and pause accuracy in Arabic. No public Arabic source has timing ground truth, so this is reported descriptively, for one speaker |
| E2 English on real speech | The same measures in English, next to the C1 result |
| E1 | Word error rate on your 20 hand-corrected clips |
| ADR 0015 | Whether the 0.2 s end margin removes real words, especially your quiet closing |
| ADR 0014 | Whether mid-sentence pauses are labelled correctly in real speech |
| ADR 0011 | Arabic punctuation (positions become `unknown`), VAD minimum speech, long sentences |
| ADR 0013 | Clipping and the plateau ratio on real speech, as a first check with one microphone |
| ADR 0016 | Whether the language check gives false notices, also with English terms in Arabic |

## Setup, all clips

- **Microphone:** the laptop's built-in microphone. **Audio only**, no video.
- **Room and distance:** a quiet room, mouth 30–50 cm from the laptop. Keep the same place and
  the same Windows sound settings for all 20 clips.
- **How to speak:** from bullet notes, not a word-for-word script.
- **Ending:** end every clip with a short closing, slightly quieter than the rest: "Thank you."
  in English, «شكراً» in Arabic. This is the ADR 0015 check.
- **App:** Windows Sound Recorder (or any recorder). In its settings choose **WAV** as the
  recording format (FLAC also works). Avoid M4A or MP3: lossy formats change the clipping check.
- **Length:** 2–3 minutes of speech per clip. Up to about 5 s of silence before you start is
  fine.
- **Content:** your own topics. Leave out other people's names and personal details.

## Where the files go

Files stay **on your laptop only**: never GitHub, Google Drive, Kaggle, Colab or any other cloud.

- Save the recordings in:
  `C:\BAU\4-The Fourth Year\Graduation_Project\PresenceAI\data\b1\audio\`
  `data/` is git-ignored (`data/*` and `*.wav` in `.gitignore`), so git never picks them up.
  Make sure this folder is not inside a OneDrive or other synced folder. A copy on your own USB
  drive is fine if you want a backup.
- Name each file exactly as in the lists below. The tools stop on any other name.
- The app writes its annotation files to `data\b1\annotations\` and cached reports to
  `data\b1\reports\`, both on your laptop only.
- Only numbers go to the repository, in `experiments/results/b1/`. No transcript text, no audio.

## English, 10 clips

| # | File name | What to record | Checks |
|---|---|---|---|
| 1 | `b1_en_01_normal.wav` | Normal presentation on a general topic, your usual pace | Rate and pause baseline |
| 2 | `b1_en_02_normal.wav` | The same, another topic | Baseline |
| 3 | `b1_en_03_normal.wav` | The same, another topic | Baseline |
| 4 | `b1_en_04_normal.wav` | The same, another topic | Baseline |
| 5 | `b1_en_05_defense.wav` | Project-defense style: explain PresenceAI to a committee | Second scenario |
| 6 | `b1_en_06_defense.wav` | Project defense, another part (for example the evaluation, or answering a question) | Second scenario |
| 7 | `b1_en_07_slow.wav` | Slow, with long pauses (1–3 s). Make some of them mid-sentence on purpose, e.g. "the results show … that the rate" | ADR 0014 |
| 8 | `b1_en_08_fast.wav` | Faster than usual, but still clear | High rate band |
| 9 | `b1_en_09_fillers.wav` | Relaxed, with natural fillers ("um", "so", "like") and **one repeated word** ("I I think") | Fillers, repetitions |
| 10 | `b1_en_10_noise.wav` | Normal, with mild steady background noise: a fan or the AC. No music, TV or other voices | Noise robustness |

## Arabic, 10 clips

The same ten types, in Modern Standard Arabic or your usual presenting Arabic, plus:

- natural «يعني» and similar markers («طيب», «يعني») where they come naturally, in every clip;
- in clip 9, **one doubled «يعني يعني»**, next to the other fillers («امم», «اه»);
- **English terms mid-sentence in clips 2, 5 and 6** (for example *slides*, *feedback*,
  *dataset*): the code-switching rule and the language check;
- **long sentences in clips 3, 4 and 6**: several clauses joined with «و» or «ثم» before a full
  stop (ADR 0011: VAD minimum speech, `max_words_per_sentence_mark`).

| # | File name | What to record | Extra |
|---|---|---|---|
| 1 | `b1_ar_01_normal.wav` | Normal presentation, usual pace | — |
| 2 | `b1_ar_02_normal.wav` | Normal, another topic | English terms |
| 3 | `b1_ar_03_normal.wav` | Normal, another topic | Long sentences |
| 4 | `b1_ar_04_normal.wav` | Normal, another topic | Long sentences |
| 5 | `b1_ar_05_defense.wav` | Project defense: explain PresenceAI to a committee | English terms |
| 6 | `b1_ar_06_defense.wav` | Project defense, another part | English terms, long sentences |
| 7 | `b1_ar_07_slow.wav` | Slow, long pauses, some mid-sentence on purpose | — |
| 8 | `b1_ar_08_fast.wav` | Faster than usual | — |
| 9 | `b1_ar_09_fillers.wav` | Relaxed, natural fillers and one «يعني يعني» | Doubled marker |
| 10 | `b1_ar_10_noise.wav` | Normal, with a fan or the AC on | — |

## After recording: pre-fill

```powershell
uv run python experiments/b1_real_speech/run.py status
```

```powershell
uv run python experiments/b1_real_speech/run.py prefill
```

`status` lists the 20 clips and what is missing. `prefill` runs the app's own analysis
(whisper_small on the laptop; an estimate of a minute or two per clip, more when the laptop is
busy) and writes two files per clip into
`data\b1\annotations\`:

- `<clip>.yaml`: Whisper's transcript, the detected speech start and end, and the words that
  ADR 0015 removed at the end;
- `<clip>.labels.txt`: every detected pause as an Audacity label track.

`prefill` never overwrites a file you have corrected. You can record a few clips, pre-fill and
correct them, then record the rest.

## Correcting one clip

You only correct what the app pre-filled. **Aim for about 15–20 minutes per clip** (about 6–8
minutes for the transcript, 6–10 for the pauses, a few for the rest). All 20 clips take about
5–7 hours, best spread over several days. If a clip takes more than 30 minutes, stop, write why
in `notes:` and tell me: the format may need to change.

1. **Transcript** (`<clip>.yaml`, field `transcript`). Listen at normal speed and change the text
   to exactly what you said: add fillers Whisper dropped ("um", "uh", «امم», «اه»), add repeated
   words, fix wrong words, and delete words you did not say. Keep `whisper_text` as it is.
   Punctuation does not affect the word count.
2. **Speech start and end** (`speech_start_s`, `speech_end_s`). The first and last speech you
   hear, in seconds; the closing counts as speech. Only change them if they are off by more
   than about 0.2 s.
3. **Removed words** (`removed_at_end`). For each word, write `real: true` if you really said it
   (for example your closing) or `real: false` if Whisper invented it.
4. **Pauses** (`<clip>.labels.txt`). The easiest way is Audacity (free; a desktop tool, not a
   project dependency): open the audio, then **File ▸ Import ▸ Labels…** and choose the labels
   file. Each label spans one pause on the waveform.
   - Drag the label edges to where the silence really starts and ends. A breath belongs to the
     pause.
   - Set each label to `sentence` (after a finished sentence), `clause` (between parts of a
     sentence, where a comma would go) or `mid` (inside a phrase, e.g. "the results … show").
     `?` means the app could not tell; replace every `?`.
   - Delete labels that are not silence, and add pauses you hear that are missing, at least
     those of about half a second or longer.
   - Export with **File ▸ Export ▸ Export Labels…** over the same `<clip>.labels.txt` file.
     Without Audacity you can edit the file in a text editor: one line per pause,
     `start<TAB>end<TAB>label`, times in seconds.
5. **Done:** set `status: corrected` in the YAML file and save.

## After correcting

```powershell
uv run python experiments/b1_real_speech/run.py score
```

- `score` compares every corrected clip with the app's analysis and writes per-clip and
  per-language numbers to `experiments/results/b1/`. Clips still `prefilled` are listed and
  skipped.
- Run it first after 2 English and 2 Arabic clips, so we can check the workflow before you do
  the rest. Then run it again when all 20 are corrected.
- Tell me when a run is done. I summarise the numbers and propose updates to ADRs 0011, 0013,
  0014, 0015 and 0016. You decide; nothing is tuned to pass.
- Nothing from `data\b1\` is ever committed; only the number files in `experiments/results/b1/`
  are.
