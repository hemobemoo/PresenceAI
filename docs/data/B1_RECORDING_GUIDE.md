# B1 recording guide (owner only)

B1 is the pilot set: **20 clips of your own speech, 10 English and 10 Arabic, 2–3 minutes
each** (ADR 0010, ADR 0016). It is the only real-speech check for several decisions that were
made on synthetic speech (C1):

| Check | What B1 shows |
|---|---|
| E2 Arabic (FINAL_10 §2) | Rate and pause positions in Arabic, reported descriptively for one speaker. Pause detection and length are not checked (pause edges are not checked by ear, ADR 0016) |
| E2 English on real speech | Rate and pause positions in English, next to the C1 result |
| E1 | Word error rate on your 20 hand-corrected clips |
| ADR 0015 | Whether the 0.2 s end margin removes real words, especially your quiet closing |
| ADR 0014 | Whether mid-sentence pauses are labelled correctly in real speech; the true position comes from your punctuation |
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
- **App:** Windows Sound Recorder; nothing else to install. In its settings:
  - set the recording format to **WAV**. Avoid M4A or MP3: lossy formats change the clipping
    check;
  - set the folder for recordings to `data\b1\audio\` (below). The default folder,
    `Documents\Sound recordings`, may be synced by OneDrive. If the folder cannot be changed,
    move each file to `data\b1\audio\` right after recording and delete it from
    `Documents\Sound recordings`.
  - Rename each recording to its name in the lists below (Sound Recorder or File Explorer).
- **Length:** 2–3 minutes of speech per clip. Up to about 5 s of silence before you start is
  fine.
- **Content:** your own topics. Leave out other people's names and personal details.

## Where the files go

Files stay **on your laptop only**, as plain WAV files (ADR 0016, open item 3): never committed to
GitHub, never synced to OneDrive or Google Drive, never uploaded to Kaggle, Colab, Hugging Face or
any other cloud. The annotations and cached reports follow the same rule. Windows device
encryption is optional.

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
- `<clip>.labels.txt`: every detected pause, one line each (`start<TAB>end<TAB>label`, times
  in seconds). All labels start as `?`; `label` fills them from your transcript.

`prefill` never overwrites a file you have corrected. You can record a few clips, pre-fill and
correct them, then record the rest.

## Correcting one clip

You correct the transcript by listening; the pause labels then come from your punctuation, so
you never edit pause times. **Aim for about 15 minutes per clip**: about 8–10 minutes for the
transcript and its punctuation, and a few minutes for the rest and any `?`. All 20 clips take
about 5 hours, best spread over several days. If a clip takes more than 30 minutes, stop, write
why in `notes:` and tell me: the format may need to change.

1. **Transcript** (`<clip>.yaml`, field `transcript`). Play the clip in Sound Recorder and
   change the text to exactly what you said:
   - add fillers Whisper dropped ("um", "uh", «امم», «اه») and repeated words;
   - fix wrong words, and delete words you did not say;
   - keep `whisper_text` as it is.
2. **Punctuation, by meaning.** Put `.` `?` `!` `؟` where a sentence really ends and `,` `،`
   `;` `؛` where a clause really ends: where a reader needs them, **not where you paused**. A
   pause inside a phrase ("the results … show") gets no mark. Punctuation does not change the
   word count.
3. **Speech start and end** (`speech_start_s`, `speech_end_s`). The first and last speech you
   hear, in seconds (Sound Recorder shows the position); the closing counts as speech. Only
   change them if they are off by more than about 0.2 s.
4. **Removed words** (`removed_at_end`). For each word, write `real: true` if you really said it
   (for example your closing) or `real: false` if Whisper invented it.
5. **Pause labels.** Save the YAML file, then run:

   ```powershell
   uv run python experiments/b1_real_speech/run.py label b1_en_01_normal
   ```

   (your clip's name, or `--all` for every pre-filled clip). It reads your punctuation after
   your last word before each pause and writes `sentence`, `clause` or `mid` into
   `<clip>.labels.txt`. It never changes pause times and never adds or deletes pauses.
   - When it cannot tell, it writes `?` and prints the pause number, its time and 3 words on
     each side, for example `pause 7 at 01:12.40-01:13.30: ... the main result | امم is ...`.
     This happens when a word you added (such as «امم») has no timing and sits right at the
     pause.
   - Listen to that pause (3 s before to 1 s after it):

     ```powershell
     uv run python experiments/b1_real_speech/run.py listen b1_en_01_normal 7
     ```

     Then either fix the punctuation in the transcript and run `label` again, or open
     `<clip>.labels.txt` in Notepad and replace that `?` (line 7 is pause 7) with `sentence`,
     `clause` or `mid`. Change only the label at the end of the line, never the two times.
   - Once a file has no `?` left, `label` keeps it; `--force` redoes it from your punctuation.
6. **Done:** set `status: corrected` in the YAML file and save. Comments you add to the YAML
   file are not kept when `label` rewrites it, so put remarks in `notes:`.

Pause edges are not checked by ear (`pause_edges_checked: false`), so B1 measures pause
*positions* but not pause detection or pause length (ADR 0016, "Amendment").

## After correcting

```powershell
uv run python experiments/b1_real_speech/run.py score
```

- `score` compares every corrected clip with the app's analysis and writes per-clip and
  per-language numbers to `experiments/results/b1/`. Clips still `prefilled` are listed and
  skipped, and so is a clip whose labels were not made by `label`.
- It reports rate, WER, pause positions, removed words, clipping and the language check. Pause
  detection (F1) and pause length are left out and the clips are listed under
  `pause_edges_not_hand_checked`, because nobody moved the pause edges by ear.
- Clips you pre-filled before 2026-11-19 have labels taken from Whisper's punctuation. `label`
  replaces those with labels from your transcript; you do not need `--force` for them.
- Run it first after 2 English and 2 Arabic clips, so we can check the workflow before you do
  the rest. Then run it again when all 20 are corrected.
- Tell me when a run is done. I summarise the numbers and propose updates to ADRs 0011, 0013,
  0014, 0015 and 0016. You decide; nothing is tuned to pass.
- Nothing from `data\b1\` is ever committed; only the number files in `experiments/results/b1/`
  are.
