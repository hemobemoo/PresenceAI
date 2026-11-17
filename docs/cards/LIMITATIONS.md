# Limitations

Started 2026-11-17. Kept up to date as experiments report (FINAL_11 §6).

## Evaluation design

- **No multi-participant user study.** PresenceAI was not evaluated with a group of real learners
  (ADR 0010). Evidence for the adaptive engine (C1) comes from a simulator whose measurement noise
  is set from public-data results (E9), from agreement with two expert raters on public talks (E13),
  and from a single-case ABAB self-study by the developer (E14). None of these shows that the app
  improves real learners' presentations. **Validation with real learners is future work.**
- **The single-case study is n = 1,** and the participant is the developer, who knows how the
  engine works. Its results are descriptive and do not generalise.
- **The expert raters are two:** the developer and one classmate. Neither is a professional
  speaking coach, and agreement between them limits what agreement with the engine can mean.
- **Simulated learners** follow the simulator's assumptions about learning, transfer and
  forgetting (FINAL_05 §8). Conclusions hold only within those assumptions; sweeps over wrong noise
  assumptions show how sensitive they are.

## Data

- Measurement validity (C2) is checked on public recordings (`docs/data/DATA_SOURCES.md`). These are
  meetings, podcasts, read speech and published talks, not novice practice presentations recorded
  on a laptop webcam, so error rates in the app may differ.
- Non-public speech and video come only from the developer's own recordings (B1, B2). The
  per-condition vision robustness results describe one person.
- **Gaze (T3) is trained on one person** (the developer's own protocol recordings) and **tested
  across people only on Columbia Gaze**. Columbia Gaze contains posed still images taken under
  controlled conditions, not webcam video of people presenting. Its known gaze angles are mapped to
  the app's zones by a fixed rule. **The cross-person accuracy reported in E7 is therefore an
  approximation** of how T3 would do on other users' webcams. It may be optimistic (posed, sharp,
  frontal images) or pessimistic (no per-user calibration, a camera setup unlike a laptop). The AMI
  focus-of-attention check is meeting video, not presentations, and is only a sanity check.
- **Arabic rate and pause accuracy (E2) has no public ground truth.** None of the allowed Arabic
  sources (Common Voice Arabic, MASC) has word-level timing or pause labels, and the synthetic
  generator (Kokoro) has no Arabic voice. Arabic E2 is measured only on the developer's own
  annotated recordings, one speaker, and is reported descriptively. Arabic speech recognition (E1)
  is covered by Common Voice Arabic and MASC (`docs/data/DATA_SOURCES.md` §3).
- Research-only and no-derivatives sources (TED, TED-LIUM 3, PodcastFillers, Columbia Gaze) are used
  for evaluation only, which limits training data.

## Synthetic speech (C1, ADR 0014)

- **E2's rate and pause targets are checked on synthetic English speech** from one TTS model
  (Kokoro, six voices). Real speakers vary more; TED-LIUM 3 gives only an approximate check on
  real speech, and Arabic E2 uses only the developer's own recordings.
- **Mid-sentence pauses in C1 are built by joining separately generated phrases**, so the phrase
  before such a pause may sound like a phrase end rather than a hesitation, and Whisper may put a
  full stop there. E2's pause-position report shows how often this happens.
- **C1 fillers are not used for filler detection (E3).** A TTS voice saying "um" on its own is
  not a real hesitation; E3 uses DisfluencySpeech, AMI and PodcastFillers.
- The noise condition uses synthetic pink noise, not real room or babble noise, and has no pass
  target.

## Loudness and pitch

- **Browser audio processing.** Auto-gain, noise suppression and echo cancellation flatten
  loudness. Live capture turns them off (P6), but for uploaded files they are unknown, so loudness
  variation and baseline comparisons have low confidence (ADR 0013).
- **Loudness variation is always low confidence on uploaded files.** This includes all
  public-video and public-audio evaluation data, because their capture settings are unknown.
  Evaluation results for loudness variation therefore describe low-confidence measurements only.
- **A change of microphone or distance cannot be detected**, so a change from your own baseline
  is at most medium confidence even when the capture settings were off.
- **Octave errors.** The pitch tracker can report half or double the real pitch. Such frames are
  folded back or dropped; the same rule can also remove a real jump of more than 9 semitones
  within half a second. The report counts how many frames were changed.
- **Clipping in compressed files** is found by a plateau rule whose values come from synthetic
  signals and are provisional. It is reported apart as "possible clipping", always low
  confidence. In synthetic tests, loud limited (mastered) audio stayed well below the rule's
  ratio. This is not yet checked on real public clips.
- Pitch is tracked between 65 and 600 Hz. Very low or very high voices may reach the limits; the
  report gives the share of frames at a limit and lowers confidence when it is high.

## Fillers and repetitions

- **Until the verbatim model T1 exists (P4), filler and repetition counts are lower bounds**,
  reported as "at least N per 100 words". Base Whisper leaves out many fillers and repetitions:
  on a public 3-minute clip it found 0.86 fillers per 100 words against a true rate of about 3.0,
  and even adding the acoustic candidates (1.54) stayed far below it. Confidence is capped at
  medium for English and low for Arabic (ADR 0012).
- **False starts (cut-off words) are not measured** until T1, because they need a verbatim
  transcript.
- **Whisper's word timings can hide fillers from the acoustic detector.** A hesitation directly
  next to a word is missed when Whisper stretches that word's timestamps over it. On a public clip
  whisper-small did exactly this; turbo did not. E3 reports how often it happens.
- The acoustic detector's values are provisional until they are tuned on DisfluencySpeech
  train/validation data and held-out AMI sessions.
- Discourse markers ("like", "so", "يعني", ...) are counted every time they occur, including
  ordinary uses, so their counts are upper bounds and information only.
