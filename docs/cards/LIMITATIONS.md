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
- **C1 cannot validate mid-sentence pause labels.** In E2, 58.9 % of true mid-sentence pauses
  were labelled as a sentence or clause boundary, mainly because of how C1 is built (above).
  The error is lenient: boundary pauses, which never count against the speaker, were labelled
  mid-sentence in only 5.1 % of cases, so the app rarely penalises a pause it should not, but it
  misses many real mid-sentence pauses. How well mid-sentence pauses are found in real speech
  is checked only on the developer's B1 recordings and on annotated real data (ADR 0014). On
  B1 the true position comes from the developer's own punctuation, one speaker.
- **C1 fillers are not used for filler detection (E3).** A TTS voice saying "um" on its own is
  not a real hesitation; E3 uses DisfluencySpeech, AMI and PodcastFillers.
- The noise condition uses synthetic pink noise, not real room or babble noise, and has no pass
  target.

## Invented words (speech recognition)

- **Base Whisper sometimes writes words that were never said**, mostly in silence at the end of
  a recording (C1 clip s10_r0: "Thank you very much for joining us. Thank you."). Words that
  start after the last detected speech are removed before anything is measured or quoted
  (ADR 0015). Three kinds are **not caught**:
  - **invented text at the start of a recording.** An earlier rule also removed words before
    the first detected speech, but on C1 it removed real first words: Whisper sometimes squeezes
    the first word into 0.00–0.22 s although speech starts later. So words at the start are
    kept;
  - an invented word that starts inside detected speech (in s10_r0, the first "Thank" begins
    0.02 s before speech ends, so it stays);
  - stray invented words in the middle of a recording (in s10_r0, a stray "and" and "I").

  These still count towards speech rate and can appear in quotes.
- **The 0.2 s margin is provisional and test-informed.** It rests on synthetic speech, and the
  change to "end only" was made after looking at the C1 test part, so C1 is no independent check
  of it. On real recordings, VAD can miss a quiet last word, and the rule would then remove a
  word that was really said. It is re-checked on the owner's recordings (B1).

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
  ratio. This is not yet checked on real public clips. The first real-speech check runs on
  the developer's B1 recordings, one laptop microphone only (ADR 0016).
- Pitch is tracked between 65 and 600 Hz. Very low or very high voices may reach the limits; the
  report gives the share of frames at a limit and lowers confidence when it is high.

## Language and Arabic text (ADR 0016)

- The language check uses Whisper's detection on the first 30 s of speech. A notice appears
  when another language is detected with probability ≥ 0.95. The value is provisional and was
  chosen after seeing the developer's B1 recordings: at 0.5, 4 of 10 English clips were
  detected as Arabic (probability up to 0.94). **Speech in English analysed as Arabic is rarely
  caught**, because accented English is detected as English with only about 0.5–0.8. The
  value rests on one speaker and has not been checked on independent accented speech. The
  notice never changes the analysis.
- The Arabic report texts are a draft translation, not yet reviewed by a native speaker.
- **Pause detection and duration have no real-speech check, and Arabic pause timing has no
  check at all.** B1's pause times are the app's own; they were not moved by ear, so B1
  reports pause positions (from the developer's transcript punctuation) but not pause F1 or
  duration error. English pause timing is checked only on synthetic speech (C1), and no
  public Arabic source has timing ground truth (ADR 0016 "Amendment").
- B1 has 20 clips of one speaker and one microphone (10 English, 10 Arabic), so its results
  describe that speaker and are reported descriptively.
- **B1 English pause positions and English WER rest on Whisper's own text.** The developer
  corrected Whisper's transcript and confirmed its punctuation and words where unchanged, so
  English WER (0.8 %) is a lower bound and English position agreement is not independent.
  The independent English evidence is the 135 pauses decided by ear (73 % agreement).
- **B1 does not test noise robustness.** The two "mild steady noise" clips measured about
  60 dB SNR, the same as the quiet clips. They were not re-recorded.
- **Arabic pause positions are not available.** Whisper small writes almost no Arabic
  punctuation, so positions are `unknown` on 9 of 10 Arabic B1 clips (ADR 0011).

## Content measures (text detectors, ADR 0018)

- **Content measures are rule-based and English only.** Thesis, introduction, conclusion and
  transition cues are lexicon matches, and hooks and supporting material are pattern rules: they
  do not understand meaning. A clear talk that avoids the listed phrases scores lower than it
  should. The lexicons are provisional; their accuracy is measured in E4 on C2.
- **A Whisper misrecognition can appear as an undefined rare term.** A word Whisper mishears
  into a rare word (or a rare spelling) is counted as jargon without a definition.
- **Sentences come from Whisper's punctuation.** With unreliable punctuation, sentence length is
  not measured and the other content measures have low confidence.
- **Rhetorical quality and persuasiveness are not scored.** The rubric counts observable
  features; it does not credit a well-made sentence and can count a deliberate choice against
  the speaker: a pause before a short closing line is a mid-sentence pause unless Whisper puts
  a full stop there, and one well-chosen unusual word counts as an undefined term. Proposed
  fixes are open items for the threshold review after E5/E6 (ADR 0017 open item 4).
- **Follow-up relevance does not catch evasive answers.** It is the similarity of the question
  and the answer's words. On the developer's 20 calibration pairs, the 5 evasive answers (which
  repeat the question's words without answering) scored above every good answer but one, so
  relevance only separates off-topic answers, and even there the highest off-topic answer
  (0.842) scored just above the lowest good one (0.841). The provisional cut (0.82) flags only
  clearly off-topic answers: 3 of 5 off-topic and 0 of 5 evasive answers in calibration. The
  pairs were written, not spoken; the cut is re-checked on real spoken answers. Evasion is left
  to the optional language-model judge (P7). Relevance is English only (ADR 0022).
- **Hedging is not measured** (deferred): it is not used by the rubric and is the measure
  closest to a confidence-style judgement.

## Synthetic presentation transcripts (C2, ADR 0021)

- **C2 transcripts are written by a language model (Qwen3.5-9B) and contain invented facts**:
  made-up statistics, studies, "famous cases" and quotations. C2 labels describe the talk's
  structure (central message, conclusion, supporting material, unexplained terms, hook), never
  whether a fact is true, so C2 measures structure detection only.
- **C2 transcripts are always labelled "synthetic"** wherever they are shown: thesis text,
  figures, tables, the demo and any example in the app. They are never presented as a real
  student's talk.
- **C2 labels are the planted design and are unverified, except for the hand-checked
  sample.** Transcripts are kept on length, format and echo checks only; no check confirms that
  a planted defect is really there (the model's own yes/no check proved unreliable and is not
  used). The owner's blinded hand check of 50 test transcripts gives the label-noise estimate,
  and E4 is also reported on that subset against the owner's labels (ADR 0021).
- **The planted labels are noisy.** In the hand check, 27 of 50 transcripts (54 %) had at
  least one planted label the owner disagreed with. Per label: central message 30 %,
  supporting material 16 %, conclusion 10 %, unexplained terms 4 %, hook 0 %. The planted
  defect was confirmed in all 10 no-central-message, no-conclusion and undefined-jargon talks,
  but **only 2 of 10 "no supporting material" talks really had none**: the owner found
  supporting material in the other 8. E4's supporting-material and central-message results on
  the planted labels are therefore unreliable; the hand-checked subset (50 items, one rater)
  is their reference.
- **One generator, one style.** All 400 transcripts come from one model with one prompt
  template, so E4 on C2 says nothing about other writing styles or real speech; public talk
  transcripts (TED, E4/E5) are the other check.

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
