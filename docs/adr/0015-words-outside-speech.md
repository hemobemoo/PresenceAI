# 0015 — Remove words Whisper invents before the first or after the last speech

- Status: **accepted** by the owner on 2026-11-18 (plan and values approved as proposed, plus
  additions 1–3 below). `edge_tolerance_s` is **provisional**.
- Date: 2026-11-18
- Decider: project owner
- Plan reference: AGENTS.md rule 4 (evidence comes from the transcript); FINAL_04 §2 (rate);
  ADR 0011 (VAD, rate); ADR 0014 §6 (tuning only on the C1 development part)
- Config keys affected: `configs/default.yaml: speech.transcript.edge_tolerance_s` (new)

## Context

Base Whisper (whisper_small, faster-whisper) sometimes writes text into silence, most often at
the end of a recording. In E2 on C1 clip s10_r0 (development part) it added "Thank you very much
for joining us. Thank you." after the last sentence (`experiments/results/e2/f1_dev_findings.md`).
Such words raise `rate.wpm`, because they are counted while speaking time ends at the last VAD
offset. They could also be quoted back as the user's own words, which breaks AGENTS.md rule 4.

## Options considered

Measured on the 86 C1 development clips, using cached transcripts and the app's VAD
(`experiments/e2_rate_pauses/outside_speech_dev.py` →
`experiments/results/e2/f1_dev_outside_speech.csv`). In the table, "real" means a word that
difflib aligns to the clip's script, and "extra" means a word that is not in the script.
Baseline: 185 extra words, mean wpm error 1.070 %.

| Rule (a word is kept when …) | tol (s) | real removed | extra removed | mean wpm error |
|---|---|---|---|---|
| **ends**: its span touches [first onset − tol, last offset + tol] | 0.0 | 3 | 9 | 0.989 % |
| ends | 0.05 | 1 | 8 | 0.983 % |
| ends | **0.1–0.5** | **0** | **8** | 0.986 % |
| regions: its span touches some region ± tol (pauses checked too) | 0.1 / 0.3 | 13 / 4 | 10 / 10 | 1.073 / 1.018 % |
| mid_ends: its midpoint lies in [first − tol, last + tol] | 0.2 / 0.3 | 50 / 2 | 11 / 9 | 1.309 / 0.983 % |

1. **Ends only (chosen).** No real words were lost at 0.1–0.5 s. All 8 removed words are in
   s10_r0's tail, which Whisper squeezed into 56.42–56.44 s, after the last VAD offset at 55.90 s.
2. **Also check pauses between regions.** This costs real words, because Whisper's word times
   drift into pauses, and it removes only 2 more extra words.
3. **Midpoint rules.** These fail at the start of a recording: Whisper stretches the first word
   back to 0.0 s while speech starts at 0.5 s.
4. **faster-whisper `hallucination_silence_threshold`** (1.2.1; requires `word_timestamps=True`,
   which is already on). Not adopted, and not measured, because it needs re-transcription. The
   code (`transcribe.py`, near "skip silence before possible hallucinations") removes a suspect
   segment only when there is more than the threshold of silence **before** it, counted from
   the last word end, and after it. Whisper starts s10_r0's invented segment 0.02 s after the
   last real word, so the option would most likely miss it. It would also change decoding for
   every recording and make all cached transcripts unusable.

## Decision

- `speech/trim.py: trim_to_speech` runs in `speech/analyze.py` right after VAD and before any
  measure. It removes each word that ends before `first onset − edge_tolerance_s` or starts
  after `last offset + edge_tolerance_s`. The kept words get a rebuilt `Transcript.text` and
  character offsets, so no evidence quote can contain a removed word. Segments shrink to the
  words they still hold, and segments left with no words are dropped.
- `speech.transcript.edge_tolerance_s: 0.2`. This is the middle of the 0.1–0.5 s range that
  cost no real words, which leaves room for timestamp drift on real speech. **It is
  provisional**, because it rests on one example in synthetic speech.
- **No speech regions → every word is removed.** The report then says "no speech detected in
  this recording" (`not_measurable["transcript"]`), not just an empty transcript.
- `SpeechReport` becomes schema version 5, with `words_outside_speech`: the removed words with
  their original times. They are kept for transparency and are never measured or quoted.
  `asr_quality` is computed on the trimmed transcript.
- The E2 runner trims the same way. The setting is part of the E2 config hash, so this is a new
  E2 config version with its own single clean test run (ADR 0014).

## Consequences

- **Re-check on B1 (owner's real recordings)** by counting real words removed. On real speech,
  VAD can miss a quiet last word, such as a trailing "thanks" or a dropped final syllable, and
  this rule would then remove it. If any real word is removed, raise the tolerance or rethink
  the rule, and record that here.
- **Not caught:**
  - an invented word that starts inside speech (the first "Thank", 55.88–56.42 s, in s10_r0
    stays);
  - stray invented words in the middle of a recording ("and", "I" in s10_r0).

  Both are listed in `docs/cards/LIMITATIONS.md`. s10_r0's wpm error is still 9.6 % after the
  fix.
- A live or streaming path that uses Whisper later must apply the same rule before quoting.
- What would show this decision is wrong: B1 or TED-LIUM clips where a removed word is in the
  reference transcript.

## Observed in the E2 test run (2026-11-18)

Config `35df562d1acd331a`, results in `experiments/results/e2/f1_adr15_results.md`. The clean
test part passes: wpm error 0.94 %, pause F1 0.970 / 0.982, duration error 0.076 s. The rule
also works as intended at the end of a recording: s17_r0's invented "… see you in the next
video." was removed.

At the start of a recording, however, the rule removed real first words: "When" in s05_r4 in
all three conditions, plus "A" and "We've" at 20 dB. Whisper sometimes squeezes the real first
word into 0.00–0.22 s while speech starts at about 0.5 s. This did not happen on the
development part, so the ADR's "0 real words" holds only there.

How to change the rule is the owner's decision. A change made now would be informed by the
test part (ADR 0014), so this is recorded here and the rule is unchanged.
