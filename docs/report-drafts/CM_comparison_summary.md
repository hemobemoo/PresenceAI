# Central-message detection: a pre-registered comparison with a negative result

## Aim

The comparison tested whether a rule-based or embedding-based method could detect, in a short
presentation transcript, whether the main point is stated and which sentence states it, well
enough to be scored in the rubric's central-message dimension. The existing thesis-cue detector
had placed all 12 real English pilot talks at level 1 (ADR 0019 open item 2), which suggested a
measurement problem rather than a speaker problem.

## Method

- **Labels.** The project owner labelled all 118 development transcripts of the synthetic set
  C2 on a blinded sheet in one shuffled order across the five planted groups
  (`experiments/c2_synth/dev_labels.py`): whether the main point is stated, the sentence that
  states it, and whether it is restated in the closing. The main point was marked as stated in
  90 transcripts and not stated in 28. Of the 90 sentences, 87 start within the first 20 % of
  the words (`experiments/results/c2/dev_positions.json`).
- **Label consistency.** Twenty items were relabelled blind, 15 first marked "stated" and 5
  first marked "not stated", about one day after the first labels. Agreement was 19 of 20 on
  "stated", 14 of 15 on the sentence, and 14 of 20 on the restatement (Cohen's κ 0.86 and
  0.40, unreliable at n = 20; `dev_relabel.json`). Because of the one-day gap, these figures
  are an upper bound on the owner's consistency.
- **Methods.** A: thesis-statement phrases taken from open public-speaking textbooks in the
  project's knowledge base, written before scoring. B: sentence position only. C0: lemma overlap
  with the rest of the talk. C: similarity of each sentence to the whole talk with the
  multilingual-e5-small embedding model. D: A combined with C.
- **Scoring.** Five-fold cross-validation grouped by topic (24 topics, seed 20261126):
  parameters were tuned on four folds and scored on the fifth, and counts were summed over the
  held-out folds. A detected sentence counted as correct only if it contained, or was contained
  in, the owner's sentence after normalisation; the right-sentence count is therefore a lower
  bound.
- **Pre-registration.** The protocol (ADR 0025 items 1–10) was accepted before any method was
  scored. The method A phrases, parameter grids, tuning objective and folds (item 11) were
  committed before the scoring code (commit ee94676). After the spaCy-only results, and before
  the e5 methods were scored, the owner changed the ranking (item 12, commit d19f501) because
  the original headline rewarded calling every talk "stated". In the same commit the agent
  also changed the tuning objective to match the new ranking; the owner had not asked for that
  change, and ADR 0025 records it as the agent's.

## Pass conditions

Set before the e5 methods were scored (ADR 0025 item 12):

- **Ranking:** the average of the right-sentence rate (out of 90) and the correct
  "not stated" rate (out of 28).
- **Minimum bar for any method:** at least 45 of 90 right sentences and at most 7 of 28 false
  positives.
- **e5 adoption:** an e5 method's average at least 0.10 above the best method without e5,
  with at most 7 false positives.

## Results

Held-out counts, `experiments/results/c2/cm_compare_v2.json`:

| Method | Average | Right sentence /90 | False positives /28 |
|---|---|---|---|
| D: A, or C when A finds nothing | 0.651 | 40 | 4 |
| C: e5 similarity to the talk | 0.626 | 42 | 6 |
| A: textbook phrases | 0.577 | 17 | 1 |
| Existing thesis-cue detector (no tuning) | 0.506 | 1 | 0 |
| C0: lemma overlap | 0.493 | 2 | 1 |
| B: position only | 0.417 | 14 | 9 |
| Always "stated" (no sentence) | 0.000 | 0 | 28 |

Under the original tuning objective, B named the right sentence in 67 of 90 talks but called
27 of the 28 "not stated" talks "stated" (`cm_compare.json`). No method reached the minimum
bar, and no e5 method met the adoption rule. On the laptop, the e5 run took 1.6 s to load the
model and 20.7 s to embed 2,435 sentences.

When method A finds a sentence, it is usually the owner's: on the held-out folds A found a
sentence in 19 talks, and 17 of them were the owner's sentence (89.5 %; Wilson 95 % interval
about 0.69–0.97).

## Decision

Decided by the project owner on 2026-11-26 (ADR 0025):

- The central-message dimension is information-only in both scenarios until a language-model
  judge is built in phase P7: it is shown, not scored.
- The application shows "Possible main point: …" only when method A finds a sentence, with
  low confidence, and says nothing otherwise. The owner set an 80 % precision condition for
  showing it, which A met on the held-out folds.
- S1 analysis does not load the e5 model.
- The E4 test sample of 60 transcripts is drawn only after the judge is frozen, and the judge
  is scored once on it.

## Limitations

- The transcripts are synthetic and come from a single generator, which places the main point
  early (87 of 90 in the opening). Position-based results reflect that habit and may not hold
  for real talks.
- One person labelled the transcripts. The relabel agreement is an upper bound because of the
  one-day gap, and the restatement labels are much less consistent than the main-point labels.
- The strict sentence match counts paraphrases and neighbouring sentences as misses, so every
  right-sentence count is a lower bound.
- The precision of method A rests on 19 cases; the lower end of its interval is below 80 %.
- Only English was studied. Arabic central message is not measured.

## Sources

- `docs/adr/0025-central-message-detection-comparison.md`, `docs/adr/0021-c2-synthetic-transcripts.md` §4c–4d, `docs/adr/0019-rubric-scoring-and-assessment.md`
- `experiments/results/c2/dev_labels.json`, `dev_positions.json`, `dev_relabel.json`, `cm_compare.json`, `cm_compare_v2.json`
- `experiments/c2_synth/dev_labels.py`, `cm_compare.py`, `cm_cues_v2.yaml`; commits ee94676, d19f501, 0378a3d
