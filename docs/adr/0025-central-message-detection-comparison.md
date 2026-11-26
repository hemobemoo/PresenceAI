# 0025 — Central-message detection: a fair comparison of methods on the owner's dev labels

- Status: **accepted** (protocol; owner, 2026-11-26), with additions 1–5 and item 10. Written
  **before any method is scored**. The method and its values are decided after the results, in
  an amendment to this ADR.
- Date: 2026-11-26
- Decider: project owner
- Plan reference: FINAL_04 §3 (central message), FINAL_10 E4, FINAL_12 P5 row (central-message
  cues tuned on the owner's dev labels before P5); ADR 0018 (text detectors), ADR 0019 open
  item 2 (floor effect), ADR 0020 (e5), ADR 0021 §4c (dev labels), ADR 0024 (unload models)
- Config keys affected: none yet. The values for the chosen method go into
  `configs/rubric.yaml` only with the amendment that decides them.

## Context

- **The current detector does not fit.** The owner labelled all 118 C2 dev talks
  (`experiments/results/c2/dev_labels.json`, detector at e21b0d1): main point stated in 90.
  The thesis-cue detector found one (tp 1, fp 0, fn 89, tn 28). One in 90 means the approach
  does not fit the material; it does not mean the phrase list is too short. Adding phrases until
  the owner's 90 sentences match would fit the generator's style, not presentations.
- **Where the main points sit** (`experiments/results/c2/dev_positions.json`, counts only,
  the app's region rule): of the owner's 90 sentences, **87 start in the opening 20 %**,
  2 in the middle, 1 in the closing 15 %. By start share: 77 in the first 10 %, 10 in
  10–20 %, 2 in 20–30 %, 1 in 70–80 %. The opening window is therefore not the cause of the
  misses. This is the generator's habit and may not hold for real talks, so methods still may
  not assume the main point is in the first sentence.

## Options considered (all scored the same way, below)

| | Method | Tuned in cross-validation | Laptop load |
|---|---|---|---|
| A | **Cue phrases v2**: thesis phrases taken from the open textbooks in the knowledge base (ADR 0020) and general claim patterns, **written and frozen before scoring**, without reading the owner's sentences or the dev texts | The opening window share only; the list is never tuned | spaCy only (seconds) |
| B | **Position only**: a sentence chosen by position in the opening (for example the first sentence after the hook window, or the opening sentence with the most content words) | Window share, rule choice | spaCy only |
| C0 | **Lemma centrality, no model**: the opening sentence whose content lemmas overlap most with the rest of the talk; "stated" when it beats the talk's median overlap by a margin | Window share, margin | spaCy only |
| C | **e5 centrality**: each opening sentence's cosine similarity to the talk, the talk being the mean of its sentence vectors (a whole talk exceeds e5's 512-token input) | Window share, margin | **Real-model run (e5)**: one embedding pass over the dev sentences, cached in git-ignored `data/c2/`; the owner is warned before it runs |
| D | The best of A / B / C0 combined with C, only if C beats both A and C0 | Weights, margin | As C (cache) |

No method may use paragraph breaks (C2 has them; transcripts do not). Positions are word shares,
as in the app.

## Decision (protocol; pre-registered)

1. **Design is blind.** The comparison code reports counts only. Nobody writing a method reads
   the owner's main-point sentences or the dev texts.
2. **Scoring on dev: 5-fold cross-validation grouped by topic** (24 dev topics, about 4–5 per
   fold, fixed seed). Parameters are tuned on four folds and scored on the fifth, rotating;
   counts are summed over the five held-out folds, with the spread across folds. The values the
   CV picks are reported, not adopted.
3. **Results table, one row per method**, with these columns side by side:
   - **headline: "stated with the owner's sentence"**, out of the owner's 90 "yes" (detector says
     stated and its sentence matches the owner's);
   - **false positives**, out of the owner's 28 "no";
   - the full confusion counts (tp, fp, fn, tn) for "stated", and "stated" with a wrong
     sentence.
   - "Always yes" is listed as a baseline row: it gets 90 of 118 "stated" right and 0 headline.
   - **Sentence match is strict** (the normalised owner sentence contained in the detector's
     sentence or the reverse, `dev_labels.py`). A detector sentence that states the same point
     in other words, or a neighbouring sentence, counts as a miss, so **the headline is a lower
     bound**.
4. **Pre-registered adoption rule for e5.** An e5 method (C or D) is adopted only if it beats
   the best spaCy-only method (A, B or C0) by **at least 10 percentage points on the headline
   (out of 90) with no more false positives**. Otherwise the best spaCy-only method is used and
   **S1 stays without e5**. Ties among spaCy-only methods go to the simpler one (B, then C0,
   then A).
5. **Closing restatement comes after** the main-point method is chosen: scored on the owner's
   36 "restated in closing" labels, comparing lemma overlap with the chosen sentence (the
   current rule) and, only if e5 was adopted, e5 similarity to the chosen sentence.
6. **If e5 is ever used in S1:** it is loaded only after Whisper has been unloaded, never both
   at once. The analysis order is transcription, unload Whisper, then e5, then unload e5
   (ADR 0024 one-analysis-at-a-time and unload rules).
7. **Owner consistency.** About 2026-12-03 the owner re-labels 20 dev items on a separate
   sheet, blind to the first labels (`dev_labels.py relabel`). Agreement on "stated", on the
   sentence (same strict rule) and on "restated" is reported as counts, with Cohen's κ marked
   unreliable at n = 20. It is the realistic ceiling for every method.
   - **Sampling rule (owner, 2026-11-26, decided before the draw):** stratified by the first
     "stated" label: **15 items the owner first marked "yes" and 5 first marked "no"**, each
     drawn at random within its stratum with the fixed seed 20261203, then shuffled together.
     15 / 5 matches the first sheet's 90 / 28 proportions, so agreement needs no reweighting.
   - **Reason:** an unstratified random draw of 20 (dry run in memory, nothing written or
     shown) held only 1 first-"no" item, which leaves agreement on "no" and κ almost
     meaningless. The seed was not changed to find a better sample. The relabel sheet shows no
     labels, so stratifying does not unblind the owner.
   - **Done 2026-11-26, about one day after the first labels** (not 2026-12-03; owner). Date
     and seed unchanged. With a one-day gap the agreement may be inflated by recall, so it is
     reported as an **upper bound on the owner's consistency**.
   - **Result** (`experiments/results/c2/dev_relabel.json`, counts only; κ unreliable at
     n = 20): main point stated, agreement **19 of 20** (both yes 15, both no 4, first no now
     yes 1; κ 0.86); the same sentence in **14 of the 15** both-yes items; restated in
     closing, agreement **14 of 20** (both yes 5, both no 9, first yes now no 1, first no now
     yes 5; κ 0.40). Consequences: the "stated" labels and sentences are consistent enough to
     be the reference for the main-point comparison. The "restated in closing" labels are
     much less consistent, so the restatement step (item 5) reads its numbers against this
     ceiling and does not tune a margin to them finely.
8. **The test part stays untouched** until the method is frozen in the amendment. E4 handling
   is in ADR 0021 §4d.
9. **Language.** Central message stays English-only. Arabic central message is **still not
   measured** and is shown as "not measured" with its reason (ADR 0017, ADR 0019).

10. **If a position-only method (B) wins** (owner, 2026-11-26): the amendment states that the
    result reflects how the C2 generator places main points (87 of 90 in the opening,
    `dev_positions.json`), and the method is **not trusted in the app until it is confirmed on
    B1 and on the E4 public talks**.
11. **Pre-registered details for A, B and C0** (written 2026-11-26, before any of them was
    scored; committed before `cm_compare.py`):
    - **Candidates:** sentences of the app's sentence split (`text.document.build`, spaCy) that
      start in the opening window W % of the words. Questions (ending in "?") are never
      candidates: a thesis is not a question (textbook guideline, `cm_cues_v2.yaml`).
    - **Grids.** All methods: W in {10, 15, 20, 25, 30}.
      A: families in {purpose}, {purpose, preview}, {purpose, preview, claim}; pick the first
      or the last matching candidate (the textbooks place the thesis toward the end of the
      introduction). Purpose and claim are the phrase lists in `cm_cues_v2.yaml`. Preview is
      a number word or digit 2–5 followed by a plural noun, or a sentence with a colon or at
      least two commas followed by "and" (a list).
      B: the 1st, 2nd or 3rd candidate, or the last one. B always says "stated" when the window
      has a candidate.
      C0: score(s) = share of the sentence's content-lemma types (NOUN, PROPN, VERB, ADJ, not
      stop words) that occur in at least one other sentence of the talk. The candidate with
      the highest score wins (ties: the earlier one). "Stated" if that score minus the median
      score over all sentences of the talk is at least m, m in {0.00, 0.05, …, 0.40}.
    - **Tuning objective** on the training folds: correct items = the owner's "yes" items whose
      chosen sentence matches the owner's sentence, plus the owner's "no" items with no
      sentence chosen. Ties go to fewer false positives, then to the first entry in the order
      listed above.
    - **Folds:** the 24 dev topics (the number at the end of each item id) shuffled with seed
      20261126 and dealt round-robin into 5 folds.
    - **Reference rows:** "always yes" (no sentence, so headline 0) and the current detector
      (thesis cue v1, `lang/en/text_cues.yaml`), both scored on all 118 with no tuning.

## Consequences

- Order of work: `relabel` tool; comparison harness with A, B, C0 (spaCy only); e5 cache and
  C, D (owner warned first); results table; the owner decides method and values (amendment,
  config keys); restatement; then the E4 test sample (ADR 0021 §4d).
- What would show the decision wrong: the frozen detector's headline on the E4 test sample
  falling far below its cross-validated dev headline, or the owner's own consistency being
  close to the detector's agreement (then the labels, not the detector, limit the result).
- C2's main points sit almost all in the opening; real talks may differ. The B1 talks and E4's
  public talks are the check.
