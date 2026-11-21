# 0018 — English text detectors for the rubric's content measures (P2-b)

- Status: **accepted** by the owner on 2026-11-21 (P2-b plan and answers 1–5). The cue lexicons
  are **provisional**: the owner reviews them in the PR, and they are adjusted on the C2
  development part only (P2-e).
- Date: 2026-11-21
- Decider: project owner
- Plan reference: FINAL_04 §3 (content measures), FINAL_05 §1 (skills), FINAL_08 (`text/`),
  FINAL_10 E4 (content detectors), ADR 0017 (rubric, definitions, thresholds)
- Config keys affected: none new. The detectors read `configs/rubric.yaml`
  (`opening_share_pct`, `closing_share_pct`, `hook_sentences`, `lang_rare_zipf`, all
  provisional, ADR 0017); `configs/models.yaml: nlp.spacy_en` (new URL, digest, unpack folder)

## Context

The S1 rubric (ADR 0017) needs eleven content measures that did not exist: central message,
organization, hook, supporting material and language. They must carry evidence (AGENTS.md rule
4), run on the laptop, and their tests must run without models.

## Decision

1. **Parser.** spaCy `en_core_web_sm` 3.8.0 (MIT) with `spacy` 3.8.x (MIT) (owner, answer 1).
   The wheel is fetched only by `scripts/download_models.py --only spacy_en` from spaCy's GitHub
   release (redirects allowed only to GitHub's release-asset hosts), verified by SHA-256
   (`1932429d…fb85`; GitHub lists no digest for this release, so it was pinned on the first
   download; the size matches GitHub's, 12,806,118 bytes), and its model folder is unpacked into
   `models/en_core_web_sm-3.8.0/` and loaded with `spacy.load(path)`. Nothing is pip-installed
   at runtime. The detectors see only a small `Tok` interface (`text/nlp.py`); tests use a fake
   parser (`tests/fixtures/fake_nlp.py`). On the sample talk the real model and the fake give
   the same eleven values (slow test).
2. **Word frequency.** wordfreq 3.1.1: code Apache-2.0; data CC BY-SA 4.0 with SUBTLEX,
   OpenSubtitles and Google Books Ngrams terms. It is used at runtime only and not redistributed
   separately; NOTICE carries the credits and the citation (owner, answer 2).
3. **Sentences and regions.** Sentences come from the transcript's punctuation
   (`speech.pauses.sentence_marks`); opening, closing and body are word shares (ADR 0017
   definitions). A sentence is in the opening when its first word is, in the closing when its
   last word is.
4. **Cues** (`lang/en/text_cues.yaml`, provisional): thesis, introduction, conclusion,
   transitions (counted only within the first 3 words of a sentence, once per sentence),
   definition, example, analogy, quotation, story, contrast lead and turn. Matched on whole words,
   ignoring case and punctuation. **Lexicon matches, not an understanding of meaning.**
5. **Detectors** (`text/central_message.py`, `organization.py`, `supporting_material.py`,
   `language.py`; `text/analyze.py` → `TextReport`, `schemas/text.py`) implement the ADR 0017
   definitions. Statistics: PERCENT, MONEY or QUANTITY entities, or a CARDINAL with digits;
   dates, times and ordinals are not statistics.
6. **Undefined rare terms** (owner, answer 3). A candidate term is a NOUN, VERB or ADJ (not a
   proper name, number or stop word, at least 3 letters) whose Zipf frequency is below 3.0 for
   both its form and its lemma, or an all-capital acronym of at least 2 letters **whose own Zipf
   frequency is below 3.0** (so "AI", "TV", "USA", "OK" are not flagged). It is defined when a
   definition cue is in the same or the next sentence. Each term counts once, at its first use.
   The 3.0 stays in `configs/rubric.yaml` as provisional.
7. **Evidence and confidence.** Every measurement quotes transcript spans exactly; a count of
   zero quotes the part that was searched (opening, closing, body or the whole talk).
   Confidence is **medium** at most (rule-based, accuracy unknown until E4) and **low** when the
   punctuation is unreliable (ADR 0011). With unreliable punctuation, sentence length is "not
   measurable" and the other measures stay, at low confidence (ADR 0017).
8. **Arabic:** every content measure is "not measurable" (FINAL_04 §5), which the rubric treats
   as missing, never low.
9. **Hedging is deferred** (owner, answer 4): it is not used by the rubric, and it is the measure
   closest to a confidence-style judgement. FINAL_04 §3 is marked accordingly.
10. **C2 and the lexicons** (owner, answer 5, for P2-e): the C2 generation prompts describe each
    defect in plain terms and are not written from these cue lists; C2 is split into a
    development and a test part, and the lexicons are adjusted on the development part only, so
    E4 on the test part is not inflated.

## Lexicon review (owner, 2026-11-21, PR #30)

1. **Normalisation before matching.** Contractions and informal forms are expanded in both the
   transcript and the lexicons ("i'll" → "i will", "we'll" → "we will", "i'm" → "i am",
   "i'd" → "i would", "gonna" → "going to"; list under `normalise` in `text_cues.yaml`). The
   "we" forms are added: "today we will", "in this talk we will", "in this presentation we
   will". A test checks that every lexicon entry survives the same normalisation as the
   transcript and matches its own wording, so no entry can silently fail.
2. **Thesis:** "today i will", "today i'm going to", "in this talk i will", "in this
   presentation i will" are removed. They announce a topic, not a main point, so they stay
   only as introduction cues.
3. **Definition:** "that is" is removed ("that is why", "that is important" would mark almost
   every term as defined); "i e" stays.
4. **Analogy:** "like a" / "like an" is skipped after "would" (so "I'd like a minute" is not an
   analogy), or when the "like" is a **hesitation** filler found by the speech detector.
   Discourse-marker events are not used for this: ADR 0012 counts every "like" as a discourse
   marker (an upper bound), so using them would remove every "like a" analogy. "is similar
   to" is removed as a duplicate of "similar to"; a sentence counts once per type anyway.
5. **Quotation:** "said that" counts only when the word before it is not "i" or "we".
6. **Transitions:** "then" counts only as the first word of a sentence.
7. **Conclusion:** added "in closing", "to close", "to finish", "the bottom line is", "let me
   leave you with".

Each change has a fixture test in `tests/unit/test_text_detectors.py`.

## Consequences

- The eleven `text.*` rubric measures are now available (`resources/rubrics/s1_presentation.yaml`,
  `docs/rubric/S1_ANCHORS.md`). P2-c adds `pauses.mid_sentence_per_min`, the scoring and the
  assessment with verified evidence.
- New runtime dependencies: `spacy` (and its MIT/BSD stack) and `wordfreq`; the licence check
  reports 0 FAIL.
- LIMITATIONS: content measures are rule-based and English only; a Whisper misrecognition can
  appear as an undefined rare term; sentences rest on Whisper's punctuation; hedging is not
  measured.
- What would show this is wrong: low per-label F1 in E4 on the C2 test part, or a high hook
  false-positive rate on the planted no-hook openings.
