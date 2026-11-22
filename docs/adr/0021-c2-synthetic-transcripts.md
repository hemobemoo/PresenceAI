# 0021 — C2 synthetic presentation transcripts: Colab, Qwen3.5-9B, design (P2-e)

- Status: **accepted** (owner, 2026-11-23, after the hand check). Platform, batch mode and
  design decided by the owner on 2026-11-22; the model confirmed after the pilots.
- Date: 2026-11-22
- Decider: project owner
- Plan reference: FINAL_07 §C (C2), FINAL_10 E4 (content detectors), FINAL_06 §5 (T2 teacher),
  ADR 0017 (hook cases), ADR 0018 (cue lists, dev-only tuning)
- Config keys affected: none (the experiment pins its model in `experiments/c2_synth/generate_c2.py`)

## Decision

1. **Platform: a free Colab T4, batch generation inside the notebook, no public endpoint**
   (owner, 2026-11-22). Colab is the first choice because Kaggle notebook access takes too long
   to become available. No project code calls an outside service (AGENTS.md rule 2 unchanged).
   Only the synthetic prompts (`resources/c2/prompts.jsonl`) and the generated transcripts are
   in the notebook. Output goes to Google Drive during the run (synthetic text only) so a
   stopped session resumes, and is then saved into the git-ignored `data/c2/`.
2. **Model: `Qwen/Qwen3.5-9B`**, commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
   - Licence: Apache-2.0 on the model card (`license: apache-2.0`) **and** in its LICENSE file
     (Apache License 2.0), not gated. Apache-2.0 puts no limit on training with its outputs.
   - Size: 9.65 B parameters, bf16 weights 19.3 GB in 4 shards; 4-bit NF4 (bitsandbytes, fp16
     compute) needs about 6–7 GB on the T4's 15 GB.
   - Before generating, every file is checked against the hash Hugging Face lists for that
     commit (SHA-256 for the weight shards and tokenizer, git blob SHA-1 for small files); any
     mismatch stops the run. **Done on 2026-11-22:** all 14 files verified against
     `Qwen/Qwen3.5-9B@c202236` (on download and again before each run).
   - Licence record: model card front matter `license: apache-2.0` with `license_link` to its
     LICENSE file; the LICENSE file at commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a` is the
     Apache License, Version 2.0; the repository is not gated (checked 2026-11-22).
   - Non-thinking mode; sampling as the model card recommends for instruct use (temperature
     0.7, top-p 0.8, top-k 20), seeded per prompt.
   - Compared, all Apache-2.0 or MIT and ungated: gemma-4-12B-it (Apache-2.0 on its card but
     **no LICENSE file** in the repository), phi-4 (MIT, 14.7 B, Dec 2024, older), Qwen3.5-27B
     and Mistral-Small-3.2-24B (Apache-2.0, too large for a T4 at 4-bit). Qwen3.5-9B is the
     newest model that meets both licence checks and fits.
   - **Option, not depended on:** a free Colab TPU, if available, could run a larger model;
     the plan does not rely on it.
3. **Resumable:** every attempt is appended and flushed (`attempts.jsonl`, `generated.jsonl`,
   `rejected.jsonl`); a new run skips finished ids and counts earlier attempts.
3a. **Quality loop** (owner, 2026-11-22; final form after pilot 2b). An attempt is kept only if:
   - it ended before the **800-token cap**, a safety net only; a capped transcript is always
     rejected and regenerated, never kept (a cut-off talk would be an unplanned "no
     conclusion");
   - it has **250–550 words** (floor lowered from 300 after pilot 3) and no markdown, heading,
     bullet, label or thinking text;
   - it does not echo the prompt ("main idea", "central idea"; "specific detail" added after
     pilot 5).
   Length comes from a **fixed shape** in the prompt, not a word count (the model ignored word
   counts in pilots 1 and 2): first "exactly 4 short paragraphs of 3 to 4 sentences each",
   then, after pilot 3, **"exactly 4 paragraphs of 5 to 6 sentences each"** (owner,
   2026-11-22).
   **The model's yes/no check is not a filter** (owner, 2026-11-22; this replaces the earlier
   rule that the defect check is the model's yes/no). Evidence, pilot 2b: the generator
   followed every hook flag, yet the model answered "attention-catching opening = yes" for all
   5 transcripts, including the 4 plain "Hello everyone…" openings. A check that wrong would
   discard correct transcripts and keep wrong ones.
   The loop **uses neither the P2-b cue lexicons nor the text detectors** (tested: the
   generator imports nothing from PresenceAI), so C2 is not filtered down to the cases our
   detectors already catch and E4 is not inflated. A failed item is regenerated with a new
   seed, **at most 3 attempts**; items that still fail are logged in `rejected.jsonl` and left
   out. Every attempt is logged with its text, and the rejection count per group is reported
   (`check_c2.py`).
4. **Pilots first:** 5 prompts, one per group, shown to the owner before the full run.
   - **Pilot 1** (topic 00, 2026-11-22; 5 transcripts in 78 s on the T4, 4-bit about 8 GB):
     every hook matched its flag and no-conclusion was clean, but all 5 exceeded 380 words
     (386–537; 4 over 450), the no-supporting-material talk used an analogy, the
     undefined-jargon talk defined one term and echoed the prompt ("The single main idea of
     this talk is…"), and the no-central-message talk summed one message up at the end.
   - Prompt fixes after pilot 1: a hard 380-word limit (later replaced by the fixed shape); no comparisons or analogies in that group; no
     definition "even in a short clause"; "do not end with a sentence that sums up one message
     either"; "say the central idea in your own words; do not use the words 'main idea'".
   - **Pilot 2** (topic 01, 2026-11-22). Run 2a, stopped after round 1: "do not use the words
     'main idea'" primed the echo (3 of 5), and asking the 25 yes/no questions in one batch ran
     the GPU out of memory. Run 2b, stopped after round 1: lengths 458–544 words against
     "stop before 350", one transcript hit the cap, one echo, and the yes/no check said "hook =
     yes" on all 5 (wrong on the 4 plain openings). Fixes: the fixed shape, 300–550 words, the
     800-token safety cap, no model yes/no filter (3a).
   - **Pilot 3** (topic 02), stopped after 2 rounds: the shape was obeyed (4 paragraphs every
     time, 217–339 tokens, no cap, echo or format problem) but every attempt had 198–298
     words, under the 300-word floor. Owner's change: the shape grows to 4 paragraphs of 5 to
     6 sentences, and the floor goes down to 250 words (ceiling 550 and the 800-token cap
     kept).
   - **Pilot 4** (topic 03, commit b14830b): all 5 passed on the first attempt (427–488 words,
     73 s), every hook as asked. Owner's review: the planted defects were right except in the
     no-conclusion group.
   - **No-conclusion leak** (owner, 2026-11-22): **pilot 2b and pilot 4 both leaked a
     summing-up ending in the no-conclusion group** (pilot 4's last sentence: "By embracing these
     diverse roles, libraries are securing their place as essential institutions in our rapidly
     changing world."). The prompt was tightened to: "End the talk inside the last point: the
     final sentence gives one more specific detail about that point. The talk stops there, with
     no sentence about the topic as a whole, the talk itself, or the future." (commit e62cea6).
   - **Borderline thesis line accepted as label noise** (owner, 2026-11-22): in pilot 4's
     no-central-message talk, a paragraph-1 sentence ("It seems the building is adapting to the
     needs of a generation that learns by doing…") could be read as a thesis. The owner accepted
     this as label noise; the hand check measures it and the prompt is not changed.
   - **Pilot 5** (no-conclusion only, topics 04, 05, 06, tightened prompt, commit e62cea6):
     all 3 passed on the first attempt (503–514 words, 61.5 s for the batch of 3). No ending
     sums up the talk; each stops on a detail of the last point. Topic 05's last sentence echoed
     the prompt ("The last specific detail I want to mention is…"), so "specific detail" was
     added to the echo check (3a): such an attempt is now rejected and regenerated. Full run
     at the commit with that check.
4a. **Full run** (2026-11-22, Colab T4, generator and prompts at commit b12f6f1, notebook
   `notebooks/c2_colab.ipynb`; torch 2.14.1+cu130, transformers 5.24.0, accelerate 1.16.0,
   bitsandbytes 0.50.2, huggingface_hub 2.4.0; about 2 h, ~75 s per batch of 5; no
   disconnect). Results from `check_c2.py` (`experiments/results/c2/generated_checks.json`):
   - **397 of 400 accepted, 3 rejected** after 3 attempts: `c2_no_conclusion_23` (test, 561
     words), `c2_no_conclusion_75` (dev, echo), `c2_no_supporting_material_79` (dev, echo).
   - 466 attempts; accepted on attempt 1: 346, attempt 2: 42, attempt 3: 9.
   - Failed attempts by group: control 22, no central message 3, no conclusion 20, no
     supporting material 14, undefined jargon 10. Reasons: length 38 (all over 550 words,
     none under 250), echo 25 ("main idea" 12 in no supporting material; "specific detail"
     in no conclusion), format 7. No attempt reached the 800-token cap.
   - **Split:** dev 118, test 279. By group, dev/test: control 24/56, no central message
     24/56, no conclusion 23/55, no supporting material 23/56, undefined jargon 24/56. Hook:
     40 of 80 in every group except no supporting material (39 of 79).
   - Median words: control 484, no central message 438, no conclusion 464, no supporting
     material 397, undefined jargon 454 (range 302–550).
   - The post-run check flags one accepted transcript (`c2_no_supporting_material_78`, test)
     for a digit; the digit is the topic's name ("3D printing"), not supporting material, so
     it is kept.
   - **Length selection:** control talks ran longest, and 22 control attempts (555–646
     words) were regenerated under the 550-word ceiling, so accepted controls are the shorter
     ones the model wrote. Length differs by group (no supporting material is shortest), so
     word count alone partly reveals the group; E4 should not use length as a feature.
   - The hand-check sample was drawn (`hand_check.py draw`, 50 items, 10 per group from the
     test part); results in 4b.
4b. **Hand check** (owner, blinded, 2026-11-23; `hand_check.py score`,
   `experiments/results/c2/hand_check.json`). The owner's labels are compared with the planted
   design on all five labels of every item.
   - **27 of 50 items (0.54) have at least one planted label the owner disagrees with.**
     Per label (of 50): central message 0.30, conclusion 0.10, supporting material 0.16,
     unexplained terms 0.04, hook 0.00.
   - **The planted defect itself** is confirmed in no central message 10/10, no conclusion
     10/10, undefined jargon 10/10, but in **no supporting material only 2/10**: the owner
     found supporting material in 8 of the 10 talks told to have none.
   - The other disagreements are on labels the group did not target: no conclusion, central
     message absent 8/10; no central message, conclusion absent 5/10; no supporting material,
     central message absent 4/10; control, central message absent 3/10 and unexplained terms
     present 2/10.
   - Scoring fix: the sheet held the quoted strings "yes"/"no", and the first score used
     `bool()`, which reads "no" as yes (it reported 0.88). `hand_check.py` now reads yes/no
     strictly and refuses any other value (tested); the numbers above are from the fixed score.
   - **Consequence for E4:** the planted labels are reliable for hook, unexplained terms and
     conclusion, less for central message, and **not for supporting material in the
     no-supporting-material group**. E4 on the planted labels reports these rates next to
     each F1, and E4's second report (the 50 hand-checked items against the owner's labels) is
     the reference for supporting material and central message.
5. **Design** (owner, unchanged from the P2-e proposal):
   - 400 transcripts: 5 groups of 80 (control, no central message, no conclusion, no supporting
     material, undefined jargon); each of 80 topics (`resources/c2/topics.yaml`, written for
     PresenceAI) appears once per group.
   - Hook crossed with each group: 40 with an attention-catching opening, 40 opening plainly
     (in the no-supporting-material group the hook may not use numbers, a case or a story).
   - Prompts describe each defect in plain words and contain **no phrase from the detectors'
     cue lists** (tested; topic names excluded).
   - Split 30/70 (dev/test) **by topic** (24 / 56 topics, 120 / 280 transcripts), hook balanced
     within each group and part, seed 20261122. Lexicons are tuned on dev only.
   - Checks: the quality loop (3a) and, after the run, automatic checks on all accepted
     transcripts (`check_c2.py`: length, format, echo, no digits in the no-supporting-material
     group). **The planted labels are otherwise unverified.**
   - **Hand check** (owner, 2026-11-22; `hand_check.py`): 10 accepted transcripts per group
     (4 defect groups + controls = 50), drawn at random **from the test part only**, seed
     20261123. The sheet is **blinded**: it shows only a neutral key and the text, shuffled;
     the planted group and hook are kept in a separate key file. For each transcript the owner
     records what is actually present: central message, conclusion, supporting material,
     unexplained technical terms, hook (yes/no each), not only whether the planted defect is
     there. The **planted-label error rate in the sample is E4's label-noise estimate** (this
     ADR and LIMITATIONS).
   - **E4 is reported twice:** on the whole test part against the planted labels, and on the
     hand-checked subset against the owner's labels.
6. **T2 teacher is decided in P4** (FINAL_06): Kaggle TPU Qwen3.8-27B (Apache-2.0, batch, no
   public endpoint) is the leading option.

## Consequences

- C2 labels are the planted design, checked as above; E4 reports per-label F1 on the test part,
  each with the hand-checked label error rate (4b), and again on the hand-checked subset
  against the owner's labels.
- Pilot results, the full run's checks and the hand-checked error rates are recorded above
  (4, 4a, 4b).
