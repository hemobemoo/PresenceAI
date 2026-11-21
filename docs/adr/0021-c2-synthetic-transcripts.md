# 0021 — C2 synthetic presentation transcripts: Colab, Qwen3.5-9B, design (P2-e)

- Status: **proposed** (2026-11-22). Platform, batch mode and design decided by the owner on
  2026-11-22; the model is proposed here and confirmed after the 5-prompt pilot.
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
2. **Model (proposed): `Qwen/Qwen3.5-9B`**, commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
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
   - it has **300–550 words** and no markdown, heading, bullet, label or thinking text;
   - it does not echo the prompt ("main idea", "central idea").
   Length comes from a **fixed shape** in the prompt ("exactly 4 short paragraphs of 3 to 4
   sentences each"), not a word count: the model ignored word counts in pilots 1 and 2.
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
   - **Pilot 3** (topic 02, new topics): shown to the owner before the full run.
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

- C2 labels are the planted design, checked as above; E4 reports per-label F1 on the test part.
- Pilot results, the full run's checks and the hand-checked error rate are added here before
  this ADR is accepted.
