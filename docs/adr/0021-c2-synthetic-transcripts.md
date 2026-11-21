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
     mismatch stops the run.
   - Non-thinking mode; sampling as the model card recommends for instruct use (temperature
     0.7, top-p 0.8, top-k 20), seeded per prompt.
   - Compared, all Apache-2.0 or MIT and ungated: gemma-4-12B-it (Apache-2.0 on its card but
     **no LICENSE file** in the repository), phi-4 (MIT, 14.7 B, Dec 2024, older), Qwen3.5-27B
     and Mistral-Small-3.2-24B (Apache-2.0, too large for a T4 at 4-bit). Qwen3.5-9B is the
     newest model that meets both licence checks and fits.
   - **Option, not depended on:** a free Colab TPU, if available, could run a larger model;
     the plan does not rely on it.
3. **Resumable:** each finished batch is appended to `generated.jsonl` and flushed; a new run
   skips ids already there.
4. **Pilot first:** 5 prompts, one per group, shown to the owner before the full run.
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
   - Checks: automatic format checks on all 400 (`check_c2.py`: length, format, no digits in the
     no-supporting-material group); a yes/no pass of the model per planted defect on all 400;
     the owner's hand check of a stratified random sample (10 per defect type + 10 controls).
     The sample's error rate goes into this ADR and LIMITATIONS as E4's label-noise estimate.
6. **T2 teacher is decided in P4** (FINAL_06): Kaggle TPU Qwen3.8-27B (Apache-2.0, batch, no
   public endpoint) is the leading option.

## Consequences

- C2 labels are the planted design, checked as above; E4 reports per-label F1 on the test part.
- Pilot results, the full run's checks and the hand-checked error rate are added here before
  this ADR is accepted.
