# Report drafts

The files in this folder are source material for the final graduation report
(`Main_Project_Report.docx`). The report itself is written later and only on the owner's request;
these drafts are not edited into it automatically.

## Rules for every draft

1. **Register.** Formal academic English suited to a university graduation report. Write in the
   third person ("this project", "the system", "the spike"), never "I", "we" or "you". Use the past
   tense for work that has been completed and the present tense only for standing facts (for
   example, what a configuration file contains).
2. **Structure.** Every draft uses the same six headings, in this order:
   1. Aim
   2. Method
   3. Pass conditions
   4. Results
   5. Decision
   6. Limitations

   Short sub-headings inside a section are allowed. A closing "Sources" list is required (rule 3).
3. **Traceability.** Every figure must be traceable to a run id, a results CSV or a commit. Cite
   the run id (for example `20261113-021333_real`) or the file path next to the figure or in the
   table caption, and list all sources at the end. Figures that cannot be traced are not reported.
4. **No fabrication.** Results are copied from files written by experiment scripts
   (`experiments/results/`). Owner observations are labelled as such. Unverified explanations are
   stated as hypotheses.
5. **Wording about people.** Describe observable behaviour and numbers, never emotions, traits
   or other inferences about a person (AGENTS.md, "Wording for any user-facing text").
6. **Decisions.** Each decision names its decider, its date and its ADR in `docs/adr/`.

## Files

| File | Covers | ADR |
|---|---|---|
| `S1_summary.md` | Spike S1, live capture transport | `docs/adr/0002-live-capture-transport.md` |
| `S4_summary.md` | Spike S4, OpenRouter free model choice and hosted-LLM privacy decision | `docs/adr/0003-openrouter-model-and-privacy.md` |
