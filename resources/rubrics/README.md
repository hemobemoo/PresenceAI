# Rubrics (CC BY-NC-SA 4.0)

The files in this folder adapt the **Oral Communication VALUE Rubric** of the Association of
American Colleges and Universities (AAC&U), licensed under CC BY-NC-SA 4.0
(https://creativecommons.org/licenses/by-nc-sa/4.0/). As derivatives they are licensed under
**CC BY-NC-SA 4.0** as well, not under the project's Apache-2.0 licence (NOTICE).

> Adapted from the Oral Communication VALUE Rubric, Association of American Colleges and
> Universities (AAC&U), CC BY-NC-SA 4.0. Changes: anchors rewritten as observable behaviour with
> provisional numbers; parts that no rule can observe are listed as left out.

| File | Status | Decision |
|---|---|---|
| `s1_presentation.yaml` | v0.2, accepted (reviewed by the supervisor); thresholds provisional | ADR 0017 |
| `s2_project_defense.yaml` | v0.1, accepted (reviewed by the supervisor); thresholds provisional; relevance thresholds pending calibration (S2-b) | ADR 0022 |

Thresholds are in `configs/rubric.yaml`. The anchors documents are generated with
`uv run python scripts/render_rubric_review.py --rubric <id>` into `docs/rubric/S1_ANCHORS.md`
(`s1_presentation`, the default) and `docs/rubric/S2_ANCHORS.md` (`s2_project_defense`). The
Arabic wording in both is a draft pending a native-speaker review (ADR 0016 open item 4).
