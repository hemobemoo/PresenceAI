"""Build the 400 C2 generation prompts (P2-e, ADR 0021; FINAL_07 §C).

    uv run python experiments/c2_synth/build_prompts.py

Design (owner, 2026-11-22):
- 5 groups of 80: control, no_central_message, no_conclusion, no_supporting_material,
  undefined_jargon. Each of the 80 topics (resources/c2/topics.yaml) appears once per group.
- Hook crossed with each group: 40 transcripts with an attention-grabbing opening, 40 without.
- Split 30/70 (dev/test) **by topic**, so all 5 versions of a topic fall in the same part; hook
  is balanced within each group and part. Seeded, so the file is reproducible.
- Each defect is described in plain words. The prompts are **not** written from the text
  detectors' cue lists (lang/en/text_cues.yaml): no cue phrase is quoted, and a test checks it.
Writes resources/c2/prompts.jsonl (synthetic prompts only; released with the code).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TOPICS = ROOT / "resources" / "c2" / "topics.yaml"
OUT = ROOT / "resources" / "c2" / "prompts.jsonl"
SEED = 20261122
DEV_TOPICS = 24  # 30 % of 80
GROUPS = (
    "control",
    "no_central_message",
    "no_conclusion",
    "no_supporting_material",
    "undefined_jargon",
)

BASE = (
    "Write the word-for-word transcript of a short spoken presentation, about 2 to 3 minutes "
    "long, given by a university student to classmates. Topic: {topic}.\n"
    "Length: between 260 and 340 words; stop before 350 words.\n"
    "Write it exactly as it would be spoken: plain paragraphs, no title, no headings, no bullet "
    "points, no stage directions, no speaker labels, no notes before or after it. Output only "
    "the transcript."
)

# What each group must do, in plain words; everything not mentioned stays like a normal talk.
# Pilot 2: naming "the main idea" in the prompt made the model echo it, so it is avoided here.
GROUP_RULES = {
    "control": (
        "Make the talk complete: early on, say in one sentence what the whole talk is trying "
        "to get across, and come back to that point at the end. Back up the points with "
        "concrete support: facts with numbers, a specific case from real life, a comparison "
        "with something familiar, or the words of a named expert. If you use a specialised "
        "term, explain it in simple words. End with a short closing that sums the talk up."
    ),
    "no_central_message": (
        "Talk about several loosely related aspects of the topic, and never say in one "
        "sentence what the whole talk is trying to get across, neither at the start nor at the "
        "end; a listener should not be able to point to one sentence that states the purpose, "
        "and do not end with a sentence that sums up one message either. Still back up the "
        "points with concrete support, explain any specialised term in simple words, and end "
        "with a short closing."
    ),
    "no_conclusion": (
        "Early on, say in one sentence what the whole talk is trying to get across, and back "
        "up the points with concrete support; explain any specialised term in simple words. "
        "Stop right after the last point of the body: no summary, no wrap-up, no closing "
        "remark and no thanks at the end."
    ),
    "no_supporting_material": (
        "Early on, say in one sentence what the whole talk is trying to get across, come back "
        "to that point at the end, and end with a short closing. Support nothing: use only "
        "general statements and opinions, with no numbers or figures, no specific cases or "
        "stories, no comparisons or analogies (never say that something is like something "
        "else), and no quoted or named experts. Explain any specialised term in simple words."
    ),
    "undefined_jargon": (
        "Early on, say in one sentence what the whole talk is trying to get across, come back "
        "to that point at the end, and back up the points with concrete support. Use at least "
        "five specialised technical terms of the field, the kind an expert would use, and never "
        "explain or define any of them, not even in a short clause after the term. End with a "
        "short closing."
    ),
}

HOOK_RULES = {
    True: (
        "Open with one or two sentences meant to catch the audience's attention before the "
        "topic is introduced."
    ),
    False: (
        "Open plainly: the very first sentence greets the audience or names the topic, with no "
        "question, surprising fact, anecdote or other attention-catching opening before it."
    ),
}
# A hook made of a number or a story would itself be supporting material.
NO_SUPPORT_HOOK = " The opening must not use numbers, a specific case or a story."


def build(topics: list[str], seed: int = SEED) -> list[dict[str, object]]:
    if len(topics) != 80 or len(set(topics)) != 80:
        raise ValueError("C2 needs 80 distinct topics")
    rng = random.Random(seed)  # noqa: S311 - reproducible design split, not security
    order = list(range(80))
    rng.shuffle(order)
    dev = set(order[:DEV_TOPICS])
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        for part, members in (("dev", sorted(dev)), ("test", sorted(set(range(80)) - dev))):
            hooks = [True, False] * (len(members) // 2)
            rng.shuffle(hooks)
            for t, hook in zip(members, hooks, strict=True):
                rule = HOOK_RULES[hook]
                if hook and group == "no_supporting_material":
                    rule += NO_SUPPORT_HOOK
                prompt = "\n".join([BASE.format(topic=topics[t]), GROUP_RULES[group], rule])
                rows.append({
                    "id": f"c2_{group}_{t:02d}", "topic_index": t, "topic": topics[t],
                    "group": group, "hook": hook, "part": part, "prompt": prompt,
                })  # fmt: skip
    rows.sort(key=lambda r: str(r["id"]))
    return rows


def main() -> int:
    topics = list(yaml.safe_load(TOPICS.read_text(encoding="utf-8"))["topics"])
    rows = build(topics)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
