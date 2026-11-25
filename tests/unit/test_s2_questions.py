"""S2 question bank (P3-e): six per language, two at random or the user's own, and no question
asks for limitations (it would prompt the limitation cue the rubric measures)."""

from __future__ import annotations

import random
import re

import pytest

from presenceai.lang import LANGUAGES, followup_questions
from presenceai.live.s2_session import pick_questions
from presenceai.text.followup import limitation_cue


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (reproducible test picks, not security)


ASKS_FOR_LIMITS = re.compile(
    r"limitation|limit|weakness|weak point|shortcoming|drawback|downside|future work|"
    r"improve|go wrong|did not|didn't|not test|out of scope|حدود|قيود|ضعف|عيوب|نقص|"
    # nor a scope or future-work invitation (owner, 2026-11-25): "out of scope" and "future
    # work" are limitation cues, so such a question hands out the level-4 condition
    r"scope|future|next step|not to build|not build|left out|would you add|"
    r"نطاق|المستقبل|مستقبلًا|لاحقًا|الخطوة التالية|ألّا تبنيه|لم تبنه",
    re.IGNORECASE,
)


@pytest.mark.parametrize("lang", LANGUAGES)
def test_six_questions_none_asking_for_limitations(lang: str):
    bank = followup_questions(lang)  # type: ignore[arg-type]
    assert len(bank) == 6 and len(set(bank)) == 6
    for q in bank:
        assert q.endswith(("?", "؟")), q
        assert not ASKS_FOR_LIMITS.search(q), q
        if lang == "en":
            assert limitation_cue(q) is None, q  # no limitation cue inside a question


def test_the_rule_catches_scope_and_future_work_invitations():
    for q in (
        "What did you decide not to build, and why?",  # the replaced Q6
        "What would you add in future work?",
        "What was out of scope?",
        "ما الذي قررت ألّا تبنيه، ولماذا؟",
    ):
        assert ASKS_FOR_LIMITS.search(q), q


def test_two_at_random_or_the_users_own():
    bank = followup_questions("en")
    a = pick_questions(bank, [None, ""], count=2, rng=seeded(1))
    assert len(a) == 2 and len(set(a)) == 2 and set(a) <= set(bank)
    own = pick_questions(bank, [" Why Python? ", None], count=2, rng=seeded(1))
    assert own[0] == "Why Python?" and own[1] in bank
    both = pick_questions(bank, ["A?", "B?"], count=2, rng=seeded(1))
    assert both == ("A?", "B?")
    seen = {pick_questions(bank, [], count=2, rng=seeded(i)) for i in range(30)}
    assert len(seen) > 5  # really random across sessions
