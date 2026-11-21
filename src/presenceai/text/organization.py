"""Organization and the opening hook (rubric dimension, ADR 0017 definitions).

- framing_parts (0-2): an introduction cue in an opening sentence, a conclusion cue in a closing
  sentence;
- transitions: sentences with a transition cue among their first 3 words (once per sentence);
  "then" counts only as the first word (owner's review);
- hook.present: one of the first N sentences is a question, contains a statistic, opens a story,
  is a quotation, or makes a contrast claim (a lead such as "most people think" with a turn such
  as "but" in that sentence or the next).
All are lexicon or pattern matches, not an understanding of the structure.
"""

from __future__ import annotations

from presenceai.lang import text
from presenceai.schemas.assessment import Measurement
from presenceai.text.document import Document, Sentence, find, match
from presenceai.text.measure import measurement
from presenceai.text.supporting_material import has_quotation, has_statistic

KEYS = ("text.organization.framing_parts", "text.organization.transitions", "text.hook.present")
TRANSITION_WITHIN_WORDS = 3


def _is_hook(doc: Document, s: Sentence) -> bool:
    if s.text.rstrip("\"'”’»)]} ").endswith(("?", "؟")):
        return True
    if has_statistic(s) or has_quotation(doc, s) or match(s.text, "story"):
        return True
    if match(s.text, "contrast_lead"):
        nxt = doc.sentences[s.index + 1] if s.index + 1 < len(doc.sentences) else None
        return bool(match(s.text, "contrast_turn") or (nxt and match(nxt.text, "contrast_turn")))
    return False


def has_transition(s: Sentence) -> bool:
    return any(
        at < TRANSITION_WITHIN_WORDS and (phrase != "then" or at == 0)
        for phrase, at in find(s.text, "transitions")
    )


def measure(doc: Document, values: dict[str, float]) -> dict[str, Measurement]:
    opening, closing = values["opening_share_pct"], values["closing_share_pct"]
    n_hook = int(values["hook_sentences"])
    intros = [s for s in doc.sentences if doc.in_opening(s) and match(s.text, "introduction")]
    concls = [s for s in doc.sentences if doc.in_closing(s) and match(s.text, "conclusion")]
    intro = intros[0] if intros else None
    concl = concls[0] if concls else None
    framing = [s for s in (intro, concl) if s is not None]
    transitions = [s for s in doc.sentences if has_transition(s)]
    first = doc.sentences[:n_hook]
    hook = next((s for s in first if _is_hook(doc, s)), None)
    hook_spans = [doc.span(hook)] if hook else [doc.span(s) for s in first]
    return {
        KEYS[0]: measurement(
            doc, len(framing), "count",
            text("en", "text.organization.framing", opening=opening, closing=closing),
            (doc.span(s) for s in framing), "all",
        ),
        KEYS[1]: measurement(
            doc, len(transitions), "sentences", text("en", "text.organization.transitions"),
            (doc.span(s) for s in transitions), "all",
        ),
        KEYS[2]: measurement(
            doc, hook is not None, "yes/no", text("en", "text.hook", sentences=n_hook),
            hook_spans, "opening",
        ),
    }  # fmt: skip
