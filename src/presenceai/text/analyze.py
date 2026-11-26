"""Text analysis: transcript -> TextReport with the rubric's content measures (P2-b, ADR 0018).

English only: for Arabic every content measure is "not measurable" (FINAL_04 §5), which the
rubric treats as missing, never as low (ADR 0017). The opening, closing and hook values come
from the scenario's rubric thresholds (`configs/rubric.yaml`).
"""

from __future__ import annotations

from presenceai.lang import language_name, text
from presenceai.rubric.loader import load_possible_main_point, load_thresholds
from presenceai.schemas.assessment import Measurement
from presenceai.schemas.speech import FillerEvent, FillerKind, Transcript
from presenceai.schemas.text import TextReport
from presenceai.settings import PauseSettings
from presenceai.text import central_message, language, organization, supporting_material
from presenceai.text.document import build
from presenceai.text.nlp import Parser

MEASURES = (
    *central_message.KEYS,
    *organization.KEYS,
    *supporting_material.KEYS,
    *language.KEYS,
)


def analyze_text(
    transcript: Transcript,
    *,
    parser: Parser,
    pauses: PauseSettings,
    scenario: str = "s1_presentation",
    fillers: tuple[FillerEvent, ...] = (),
) -> TextReport:
    """`fillers`: the speech report's filler events; only hesitations are used, to skip a
    "like a" that is a filler. Discourse markers are not used: ADR 0012 counts every "like" as
    one (an upper bound), which would remove every "like a" analogy."""
    lang = transcript.language
    if lang != "en" or not transcript.words:
        reason = (
            text(lang, "text.not_measured_language", language=language_name(lang, lang))
            if lang != "en"
            else text(lang, "common.no_words")
        )
        return TextReport(
            language=lang,
            scenario=scenario,
            punctuation_reliable=False,
            sentences=0,
            measurements={},
            not_measurable=dict.fromkeys(MEASURES, reason),
        )
    values = load_thresholds(scenario).values
    doc = build(
        transcript,
        parser,
        pauses,
        opening_pct=values["opening_share_pct"],
        closing_pct=values["closing_share_pct"],
        hesitation_words=frozenset(
            i
            for e in fillers
            if e.kind == FillerKind.HESITATION
            for i in range(e.first_word, e.last_word + 1)
        ),
    )
    found: dict[str, Measurement | None] = {
        **central_message.measure(doc, load_possible_main_point().window_pct),
        **organization.measure(doc, values),
        **supporting_material.measure(doc),
        **language.measure(doc, values),
    }
    reasons = {"text.language.mean_sentence_words": text("en", "text.unreliable_punctuation")}
    return TextReport(
        language=lang,
        scenario=scenario,
        punctuation_reliable=doc.punctuation_reliable,
        sentences=len(doc.sentences),
        measurements={k: m for k, m in found.items() if m is not None},
        not_measurable={k: reasons[k] for k, m in found.items() if m is None},
    )
