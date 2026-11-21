"""Recording -> speech report -> text report -> rubric assessment (P2-c, ADR 0019)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from presenceai.rubric import loader, score
from presenceai.schemas.report import AssessmentReport
from presenceai.schemas.speech import SpeechLanguage, SpeechReport
from presenceai.settings import AsrSettings, SpeechSettings
from presenceai.speech import analyze as speech_analyze
from presenceai.text import analyze as text_analyze
from presenceai.text.nlp import Parser, Tok


def no_parser(texts: Sequence[str]) -> list[list[Tok]]:
    """For languages whose content is not measured (Arabic): never called."""
    raise RuntimeError("the text parser is only used for English")


def assess_report(
    speech: SpeechReport, *, parser: Parser, cfg: SpeechSettings, scenario: str
) -> AssessmentReport:
    """Text analysis and rubric scoring on an existing speech report."""
    text_report = text_analyze.analyze_text(
        speech.transcript, parser=parser, pauses=cfg.pauses, scenario=scenario,
        fillers=speech.fillers,
    )  # fmt: skip
    rubric = loader.load_rubric(scenario)
    thresholds = loader.load_thresholds(scenario)
    assessment = score.assess(rubric, thresholds, speech, text_report, cfg)
    return AssessmentReport(speech=speech, text=text_report, assessment=assessment)


def assess_file(
    path: str | Path,
    *,
    language: SpeechLanguage,
    model: Any,
    model_name: str,
    options: AsrSettings,
    speech: SpeechSettings,
    parser: Parser,
    scenario: str = "s1_presentation",
    **speech_kwargs: Any,
) -> AssessmentReport:
    report = speech_analyze.analyze_file(
        path, language=language, model=model, model_name=model_name, options=options,
        speech=speech, **speech_kwargs,
    )  # fmt: skip
    return assess_report(report, parser=parser, cfg=speech, scenario=scenario)
