"""Recording -> speech report -> text report -> rubric assessment (P2-c, ADR 0019).

S2 (ADR 0022): each follow-up answer recording gets its own speech report; a recording that
cannot be read is a failed recording (missing values, never low).

P3-a: `limit_s` is the session's time limit for recordings made in the app (None for an uploaded
file: overrun is then missing). `strict=False` is the app's mode: a failed text analysis makes the
text measures missing ("text analysis failed") and a failed dimension shows "not measurable"
(ADR 0019); both are logged as the stage or dimension and the error class only."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from av.error import FFmpegError

from presenceai.knowledge.embed import Embedder
from presenceai.lang import text
from presenceai.pipeline.followup import followup_report
from presenceai.rubric import loader, score
from presenceai.schemas.followup import FollowupInput, FollowupReport
from presenceai.schemas.report import AssessmentReport
from presenceai.schemas.rubric import Rubric
from presenceai.schemas.speech import SpeechLanguage, SpeechReport
from presenceai.schemas.text import TextReport
from presenceai.settings import AsrSettings, SpeechSettings
from presenceai.speech import analyze as speech_analyze
from presenceai.text import analyze as text_analyze
from presenceai.text.nlp import Parser, Tok

log = logging.getLogger(__name__)


def no_parser(texts: Sequence[str]) -> list[list[Tok]]:
    """For languages whose content is not measured (Arabic): never called."""
    raise RuntimeError("the text parser is only used for English")


def _text_failed(speech: SpeechReport, rubric: Rubric, scenario: str) -> TextReport:
    """Every text measure missing because the text analysis failed (app mode)."""
    lang = "ar" if speech.transcript.language == "ar" else "en"
    reason = text(lang, "assessment.text_failed")
    missing = {k: reason for k in rubric.measures if k.startswith("text.")}
    return TextReport(
        language=lang, scenario=scenario, punctuation_reliable=False, sentences=0,
        measurements={}, not_measurable=missing,
    )  # fmt: skip


def assess_report(
    speech: SpeechReport,
    *,
    parser: Parser,
    cfg: SpeechSettings,
    scenario: str,
    followups: FollowupReport | None = None,
    limit_s: float | None = None,
    strict: bool = True,
) -> AssessmentReport:
    """Text analysis and rubric scoring on an existing speech report."""
    rubric = loader.load_rubric(scenario)
    thresholds = loader.load_thresholds(scenario)
    try:
        text_report = text_analyze.analyze_text(
            speech.transcript, parser=parser, pauses=cfg.pauses, scenario=scenario,
            fillers=speech.fillers,
        )  # fmt: skip
    except Exception as e:
        if strict:
            raise
        log.warning("stage failed: stage=text error=%s", type(e).__name__)
        text_report = _text_failed(speech, rubric, scenario)
    assessment = score.assess(
        rubric, thresholds, speech, text_report, cfg, followups, limit_s=limit_s, strict=strict
    )
    return AssessmentReport(
        speech=speech, text=text_report, assessment=assessment, followups=followups
    )


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
    followups: FollowupInput | None = None,
    followups_dir: Path | None = None,
    embedder: Embedder | None = None,
    limit_s: float | None = None,
    strict: bool = True,
    **speech_kwargs: Any,
) -> AssessmentReport:
    report = speech_analyze.analyze_file(
        path, language=language, model=model, model_name=model_name, options=options,
        speech=speech, **speech_kwargs,
    )  # fmt: skip
    fu = None
    if followups is not None:
        base = followups_dir or Path(path).parent

        def answer(file: str) -> SpeechReport | None:
            try:
                return speech_analyze.analyze_file(
                    base / file, language=language, model=model, model_name=model_name,
                    options=options, speech=speech, **speech_kwargs,
                )  # fmt: skip
            except (OSError, ValueError, FFmpegError):
                return None  # the recording failed its checks: missing, never low

        answers = [answer(a.file) for a in followups.answers]
        fu = followup_report(
            [a.question for a in followups.answers], answers,
            [a.window_s for a in followups.answers], language="ar" if language == "ar" else "en",
            embedder=embedder, sources=[Path(a.file).name for a in followups.answers],
        )  # fmt: skip
    return assess_report(
        report, parser=parser, cfg=speech, scenario=scenario, followups=fu, limit_s=limit_s,
        strict=strict,
    )  # fmt: skip
