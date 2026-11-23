"""Command-line entry point (`uv run presenceai ...`)."""

from __future__ import annotations

from enum import StrEnum
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Annotated

import typer
from av.error import FFmpegError

from presenceai.settings import get_settings

app = typer.Typer(
    name="presenceai",
    help="PresenceAI: adaptive, privacy-preserving presentation coach.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _main() -> None:
    """PresenceAI command-line interface."""


@app.command()
def version() -> None:
    """Print the installed PresenceAI version."""
    typer.echo(f"presenceai {_dist_version('presenceai')}")


class Lang(StrEnum):
    EN = "en"
    AR = "ar"


class AsrModel(StrEnum):
    WHISPER_SMALL = "whisper_small"
    WHISPER_TURBO = "whisper_turbo"


class Processing(StrEnum):
    UNKNOWN = "unknown"
    OFF = "off"
    ON = "on"


@app.command()
def analyze(
    file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Audio or video file."),
    ],
    lang: Annotated[Lang, typer.Option("--lang", help="Language spoken in the recording.")],
    model: Annotated[
        AsrModel | None,
        typer.Option("--model", help="ASR model (default: `asr.model` in configs/default.yaml)."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", dir_okay=False, help="Write the JSON report here (default: stdout)."),
    ] = None,
    capture_processing: Annotated[
        Processing,
        typer.Option(
            "--capture-processing",
            help="Browser auto-gain / noise suppression / echo cancellation during recording. "
            "Uploaded files are 'unknown' (loudness variation then has low confidence).",
        ),
    ] = Processing.UNKNOWN,
    baseline: Annotated[
        Path | None,
        typer.Option(
            "--baseline",
            exists=True,
            dir_okay=False,
            help="An earlier report (JSON) of your own: adds the change in speech level.",
        ),
    ] = None,
) -> None:
    """Transcribe a recording and write a JSON speech report (derived data only, no audio)."""
    from pydantic import ValidationError

    from presenceai.schemas.speech import CaptureProcessing, SpeechReport
    from presenceai.speech import analyze as speech_analyze
    from presenceai.speech import asr

    settings = get_settings()
    options = settings.asr
    state = capture_processing.value
    capture = CaptureProcessing(
        auto_gain_control=state, noise_suppression=state, echo_cancellation=state
    )
    base = None
    if baseline is not None:
        try:
            earlier = SpeechReport.model_validate_json(baseline.read_text(encoding="utf-8"))
        except (ValidationError, ValueError, OSError):
            typer.echo(f"{baseline.name}: not a PresenceAI speech report", err=True)
            raise typer.Exit(1) from None
        if earlier.loudness_profile is None:
            typer.echo(f"{baseline.name}: the report has no loudness profile", err=True)
            raise typer.Exit(1)
        base = (earlier.loudness_profile, earlier.capture, baseline.name)
    model_name = model.value if model is not None else options.model
    try:
        whisper = asr.load_model(model_name, options)
        report = speech_analyze.analyze_file(
            file,
            language=lang.value,
            model=whisper,
            model_name=model_name,
            options=options,
            speech=settings.speech,
            capture=capture,
            baseline=base,
        )
    except asr.ModelNotAvailable as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    except (ValueError, FFmpegError) as exc:
        typer.echo(f"{file.name}: cannot read audio ({type(exc).__name__})", err=True)
        raise typer.Exit(1) from None
    payload = report.model_dump_json(indent=2)
    if out is None:
        typer.echo(payload)
    else:
        out.write_text(payload + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}", err=True)


class Scenario(StrEnum):
    S1 = "s1_presentation"
    S2 = "s2_project_defense"


@app.command()
def assess(
    file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Audio or video file."),
    ],
    lang: Annotated[Lang, typer.Option("--lang", help="Language spoken in the recording.")],
    scenario: Annotated[
        Scenario, typer.Option("--scenario", help="Scenario whose rubric is used.")
    ] = Scenario.S1,
    model: Annotated[
        AsrModel | None,
        typer.Option("--model", help="ASR model (default: `asr.model` in configs/default.yaml)."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", dir_okay=False, help="Write the JSON report here (default: stdout)."),
    ] = None,
    followups: Annotated[
        Path | None,
        typer.Option(
            "--followups",
            exists=True,
            dir_okay=False,
            readable=True,
            help="S2 only: JSON with the follow-up questions and answer recordings.",
        ),
    ] = None,
) -> None:
    """Speech report, text report and rubric assessment with verified evidence (JSON)."""
    from presenceai.pipeline import assess as pipeline
    from presenceai.rubric.evidence import EvidenceError
    from presenceai.schemas.followup import FollowupInput
    from presenceai.speech import asr
    from presenceai.text.nlp import spacy_parser

    fu_input = None
    if followups is not None:
        if scenario is not Scenario.S2:
            typer.echo("--followups is only used with --scenario s2_project_defense", err=True)
            raise typer.Exit(2)
        try:
            fu_input = FollowupInput.model_validate_json(followups.read_text(encoding="utf-8"))
        except ValueError as exc:
            typer.echo(f"{followups.name}: not a valid follow-up file ({exc})", err=True)
            raise typer.Exit(2) from None
    settings = get_settings()
    options = settings.asr
    model_name = model.value if model is not None else options.model
    try:
        embedder = None
        if fu_input is not None and lang is Lang.EN:
            from presenceai.knowledge.embed import E5Embedder

            embedder = E5Embedder()  # relevance (ADR 0022); English only
        whisper = asr.load_model(model_name, options)
        parser = spacy_parser() if lang is Lang.EN else pipeline.no_parser
        report = pipeline.assess_file(
            file,
            language=lang.value,
            model=whisper,
            model_name=model_name,
            options=options,
            speech=settings.speech,
            parser=parser,
            scenario=scenario.value,
            followups=fu_input,
            followups_dir=followups.parent if followups is not None else None,
            embedder=embedder,
        )
    except EvidenceError:
        raise  # a rule-based finding with unverified evidence is a bug (ADR 0019)
    except asr.ModelNotAvailable as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    except (ValueError, FFmpegError) as exc:
        typer.echo(f"{file.name}: cannot read audio ({type(exc).__name__})", err=True)
        raise typer.Exit(1) from None
    payload = report.model_dump_json(indent=2)
    if out is None:
        typer.echo(payload)
    else:
        out.write_text(payload + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}", err=True)
