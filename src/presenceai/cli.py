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
) -> None:
    """Transcribe a recording and write a JSON speech report (derived data only, no audio)."""
    from presenceai.speech import analyze as speech_analyze
    from presenceai.speech import asr

    options = get_settings().asr
    model_name = model.value if model is not None else options.model
    try:
        whisper = asr.load_model(model_name, options)
        report = speech_analyze.analyze_file(
            file, language=lang.value, model=whisper, model_name=model_name, options=options
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
