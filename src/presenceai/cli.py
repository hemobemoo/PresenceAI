"""Command-line entry point (`uv run presenceai ...`)."""

from __future__ import annotations

from importlib.metadata import version as _dist_version

import typer

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
