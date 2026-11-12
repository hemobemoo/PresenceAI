from importlib.metadata import version

from typer.testing import CliRunner

from presenceai.cli import app


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"presenceai {version('presenceai')}"
