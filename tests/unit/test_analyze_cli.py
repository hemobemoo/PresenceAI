"""`presenceai analyze <file> --lang en|ar` with a fake model (speech/analyze.py + cli.py)."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from presenceai import cli
from presenceai.schemas.speech import SpeechReport
from presenceai.settings import AsrSettings
from presenceai.speech import analyze, asr


def test_analyze_file_builds_a_report(fake_whisper: Any, silent_wav: Path):
    report = analyze.analyze_file(
        silent_wav,
        language="en",
        model=fake_whisper(),
        model_name="whisper_small",
        options=AsrSettings(),
    )
    assert isinstance(report, SpeechReport)
    assert report.source_name == "talk.wav"
    assert report.model == "whisper_small"
    assert report.transcript.text.startswith("Hello everyone.")
    assert report.asr_quality.word_count == 6


@pytest.fixture
def patched_loader(monkeypatch: pytest.MonkeyPatch, fake_whisper: Any) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_load(name: str, options: AsrSettings, **_: Any) -> Any:
        seen["name"], seen["options"] = name, options
        return fake_whisper()

    monkeypatch.setattr(asr, "load_model", fake_load)
    return seen


def test_cli_prints_json_report(patched_loader: dict[str, Any], silent_wav: Path):
    result = CliRunner().invoke(cli.app, ["analyze", str(silent_wav), "--lang", "en"])
    assert result.exit_code == 0, result.output
    report = SpeechReport.model_validate_json(result.stdout)
    assert report.transcript.language == "en"
    assert patched_loader["name"] == "whisper_small"  # default from configs/default.yaml


def test_cli_writes_report_file_and_uses_chosen_model(
    patched_loader: dict[str, Any], silent_wav: Path, tmp_path: Path
):
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli.app,
        ["analyze", str(silent_wav), "--lang", "ar", "--model", "whisper_turbo", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["transcript"]["language"] == "ar"
    assert data["model"] == "whisper_turbo"
    assert patched_loader["name"] == "whisper_turbo"


def test_cli_rejects_unsupported_language(patched_loader: dict[str, Any], silent_wav: Path):
    result = CliRunner().invoke(cli.app, ["analyze", str(silent_wav), "--lang", "fr"])
    assert result.exit_code == 2
    assert "name" not in patched_loader


def test_cli_rejects_missing_file(patched_loader: dict[str, Any], tmp_path: Path):
    result = CliRunner().invoke(cli.app, ["analyze", str(tmp_path / "nope.wav"), "--lang", "en"])
    assert result.exit_code == 2


def test_cli_reports_missing_model_without_traceback(
    monkeypatch: pytest.MonkeyPatch, silent_wav: Path
):
    def missing(name: str, options: AsrSettings, **_: Any) -> Any:
        raise asr.ModelNotAvailable(f"{name}: run scripts/download_models.py --only {name}")

    monkeypatch.setattr(asr, "load_model", missing)
    result = CliRunner().invoke(cli.app, ["analyze", str(silent_wav), "--lang", "en"])
    assert result.exit_code == 1
    assert "download_models.py" in result.output
    assert "Traceback" not in result.output


def test_cli_rejects_a_file_without_audio(patched_loader: dict[str, Any], tmp_path: Path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not audio", encoding="utf-8")
    result = CliRunner().invoke(cli.app, ["analyze", str(bad), "--lang", "en"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
