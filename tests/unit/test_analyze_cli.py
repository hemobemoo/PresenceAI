"""`presenceai analyze <file> --lang en|ar` with a fake model (speech/analyze.py + cli.py)."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from presenceai import cli
from presenceai.schemas.speech import PausePosition, PitchTrack, SpeechRegion, SpeechReport
from presenceai.settings import AsrSettings
from presenceai.speech import analyze, asr, pitch, vad

REGIONS = (SpeechRegion(start_s=0.0, end_s=1.2), SpeechRegion(start_s=2.0, end_s=3.6))


def fake_vad(samples: Any, opts: Any = None) -> tuple[SpeechRegion, ...]:
    return REGIONS


def fake_pitch(samples: Any, sr: int, cfg: Any) -> PitchTrack:
    n = 400  # 4 s at 10 ms, all unvoiced: no acoustic candidates
    return PitchTrack(
        hop_s=0.01, fmin_hz=65.0, fmax_hz=400.0, f0_hz=(0.0,) * n, voiced=(False,) * n
    )


def test_analyze_file_builds_a_report(fake_whisper: Any, silent_wav: Path, speech_cfg: Any):
    report = analyze.analyze_file(
        silent_wav,
        language="en",
        model=fake_whisper(),
        model_name="whisper_small",
        options=AsrSettings(),
        speech=speech_cfg,
        vad=fake_vad,
        pitch_tracker=fake_pitch,
    )
    assert isinstance(report, SpeechReport)
    assert report.schema_version == 5
    assert report.words_outside_speech == ()  # every word lies inside speech
    assert report.source_name == "talk.wav"
    assert report.model == "whisper_small"
    assert report.transcript.text.startswith("Hello everyone.")
    assert report.asr_quality.word_count == 6
    assert report.speech_regions == REGIONS
    (pause,) = report.pauses  # 0.8 s gap after "everyone."
    assert pause.position == PausePosition.SENTENCE_BOUNDARY
    assert report.measurements["pauses.count_ge_0.25s"].value == 1
    # only 3.6 s of speech: rate and pace are "not measurable", never a low score
    assert {"rate.wpm", "rate.articulation_wpm", "pace.syllables_per_s"} <= set(
        report.not_measurable
    )
    assert "rate.wpm" not in report.measurements
    # slice 3: no fillers in "Hello everyone. Today I will talk."; base model: "at least", no range
    assert report.fillers == () and report.repetitions == ()
    assert report.measurements["fillers.count"].value == 0
    assert report.measurements["fillers.per_100_words"].method.startswith("at least 0.0")
    assert "fillers.per_100_words_upper" not in report.measurements
    assert report.pitch is not None and "pitch" not in report.model_dump()


@pytest.fixture
def patched_loader(monkeypatch: pytest.MonkeyPatch, fake_whisper: Any) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_load(name: str, options: AsrSettings, **_: Any) -> Any:
        seen["name"], seen["options"] = name, options
        return fake_whisper()

    monkeypatch.setattr(asr, "load_model", fake_load)
    monkeypatch.setattr(vad, "detect_speech", fake_vad)
    monkeypatch.setattr(pitch, "track", fake_pitch)
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


def test_pyin_runs_once_per_recording(fake_whisper: Any, silent_wav: Path, speech_cfg: Any):
    calls: list[int] = []

    def counting(samples: Any, sr: int, cfg: Any) -> PitchTrack:
        calls.append(1)
        return fake_pitch(samples, sr, cfg)

    analyze.analyze_file(
        silent_wav, language="en", model=fake_whisper(), model_name="whisper_small",
        options=AsrSettings(), speech=speech_cfg, vad=fake_vad, pitch_tracker=counting,
    )  # fmt: skip
    assert calls == [1]


def test_uploaded_files_have_unknown_capture_processing(
    patched_loader: dict[str, Any], silent_wav: Path
):
    result = CliRunner().invoke(cli.app, ["analyze", str(silent_wav), "--lang", "en"])
    report = SpeechReport.model_validate_json(result.stdout)
    assert report.capture.auto_gain_control == "unknown"
    assert report.capture.noise_suppression == "unknown"
    assert report.capture.echo_cancellation == "unknown"


def test_capture_processing_can_be_declared_off(patched_loader: dict[str, Any], silent_wav: Path):
    result = CliRunner().invoke(
        cli.app, ["analyze", str(silent_wav), "--lang", "en", "--capture-processing", "off"]
    )
    assert result.exit_code == 0, result.output
    report = SpeechReport.model_validate_json(result.stdout)
    assert report.capture.all_off


def test_baseline_report_is_read(patched_loader: dict[str, Any], silent_wav: Path, tmp_path: Path):
    base = tmp_path / "base.json"
    first = CliRunner().invoke(
        cli.app, ["analyze", str(silent_wav), "--lang", "en", "--out", str(base)]
    )
    assert first.exit_code == 0, first.output
    result = CliRunner().invoke(
        cli.app, ["analyze", str(silent_wav), "--lang", "en", "--baseline", str(base)]
    )
    assert result.exit_code == 0, result.output
    report = SpeechReport.model_validate_json(result.stdout)
    assert report.baseline_source == "base.json"


def test_bad_baseline_file_is_rejected(
    patched_loader: dict[str, Any], silent_wav: Path, tmp_path: Path
):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    result = CliRunner().invoke(
        cli.app, ["analyze", str(silent_wav), "--lang", "en", "--baseline", str(bad)]
    )
    assert result.exit_code == 1 and "Traceback" not in result.output


def test_words_after_the_last_speech_are_left_out_of_the_report(
    fake_whisper: Any, whisper_parts: Any, silent_wav: Path, speech_cfg: Any
):
    """ADR 0015: an invented "Thank you." 0.3 s after the last speech is removed before any
    measure or quote, and listed apart."""
    segs = [*fake_whisper().segments, whisper_parts.Segment(
        3.9, 4.0, " Thank you.",
        [whisper_parts.Word(" Thank", 3.9, 4.0, 0.2), whisper_parts.Word(" you.", 4.0, 4.0, 0.9)],
    )]  # fmt: skip
    report = analyze.analyze_file(
        silent_wav, language="en", model=fake_whisper(segs), model_name="whisper_small",
        options=AsrSettings(), speech=speech_cfg, vad=fake_vad, pitch_tracker=fake_pitch,
    )  # fmt: skip
    assert report.transcript.text == "Hello everyone. Today I will talk."
    assert [w.text for w in report.words_outside_speech] == ["Thank", "you."]
    assert report.asr_quality.word_count == 6
    assert "transcript" not in report.not_measurable


def test_no_speech_is_said_in_the_report(fake_whisper: Any, silent_wav: Path, speech_cfg: Any):
    report = analyze.analyze_file(
        silent_wav, language="en", model=fake_whisper(), model_name="whisper_small",
        options=AsrSettings(), speech=speech_cfg, vad=lambda _: (), pitch_tracker=fake_pitch,
    )  # fmt: skip
    assert report.not_measurable["transcript"] == "no speech detected in this recording"
    assert report.transcript.text == "" and report.transcript.words == ()
    assert len(report.words_outside_speech) == 6
    assert report.asr_quality.word_count == 0
