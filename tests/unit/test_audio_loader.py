"""ADR 0006: audio reaches faster-whisper only as an array, through presenceai.speech.audio."""

import ast
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from presenceai.speech import audio

REPO = Path(__file__).resolve().parents[2]
LOADER = REPO / "src" / "presenceai" / "speech" / "audio.py"
SCANNED = ("src", "experiments", "scripts")


def _write_wav(path: Path, samples: np.ndarray, rate: int, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype("<i2").tobytes())


def test_load_audio_returns_the_same_samples_for_16k_mono(tmp_path: Path):
    pcm = np.array([0, 16384, -16384, 32767, -32768] * 100, dtype=np.int16)
    _write_wav(tmp_path / "a.wav", pcm, 16_000)
    out = audio.load_audio(tmp_path / "a.wav")
    assert out.dtype == np.float32 and out.ndim == 1
    np.testing.assert_allclose(out, pcm.astype(np.float32) / 32768.0, atol=1e-6)


def test_load_audio_resamples_and_downmixes_to_16k_mono(tmp_path: Path):
    t = np.arange(44_100) / 44_100  # 1 s of 440 Hz, stereo, 44.1 kHz
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    _write_wav(tmp_path / "s.wav", np.repeat(tone, 2), 44_100, channels=2)
    out = audio.load_audio(str(tmp_path / "s.wav"))
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(len(out) - audio.SAMPLE_RATE) <= 32  # 1 s at 16 kHz
    assert 0.3 < float(np.abs(out).max()) <= 1.0


def test_transcribe_gives_the_model_an_array_never_a_path(tmp_path: Path):
    _write_wav(tmp_path / "a.wav", np.zeros(1600, dtype=np.int16), 16_000)
    seen: dict[str, Any] = {}

    class FakeModel:
        def transcribe(self, audio_in: Any, **options: Any):
            seen["audio"], seen["options"] = audio_in, options
            return iter(["seg1", "seg2"]), "info"

    segments, info = audio.transcribe(FakeModel(), tmp_path / "a.wav", language="en")
    assert isinstance(seen["audio"], np.ndarray) and seen["audio"].dtype == np.float32
    assert seen["options"] == {"language": "en"}
    assert segments == ["seg1", "seg2"] and info == "info"


def _python_files() -> list[Path]:
    files: list[Path] = []
    for top in SCANNED:
        files += [p for p in (REPO / top).rglob("*.py") if ".venv" not in p.parts]
    return files


def violations(source: str, rel: str, is_loader: bool = False) -> list[str]:
    """Calls or imports that could hand faster-whisper a file path (ADR 0006).

    Allowed: `.transcribe(` inside the shared loader, and `audio.transcribe(...)` (the loader's
    own function, imported as `from presenceai.speech import audio`).
    """
    found = []
    for node in ast.walk(ast.parse(source, filename=rel)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "transcribe"
            and not is_loader
            and not (isinstance(node.func.value, ast.Name) and node.func.value.id == "audio")
        ):
            found.append(f"{rel}:{node.lineno} calls .transcribe( directly")
        if (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("faster_whisper")
            and (
                (node.module or "").endswith(".audio")
                or any(a.name == "decode_audio" for a in node.names)
            )
        ):
            found.append(f"{rel}:{node.lineno} imports faster-whisper's decoder")
        if isinstance(node, ast.Attribute) and node.attr == "decode_audio":
            found.append(f"{rel}:{node.lineno} uses decode_audio")
    return found


def test_only_the_shared_loader_calls_transcribe_or_decodes_for_faster_whisper():
    """Fails CI if code anywhere else could hand faster-whisper a file path."""
    problems = []
    for path in _python_files():
        rel = path.relative_to(REPO).as_posix()
        problems += violations(path.read_text(encoding="utf-8"), rel, is_loader=path == LOADER)
    assert not problems, "Use presenceai.speech.audio.transcribe (ADR 0006):\n" + "\n".join(
        problems
    )


@pytest.mark.parametrize(
    ("source", "bad"),
    [
        ("model.transcribe('clip.wav', language='en')", True),
        ("seg, info = whisper.transcribe(str(path))", True),
        ("from faster_whisper.audio import decode_audio", True),
        ("from faster_whisper import decode_audio", True),
        ("x = faster_whisper.decode_audio(p)", True),
        ("segments, info = audio.transcribe(model, path, language='en')", False),
        ("from faster_whisper import WhisperModel", False),
    ],
)
def test_the_guard_rule(source: str, bad: bool):
    assert bool(violations(source, "snippet.py")) is bad


def test_the_scan_covers_the_known_caller():
    scanned = {p.relative_to(REPO).as_posix() for p in _python_files()}
    assert "experiments/s3_unsloth_whisper/check_local.py" in scanned
    assert "src/presenceai/speech/audio.py" in scanned


def test_load_audio_rejects_a_file_without_audio(tmp_path: Path):
    junk = tmp_path / "not_audio.wav"
    junk.write_bytes(b"this is not audio")
    with pytest.raises(Exception):  # noqa: B017 (PyAV raises its own error types)
        audio.load_audio(junk)
