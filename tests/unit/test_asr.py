"""speech/asr.py with a fake faster-whisper model: no model files, no network."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from presenceai.schemas.speech import Transcript
from presenceai.settings import AsrSettings
from presenceai.speech import asr

OPTIONS = AsrSettings()


def test_transcribe_builds_text_and_exact_word_offsets(fake_whisper: Any, silent_wav: Path):
    t = asr.transcribe_file(fake_whisper(), silent_wav, language="en", options=OPTIONS)
    assert isinstance(t, Transcript)
    assert t.text == "Hello everyone. Today I will talk."
    for w in t.words:
        assert t.text[w.start_char : w.end_char] == w.text
    assert [w.text for w in t.words][:2] == ["Hello", "everyone."]
    assert t.words[0].start_s == 0.0 and t.words[-1].end_s == 3.6
    assert t.duration_s == 4.0 and t.language == "en"
    assert len(t.segments) == 2 and t.segments[1].no_speech_prob == 0.03


def test_transcribe_passes_an_array_language_and_word_timestamps(
    fake_whisper: Any, silent_wav: Path
):
    model = fake_whisper()
    asr.transcribe_file(model, silent_wav, language="ar", options=OPTIONS)
    (call,) = model.calls
    assert isinstance(call["audio"], np.ndarray) and call["audio"].dtype == np.float32
    opts = call["options"]
    assert opts["language"] == "ar"
    assert opts["word_timestamps"] is True
    assert opts["vad_filter"] is False  # VAD is our own step (slice 2), not faster-whisper's
    assert opts["beam_size"] == OPTIONS.beam_size
    assert opts["temperature"] == OPTIONS.temperature
    assert opts["condition_on_previous_text"] is OPTIONS.condition_on_previous_text


def test_segments_without_words_and_blank_words_are_skipped(
    fake_whisper: Any, whisper_parts: Any, silent_wav: Path
):
    W, S = whisper_parts.Word, whisper_parts.Segment
    segs = [
        S(0.0, 1.0, " ", None),
        S(1.0, 2.0, " Hi", [W(" ", 1.0, 1.1, 0.5), W(" Hi", 1.1, 1.5, 0.9)]),
    ]
    t = asr.transcribe_file(fake_whisper(segs), silent_wav, language="en", options=OPTIONS)
    assert t.text == "Hi"
    assert len(t.segments) == 1 and len(t.words) == 1


def test_silence_gives_an_empty_transcript(fake_whisper: Any, silent_wav: Path):
    t = asr.transcribe_file(fake_whisper([]), silent_wav, language="en", options=OPTIONS)
    assert t.text == "" and t.words == () and t.segments == ()


def test_word_end_before_start_is_clamped(fake_whisper: Any, whisper_parts: Any, silent_wav: Path):
    """faster-whisper occasionally returns end < start by a few ms; keep the word, fix the time."""
    W, S = whisper_parts.Word, whisper_parts.Segment
    segs = [S(0.0, 1.0, " a", [W(" a", 0.50, 0.49, 0.9)])]
    t = asr.transcribe_file(fake_whisper(segs), silent_wav, language="en", options=OPTIONS)
    assert t.words[0].start_s == t.words[0].end_s == 0.50


def test_quality_summary(fake_whisper: Any, silent_wav: Path):
    t = asr.transcribe_file(fake_whisper(), silent_wav, language="en", options=OPTIONS)
    q = asr.quality(t)
    assert q.word_count == 6
    assert q.mean_word_probability == pytest.approx((0.98 + 0.90 + 0.95 + 0.99 + 0.80 + 0.70) / 6)
    assert q.mean_no_speech_prob == pytest.approx((0.01 + 0.03) / 2)


def test_quality_of_empty_transcript_is_none(fake_whisper: Any, silent_wav: Path):
    t = asr.transcribe_file(fake_whisper([]), silent_wav, language="en", options=OPTIONS)
    q = asr.quality(t)
    assert q.word_count == 0 and q.mean_word_probability is None and q.mean_no_speech_prob is None


# --- model location: local folders only, never an implicit download -------------------------


def _models_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump({"asr": {"whisper_small": {"dir": "whisper-small-ct2-int8"}}}),
        encoding="utf-8",
    )
    return path


def test_model_dir_comes_from_models_yaml(tmp_path: Path):
    d = asr.model_dir("whisper_small", models_yaml=_models_yaml(tmp_path), models_root=tmp_path)
    assert d == tmp_path / "whisper-small-ct2-int8"


def test_unknown_model_name(tmp_path: Path):
    with pytest.raises(asr.ModelNotAvailable, match="unknown"):
        asr.model_dir("whisper_huge", models_yaml=_models_yaml(tmp_path), models_root=tmp_path)


def test_missing_model_says_how_to_download(tmp_path: Path):
    with pytest.raises(asr.ModelNotAvailable, match=r"download_models.py --only whisper_small"):
        asr.load_model(
            "whisper_small",
            OPTIONS,
            models_yaml=_models_yaml(tmp_path),
            models_root=tmp_path,
        )


def test_load_model_uses_the_local_folder_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    folder = tmp_path / "whisper-small-ct2-int8"
    folder.mkdir()
    for name in asr.REQUIRED_FILES:
        (folder / name).write_text("{}", encoding="utf-8")
    seen: dict[str, Any] = {}

    class FakeWhisperModelClass:
        def __init__(self, path: str, **kwargs: Any) -> None:
            seen["path"], seen["kwargs"] = path, kwargs

    monkeypatch.setattr(asr, "_whisper_model_class", lambda: FakeWhisperModelClass)
    asr.load_model(
        "whisper_small", OPTIONS, models_yaml=_models_yaml(tmp_path), models_root=tmp_path
    )
    assert seen["path"] == str(folder)
    assert seen["kwargs"]["local_files_only"] is True
    assert seen["kwargs"]["device"] == OPTIONS.device
    assert seen["kwargs"]["compute_type"] == OPTIONS.compute_type


def test_real_models_yaml_has_a_folder_for_each_asr_choice():
    for name in asr.MODEL_NAMES:
        assert asr.model_dir(name).name  # raises if the config entry or its `dir` is missing
