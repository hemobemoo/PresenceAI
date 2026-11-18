"""B1 annotation and scoring (ADR 0016): pre-filled files, Audacity labels, descriptive scores.

No real recordings: reports come from the fake Whisper model and silent audio.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from presenceai.evalconfig import REPO_ROOT, load_evaluation
from presenceai.evaluation import b1
from presenceai.schemas.b1 import B1Annotation, LabelledPause, Prefill
from presenceai.schemas.speech import SpeechRegion
from presenceai.settings import AsrSettings, Settings
from presenceai.speech import analyze

EV = load_evaluation()
SETTINGS = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
META = Prefill(model="whisper_small", config_hash="abc", commit=None)

spec = importlib.util.spec_from_file_location(
    "b1_run", REPO_ROOT / "experiments" / "b1_real_speech" / "run.py"
)
assert spec is not None and spec.loader is not None
run = importlib.util.module_from_spec(spec)
sys.modules["b1_run"] = run
spec.loader.exec_module(run)


def _report(fake_whisper: Any, wav: Path, language: str = "en", segments: Any = None) -> Any:
    return analyze.analyze_file(
        wav,
        language=language,  # type: ignore[arg-type]
        model=fake_whisper(segments, language=language),
        model_name="whisper_small",
        options=AsrSettings(),
        speech=SETTINGS.speech,
        vad=lambda _: (SpeechRegion(start_s=0.0, end_s=1.2), SpeechRegion(start_s=2.0, end_s=3.6)),
    )


def test_twenty_planned_clips_with_fixed_names():
    stems = b1.expected_stems()
    assert len(stems) == 20 and stems[0] == "b1_en_01_normal" and stems[-1] == "b1_ar_10_noise"
    assert b1.parse_stem("b1_ar_07_slow") == ("ar", 7)
    for bad in ("b1_ar_07_fast", "b1_fr_01_normal", "talk", "b1_en_11_normal"):
        with pytest.raises(ValueError):
            b1.parse_stem(bad)


def test_prefill_copies_whisper_text_and_pauses(fake_whisper: Any, silent_wav: Path):
    report = _report(fake_whisper, silent_wav)
    annotation, labels = b1.prefill(report, "b1_en_01_normal", META)
    assert annotation.status == "prefilled"
    assert annotation.transcript == annotation.whisper_text == "Hello everyone. Today I will talk."
    assert (annotation.speech_start_s, annotation.speech_end_s) == (0.0, 3.6)
    pauses = b1.read_labels(labels)
    assert [(p.start_s, p.end_s, p.label) for p in pauses] == [(1.2, 2.0, "sentence")]


def test_arabic_flags_and_undecided_labels(fake_whisper: Any, silent_wav: Path, whisper_parts: Any):
    seg = whisper_parts.Segment(
        0.0, 3.6, "", [whisper_parts.Word(" مرحبا", 0.0, 1.0, 0.9),
                       whisper_parts.Word(" بكم", 2.1, 3.0, 0.9)],
    )  # fmt: skip
    report = _report(fake_whisper, silent_wav, "ar", [seg])
    annotation, labels = b1.prefill(report, "b1_ar_05_defense", META)
    assert annotation.code_switching and not annotation.long_sentences
    assert b1.read_labels(labels)[0].label == "?"  # no punctuation: position unknown
    again = b1.load_annotation(b1.dump_annotation(annotation))
    assert again == annotation  # Arabic survives the YAML round trip


def test_english_clips_carry_no_arabic_flags(fake_whisper: Any, silent_wav: Path):
    annotation, _ = b1.prefill(_report(fake_whisper, silent_wav), "b1_en_06_defense", META)
    assert not annotation.code_switching and not annotation.long_sentences


def test_audacity_labels_round_trip_and_skip_frequency_lines():
    items = [LabelledPause(start_s=1.5, end_s=2.25, label="mid")]
    text = b1.write_labels(items) + "\\\t100.0\t2000.0\n"
    assert b1.read_labels(text) == tuple(items)
    with pytest.raises(ValueError, match="tabs"):
        b1.read_labels("1.0 2.0 mid\n")
    with pytest.raises(ValueError, match="sentence, clause or mid"):
        b1.read_labels("1.0\t2.0\tlong\n")


def test_corrected_needs_every_removed_word_marked():
    base = {
        "clip": "b1_en_01_normal.wav", "language": "en", "number": 1, "clip_type": "normal",
        "prefill": META.model_dump(), "speech_start_s": 0.0, "speech_end_s": 3.0,
        "whisper_text": "a", "transcript": "a",
        "removed_at_end": [{"text": "you.", "start_s": 3.5, "end_s": 3.8}],
    }  # fmt: skip
    B1Annotation.model_validate(base)
    with pytest.raises(ValueError, match="real"):
        B1Annotation.model_validate(base | {"status": "corrected"})


def test_wer():
    assert b1.wer(["a", "b", "c"], ["a", "c"]) == pytest.approx(1 / 3)
    assert b1.wer([], ["a"]) is None
    assert b1.words("Hello, everyone.", "en") == ["hello", "everyone"]


def _corrected(fake_whisper: Any, silent_wav: Path) -> tuple[Any, Any]:
    report = _report(fake_whisper, silent_wav)
    annotation, _ = b1.prefill(report, "b1_en_09_fillers", META)
    fixed = annotation.model_copy(
        update={
            "status": "corrected",
            "transcript": "Hello um everyone. Today I I will talk.",
            "speech_start_s": 0.0,
            "speech_end_s": 3.6,
        }
    )
    return report, fixed


def test_score_clip(fake_whisper: Any, silent_wav: Path):
    report, fixed = _corrected(fake_whisper, silent_wav)
    labels = [LabelledPause(start_s=1.25, end_s=2.0, label="sentence")]
    s = b1.score_clip(fixed, labels, report, EV, SETTINGS.speech.pauses, pileup_ratio_max=1.1)
    assert s.words_true == 8 and s.words_true_no_hesitations == 7 and s.words_measured == 6
    assert s.wpm_true == pytest.approx(8 / 3.6 * 60)
    assert s.wer == pytest.approx(2 / 8)  # "um" and the second "I" were dropped
    assert s.wer_no_hesitations == pytest.approx(1 / 7)
    assert s.true_pauses == 1 and s.tp["0.5"] == 1 and s.fn["1.0"] == 0
    assert s.positions == (("sentence_boundary", "sentence_boundary"),)
    assert s.duration_abs_errors_s == (pytest.approx(0.05),)
    assert s.pileup_ratio_max == 1.1 and s.language_mismatch is False
    summary = b1.summarise([s])
    assert summary["pause_f1"]["0.5"] == 1.0 and summary["removed_real"] == 0
    json.dumps(summary)  # results are plain numbers


def test_scores_hold_no_transcript_text(fake_whisper: Any, silent_wav: Path):
    report, fixed = _corrected(fake_whisper, silent_wav)
    s = b1.score_clip(fixed, [], report, EV, SETTINGS.speech.pauses, pileup_ratio_max=None)
    dumped = s.model_dump_json()
    for word in ("Hello", "everyone", "Today", "will"):
        assert word not in dumped


def test_score_refuses_unfinished_annotations(fake_whisper: Any, silent_wav: Path):
    report, fixed = _corrected(fake_whisper, silent_wav)
    with pytest.raises(ValueError, match="not corrected"):
        b1.score_clip(
            fixed.model_copy(update={"status": "prefilled"}), [], report, EV,
            SETTINGS.speech.pauses, pileup_ratio_max=None,
        )  # fmt: skip
    with pytest.raises(ValueError, match="'\\?'"):
        b1.score_clip(
            fixed, [LabelledPause(start_s=1.2, end_s=2.0, label="?")], report, EV,
            SETTINGS.speech.pauses, pileup_ratio_max=None,
        )  # fmt: skip


def test_prefill_never_overwrites_corrected_work(
    tmp_path: Path, fake_whisper: Any, silent_wav: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = run.Paths(data=tmp_path / "b1", results=tmp_path / "results")
    paths.audio.mkdir(parents=True)
    (paths.audio / "b1_en_01_normal.wav").write_bytes(silent_wav.read_bytes())
    (paths.audio / "b1_en_02_normal.wav").write_bytes(silent_wav.read_bytes())
    report = _report(fake_whisper, silent_wav)
    monkeypatch.setattr(run.Analyser, "report", lambda self, stem, clip: report)

    assert run.cmd_prefill(paths, SETTINGS, force=False) == 0
    first = paths.annotations / "b1_en_01_normal.yaml"
    annotation = b1.load_annotation(first.read_text(encoding="utf-8"))
    done = annotation.model_copy(update={"status": "corrected", "transcript": "my words"})
    first.write_text(b1.dump_annotation(done), encoding="utf-8")

    assert run.cmd_prefill(paths, SETTINGS, force=True) == 0
    assert b1.load_annotation(first.read_text(encoding="utf-8")).transcript == "my words"
    assert (paths.annotations / "b1_en_02_normal.labels.txt").is_file()


def test_misnamed_audio_stops_the_run(tmp_path: Path):
    paths = run.Paths(data=tmp_path / "b1", results=tmp_path / "results")
    paths.audio.mkdir(parents=True)
    (paths.audio / "recording 1.wav").write_bytes(b"")
    with pytest.raises(ValueError, match=re.escape("b1_<en|ar>")):
        run.clips(paths)


def test_data_folder_is_git_ignored():
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/*" in ignore and "*.wav" in ignore
