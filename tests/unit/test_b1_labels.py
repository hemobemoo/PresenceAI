"""B1 pause labels from the owner's corrected transcript, and `listen` (ADR 0016 amendment).

Fake Whisper words with times, a corrected transcript, and the detected pauses: the label of a
pause is the owner's punctuation after their last word before it. No audio is played.
"""

import importlib.util
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from presenceai.evalconfig import REPO_ROOT, load_evaluation
from presenceai.evaluation import b1
from presenceai.schemas.assessment import Confidence
from presenceai.schemas.b1 import LabelledPause, Prefill
from presenceai.schemas.speech import AsrQuality, PauseEvent, PausePosition, SpeechReport

spec = importlib.util.spec_from_file_location(
    "b1_run_labels", REPO_ROOT / "experiments" / "b1_real_speech" / "run.py"
)
assert spec is not None and spec.loader is not None
run = importlib.util.module_from_spec(spec)
sys.modules["b1_run_labels"] = run
spec.loader.exec_module(run)

# Whisper heard (no punctuation, two words misheard): pauses after "results", "method", "well"
WHISPER = [
    ("Today", 0.0, 0.4), ("we", 0.5, 0.7), ("discuss", 0.8, 1.2), ("results", 1.3, 1.8),
    ("then", 2.7, 3.0), ("the", 3.1, 3.2), ("method", 3.3, 3.8),
    ("words", 4.6, 5.0), ("well", 5.1, 5.5),
    ("finally", 6.1, 6.6),
]  # fmt: skip
PAUSES = [(1.9, 2.6), (3.9, 4.5), (5.6, 6.0)]
MARGIN = load_evaluation().b1.labels.overlap_margin_s


def _report(sb: Any, words: list[tuple[str, float, float]], language: str = "en") -> SpeechReport:
    t = sb.transcript(words, language=language, duration_s=8.0)
    return SpeechReport(
        source_name="clip.wav",
        model="whisper_small",
        transcript=t,
        asr_quality=AsrQuality(word_count=len(t.words)),
        pauses=tuple(
            PauseEvent(
                start_s=a, end_s=b, position=PausePosition.UNKNOWN, confidence=Confidence.LOW
            )
            for a, b in PAUSES
        ),
    )


def _pauses() -> list[LabelledPause]:
    return [LabelledPause(start_s=a, end_s=b, label="?") for a, b in PAUSES]


def _labels(sb: Any, text: str, language: Any = "en", words: Any = None) -> list[Any]:
    report = _report(sb, words or WHISPER, language)
    return b1.label_pauses(_pauses(), b1.timed_tokens(text, report, language), margin_s=MARGIN)


def test_alignment_pairs_misheard_words_by_position():
    pairs = b1.align(["we", "discussed", "results"], ["we", "discuss", "results"])
    assert pairs == [(0, 0), (1, 1), (2, 2)]
    pairs = b1.align(["the", "um", "method"], ["the", "method"])  # inserted filler
    assert pairs == [(0, 0), (1, None), (2, 1)]
    pairs = b1.align(["works"], ["works", "well"])  # deleted word
    assert pairs == [(0, 0), (None, 1)]


def test_sentence_clause_and_mid_from_my_punctuation(sb: Any):
    # misheard: discussed/discuss, works/words; filler «um» added away from any pause;
    # "well" deleted (Whisper invented or misheard it), so "works." is the last word before pause 3
    text = "Today we discussed results. Then the um method, works. finally"
    labels = _labels(sb, text)
    assert [x.label for x in labels] == ["sentence", "clause", "sentence"]
    assert all(x.reason is None for x in labels)

    labels = _labels(sb, "Today we discussed results then the method works well finally")
    assert [x.label for x in labels] == ["mid", "mid", "mid"]


def test_arabic_marks(sb: Any):
    words = [("ما", 0.0, 0.4), ("النتيجة", 0.5, 1.8), ("نبدأ", 2.7, 3.8), ("الآن", 4.6, 5.5),
             ("ثم", 6.1, 6.6)]  # fmt: skip
    labels = _labels(sb, "ما النتيجة؟ نبدأ، الآن. ثم", "ar", words)
    assert [x.label for x in labels] == ["sentence", "clause", "sentence"]
    labels = _labels(sb, "ما النتيجة ؟ نبدأ ، الآن ثم", "ar", words)  # marks typed apart
    assert [x.label for x in labels] == ["sentence", "clause", "mid"]


def test_added_word_next_to_a_pause_is_ambiguous(sb: Any):
    # «um» has no timing and sits between "method" and "words": it could be on either side
    labels = _labels(sb, "Today we discussed results. Then the method. um works well finally")
    assert [x.label for x in labels] == ["sentence", "?", "mid"]
    q = labels[1]
    assert q.number == 2 and q.pause.start_s == 3.9
    assert q.before == ("Then", "the", "method.") and q.after == ("um", "works", "well")
    assert "added" in (q.reason or "")


def test_no_word_on_one_side_is_ambiguous(sb: Any):
    labels = _labels(sb, "Today we discussed results.", words=WHISPER[:4])
    assert [x.label for x in labels] == ["?", "?", "?"]


def test_a_word_inside_the_pause_is_ambiguous(sb: Any):
    words = [*WHISPER[:4], ("so", 2.1, 2.3), *WHISPER[4:]]
    labels = _labels(
        sb, "Today we discussed results. so then the method words well finally", words=words
    )
    assert labels[0].label == "?" and "inside" in (labels[0].reason or "")


# Whisper's Arabic word times often run into the silence next to a word, so a word's midpoint
# lands inside the VAD pause (B1 Arabic clips, 2026-11-19). Pauses: 1.9-2.6, 3.9-4.5, 5.6-6.0.
AR_OVERLAP = [
    ("ما", 0.0, 0.4),
    ("النتيجة", 0.5, 2.4),  # ends 0.5 s inside pause 1 (1.4 s before it, 0 after): before
    ("نبدأ", 2.5, 3.0),
    ("الآن", 3.1, 3.5),
    ("ثم", 3.9, 4.9),  # starts at pause 2's start (0 before, 0.4 after): after
    ("نرى", 5.0, 5.4),
    ("الفرق", 5.5, 6.1),  # 0.1 s outside the pause on each side: cannot tell
    ("اليوم", 6.2, 6.6),
]


def test_word_overlapping_a_pause_goes_to_the_side_where_more_of_it_lies(sb: Any):
    labels = _labels(sb, "ما النتيجة؟ نبدأ الآن، ثم نرى الفرق اليوم", "ar", AR_OVERLAP)
    assert [x.label for x in labels] == ["sentence", "clause", "?"]
    assert "both sides" in (labels[2].reason or "")
    assert [(p.pause.start_s, p.pause.end_s) for p in labels] == PAUSES  # times untouched


def test_side_of_a_pause_uses_the_margin():
    pause = LabelledPause(start_s=2.0, end_s=3.0, label="?")

    def side(a: float, b: float) -> str | None:
        return b1.side_of_pause(b1.Token(raw="w", norm="w", start_s=a, end_s=b), pause, 0.1)

    assert side(1.0, 1.9) == "before" and side(3.1, 3.5) == "after"  # no overlap
    assert side(1.4, 2.5) == "before"  # midpoint 1.95: the midpoint rule decides
    assert side(1.5, 2.9) == "before"  # midpoint inside; 0.5 s outside before, 0 after
    assert side(2.05, 3.3) == "after"  # 0 before, 0.3 after
    assert side(1.95, 3.2) == "after"  # 0.05 before, 0.2 after: 0.15 more after
    assert side(1.95, 3.1) is None  # 0.05 vs 0.1: within the margin
    assert side(2.2, 2.8) is None  # all inside the pause


# --- the label command ------------------------------------------------------------------------


@pytest.fixture
def clip(tmp_path: Path, sb: Any) -> Any:
    paths = run.Paths(data=tmp_path / "b1", results=tmp_path / "results")
    paths.audio.mkdir(parents=True)
    _write_wav(paths.audio / "b1_en_01_normal.wav", 8.0)
    report = _report(sb, WHISPER)
    meta = Prefill(model="whisper_small", config_hash="h1")
    annotation, labels = b1.prefill(report, "b1_en_01_normal", meta)
    paths.annotations.mkdir(parents=True)
    paths.reports.mkdir(parents=True)
    (paths.reports / "b1_en_01_normal.h1.json").write_text(report.model_dump_json(), "utf-8")
    fixed = annotation.model_copy(
        update={"transcript": "Today we discussed results. Then the method, works well finally"}
    )
    (paths.annotations / "b1_en_01_normal.yaml").write_text(b1.dump_annotation(fixed), "utf-8")
    (paths.annotations / "b1_en_01_normal.labels.txt").write_text(labels, "utf-8")
    return paths


def _read(paths: Any) -> tuple[Any, tuple[LabelledPause, ...]]:
    a = b1.load_annotation((paths.annotations / "b1_en_01_normal.yaml").read_text("utf-8"))
    labels = b1.read_labels((paths.annotations / "b1_en_01_normal.labels.txt").read_text("utf-8"))
    return a, labels


def test_label_fills_labels_and_keeps_times(clip: Any, capsys: pytest.CaptureFixture[str]):
    assert run.label_clip(clip, "b1_en_01_normal", force=False) == 0
    a, labels = _read(clip)
    assert [(p.start_s, p.end_s) for p in labels] == PAUSES  # times untouched, none added
    assert [p.label for p in labels] == ["sentence", "clause", "mid"]
    assert a.labels_method == "transcript_punctuation" and a.pause_edges_checked is False
    assert "3 pauses" in capsys.readouterr().out


def test_label_keeps_a_finished_file_unless_forced(clip: Any):
    run.label_clip(clip, "b1_en_01_normal", force=False)
    path = clip.annotations / "b1_en_01_normal.labels.txt"
    mine = path.read_text("utf-8").replace("\tmid\n", "\tsentence\n")  # decided by ear
    path.write_text(mine, "utf-8")
    run.label_clip(clip, "b1_en_01_normal", force=False)
    assert path.read_text("utf-8") == mine
    run.label_clip(clip, "b1_en_01_normal", force=True)
    assert path.read_text("utf-8").endswith("\tmid\n")


def test_label_redoes_a_file_with_question_marks(clip: Any, capsys: pytest.CaptureFixture[str]):
    yml = clip.annotations / "b1_en_01_normal.yaml"
    a, _ = _read(clip)
    yml.write_text(
        b1.dump_annotation(a.model_copy(update={"transcript": "Today we discussed results. "
                                                "Then the method. um works well finally"})),
        "utf-8",
    )  # fmt: skip
    assert run.label_clip(clip, "b1_en_01_normal", force=False) == 1
    out = capsys.readouterr().out
    assert "pause 2 at 00:03.90-00:04.50" in out and "method. | um works well" in out
    a, _ = _read(clip)
    fixed = a.model_copy(update={"transcript": "Today we discussed results. Then the method. "
                                               "works well finally"})  # fmt: skip
    yml.write_text(b1.dump_annotation(fixed), "utf-8")
    assert run.label_clip(clip, "b1_en_01_normal", force=False) == 0  # "?" present: allowed
    assert [p.label for p in _read(clip)[1]] == ["sentence", "sentence", "mid"]


def test_label_needs_the_prefill_report(clip: Any):
    (clip.reports / "b1_en_01_normal.h1.json").unlink()
    with pytest.raises(SystemExit, match="prefill --force"):
        run.label_clip(clip, "b1_en_01_normal", force=False)


# --- listen -----------------------------------------------------------------------------------

SR = 16_000


def _write_wav(path: Path, seconds: float) -> None:
    ramp = (np.arange(int(seconds * SR)) % 30_000).astype("<i2")  # sample i holds i % 30000
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(ramp.tobytes())


def _played(clip: Any, number: int) -> tuple[np.ndarray, Path]:
    heard: list[tuple[np.ndarray, Path]] = []

    def player(path: Path) -> None:
        with wave.open(str(path), "rb") as w:
            frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        heard.append((frames, path))

    run.cmd_listen(clip, "b1_en_01_normal", number, player=player)
    assert len(heard) == 1
    return heard[0]


def test_listen_plays_3_s_before_to_1_s_after(clip: Any):
    frames, path = _played(clip, 2)  # pause 3.9-4.5 s
    assert frames.size == round(5.5 * SR) - round(0.9 * SR)
    assert frames[0] == round(0.9 * SR) % 30_000
    assert not path.exists()  # the temporary slice is deleted


def test_listen_clamps_to_the_recording(clip: Any):
    labels = clip.annotations / "b1_en_01_normal.labels.txt"
    labels.write_text("1.000000\t1.500000\t?\n7.500000\t7.900000\t?\n", "utf-8")
    frames, _ = _played(clip, 1)
    assert frames[0] == 0 and frames.size == round(2.5 * SR)
    frames, _ = _played(clip, 2)
    assert frames.size == 8 * SR - round(4.5 * SR)


def test_listen_rejects_an_unknown_pause(clip: Any):
    with pytest.raises(SystemExit, match="1 to 3"):
        run.cmd_listen(clip, "b1_en_01_normal", 4, player=lambda _: None)


def test_score_skips_unfinished_labels_and_leaves_out_pause_f1(
    clip: Any, sb: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from presenceai.settings import Settings

    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    report = _report(sb, WHISPER)
    monkeypatch.setattr(run.Analyser, "report", lambda self, stem, path: report)
    yml = clip.annotations / "b1_en_01_normal.yaml"
    a, _ = _read(clip)
    yml.write_text(b1.dump_annotation(a.model_copy(update={"status": "corrected"})), "utf-8")

    assert run.cmd_score(clip, settings) == 1  # labels still from prefill
    assert "b1_en_01_normal (run label)" in capsys.readouterr().out

    run.label_clip(clip, "b1_en_01_normal", force=False)
    assert run.cmd_score(clip, settings) == 0
    out = capsys.readouterr().out
    assert "pause edges not hand-checked" in out
    summary = next(clip.results.glob("*_summary.json")).read_text("utf-8")
    assert '"pause_f1": {}' in summary and "Today" not in summary
