"""experiments/e2_rate_pauses/run.py: the E2 runner on fake clips (no model, no network).

Clips come from the C1 construction with a fake TTS; the transcriber gives one word per true
word, spread over the true speech spans; the VAD returns the true speech spans."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fake_tts import FakeTTS

from presenceai.evalconfig import REPO_ROOT, load_evaluation
from presenceai.schemas.c1 import ClipTruth
from presenceai.schemas.speech import SpeechRegion, Transcript
from presenceai.settings import Settings
from presenceai.synth import c1, scripts

spec = importlib.util.spec_from_file_location(
    "e2_run", REPO_ROOT / "experiments" / "e2_rate_pauses" / "run.py"
)
assert spec is not None and spec.loader is not None
run = importlib.util.module_from_spec(spec)
sys.modules["e2_run"] = run
spec.loader.exec_module(run)

EV = load_evaluation()
SPEECH = Settings(_env_file=None).speech  # pyright: ignore[reportCallIssue]
TEXT = (
    "Good morning, everyone. Today I want to share | a simple idea about planning. "
    "Small steps help us | finish big tasks. "
) * 5


def build(tmp: Path) -> tuple[Any, list[ClipTruth]]:
    paths = run.Paths("t", data=tmp / "data", results_c1=tmp / "c1", results_e2=tmp / "e2")
    paths.clips.mkdir(parents=True)
    paths.truth.mkdir(parents=True)
    cfg = EV.c1.model_copy(update={"clip_s": (1.0, 500.0)})
    truths = []
    for i in range(4):
        s = scripts.parse_script(f"s{i + 1:02d}", "af_heart", "T", TEXT)
        plan = c1.plan_clip(s, 0, 150.0, c1.clip_rng(cfg, i, 0), cfg)
        res = c1.render(s, plan, FakeTTS(), cfg, 3.0)
        assert res.truth is not None and res.clean is not None
        audio = c1.add_noise_floor(res.clean, cfg.noise_floor_dbfs, c1.clip_rng(cfg, i, 0))
        sf.write(paths.clips / f"{plan.clip_id}.flac", audio, 24_000, subtype="PCM_16")
        (paths.truth / f"{plan.clip_id}.json").write_text(res.truth.model_dump_json())
        truths.append(res.truth)
    with paths.split_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "script_id", "voice", "part", "wpm_true", "rate_band"])
        for k, t in enumerate(truths):
            w.writerow([t.clip_id, t.script_id, t.voice, "dev" if k == 0 else "test", t.wpm, ""])
    return paths, truths


def fake_transcriber(truths: list[ClipTruth]) -> Any:
    by_name = {f"{t.clip_id}.flac": t for t in truths}

    def transcribe(path: Path) -> Transcript:
        from speech_builders import transcript

        t = by_name[path.name]
        tokens = []
        groups = np.array_split(np.arange(t.words), len(t.speech_spans))
        for (a, b), group in zip(t.speech_spans, groups, strict=True):
            n = max(1, group.size)
            for j, k in enumerate(group):
                ws = a + (b - a) * j / n
                tokens.append((f"w{k}", ws, ws + (b - a) / n * 0.9))
        return transcript(tokens, duration_s=t.duration_s)

    return transcribe


def truth_vad(truths: list[ClipTruth]) -> Any:
    def vad(samples: np.ndarray) -> tuple[SpeechRegion, ...]:
        dur = samples.size / 16_000
        t = min(truths, key=lambda x: abs(x.duration_s - dur))
        return tuple(SpeechRegion(start_s=a, end_s=b) for a, b in t.speech_spans)

    return vad


def test_evaluate_scores_only_the_requested_part(tmp_path: Path):
    paths, truths = build(tmp_path)
    scores, extras, meta = run.evaluate(
        paths, "test", "clean", transcribe=fake_transcriber(truths), vad=truth_vad(truths),
        ev=EV, speech=SPEECH, progress=lambda _: None,
    )  # fmt: skip
    assert len(scores) == 3 and all(s.part == "test" for s in scores)
    for s in scores:
        assert s.wpm_error_pct is not None and s.wpm_error_pct < 1.0  # VAD = truth
        f1_all = {t.threshold_s: (t.tp_all, t.fp_all, t.fn_all) for t in s.thresholds}
        assert f1_all[1.0][1] == 0 and f1_all[1.0][2] == 0  # every true pause found
    assert "syllable_rate_vs_wpm_true_r" in meta and len(extras) == 3
    assert "syllable_rate_vs_articulation_true_r" in meta
    assert all(x["articulation_wpm_true"] > 0 for x in extras)


def test_noise_clips_are_written_once_at_the_requested_snr(tmp_path: Path):
    paths, truths = build(tmp_path)
    t = truths[1]
    out = run.noisy_clip(paths, t, 1, "snr10", EV)
    again = run.noisy_clip(paths, t, 1, "snr10", EV)
    assert out == again and out.parent.name == "snr10"
    clean, sr = sf.read(paths.clips / f"{t.clip_id}.flac", dtype="float64")
    noisy, _ = sf.read(out, dtype="float64")
    mask = c1.span_mask(t.speech_spans, sr, clean.size)
    p_s = np.mean(clean[mask] ** 2)
    p_n = np.mean((noisy - clean) ** 2)
    assert 10 * np.log10(p_s / p_n) == pytest.approx(10.0, abs=0.3)  # 16-bit rounding
    with pytest.raises(ValueError, match="snr_db"):
        run.noisy_clip(paths, t, 1, "snr5", EV)


def test_the_clean_test_part_is_scored_once_per_config(tmp_path: Path):
    paths, _ = build(tmp_path)
    run.record(paths, {"run": "t", "part": "test", "condition": "clean", "config_hash": "abc",
                       "finished": "2026-11-17T00:00:00+00:00"})  # fmt: skip
    with pytest.raises(RuntimeError, match="already scored"):
        run.check_ledger(paths, "test", "clean", "abc", None)
    run.check_ledger(paths, "test", "clean", "abc", "VAD bug fixed")  # a recorded reason
    run.check_ledger(paths, "test", "clean", "other", None)  # a new config version
    run.check_ledger(paths, "test", "snr20", "abc", None)  # noise runs are descriptive
    run.check_ledger(paths, "dev", "clean", "abc", None)  # dev is for tuning


def test_results_files_are_written(tmp_path: Path):
    from presenceai.evaluation import e2

    paths, truths = build(tmp_path)
    scores, extras, meta = run.evaluate(
        paths, "test", "clean", transcribe=fake_transcriber(truths), vad=truth_vad(truths),
        ev=EV, speech=SPEECH, progress=lambda _: None,
    )  # fmt: skip
    out = run.write_results(paths, "test", "clean", scores, extras, e2.summarise(scores, EV), meta)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["part"] == "test" and "passes" in data["summary"]
    rows = list(
        csv.DictReader(out.with_name(out.name.replace("_summary.json", "_clips.csv")).open())
    )
    assert len(rows) == 3 and "excluded_0.5" in rows[0] and "fp_all_1" in rows[0]


def test_config_hash_changes_with_vad_values():
    a = run.config_hash(EV, SPEECH, Settings(_env_file=None).asr, "whisper_small")  # pyright: ignore[reportCallIssue]
    vad = SPEECH.vad.model_copy(update={"min_speech_duration_ms": 100})
    b = run.config_hash(EV, SPEECH.model_copy(update={"vad": vad}),
                        Settings(_env_file=None).asr, "whisper_small")  # pyright: ignore[reportCallIssue]  # fmt: skip
    assert a != b
