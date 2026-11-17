"""E2: speech rate and pause accuracy on C1 (ADR 0009 targets, ADR 0011, ADR 0014).

    uv run python experiments/e2_rate_pauses/run.py split --run f1
    uv run python experiments/e2_rate_pauses/run.py score --run f1 --part dev
    uv run python experiments/e2_rate_pauses/run.py score --run f1 --part test
    uv run python experiments/e2_rate_pauses/run.py score --run f1 --part test --condition snr20

- The app's own functions: `asr.transcribe_file` (whisper_small, settings from
  configs/default.yaml), `vad.detect_speech`, `pauses.find_pauses`, `rate.word_rate`, plus
  `rate.syllable_rate` for the live-pace correlation. pYIN is not needed.
- Transcripts are cached per clip file, model and ASR options (data/c1/<run>/cache/), so a VAD
  change re-runs only VAD and pauses. Words outside speech are removed after VAD, as in the app
  (`trim.trim_to_speech`, ADR 0015); the cache holds the untrimmed transcript.
- `split` writes experiments/results/c1/<run>_split.csv and prints counts per part by band and
  voice; it stops on an empty cell.
- `score --part test --condition clean` is the pass/fail run. It is refused a second time for
  the same config (ledger: experiments/results/e2/ledger.json) unless `--rerun-reason` says why.
- Noise conditions (`snr20`, `snr10`) mix seeded pink noise into the test clips at the given
  speech-to-noise ratio over the true speech spans; descriptive only, no pass target.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from presenceai.evalconfig import REPO_ROOT, EvaluationSettings, load_evaluation
from presenceai.evaluation import e2
from presenceai.schemas.c1 import ClipTruth, E2ClipScore, Part
from presenceai.schemas.speech import PauseEvent, SpeechRegion, Transcript
from presenceai.settings import AsrSettings, SpeechSettings
from presenceai.synth import c1, scripts, split

Transcriber = Callable[[Path], Transcript]
Vad = Callable[[np.ndarray], tuple[SpeechRegion, ...]]


@dataclass(frozen=True)
class Paths:
    run: str
    data: Path = REPO_ROOT / "data" / "c1"
    results_c1: Path = REPO_ROOT / "experiments" / "results" / "c1"
    results_e2: Path = REPO_ROOT / "experiments" / "results" / "e2"

    @property
    def clips(self) -> Path:
        return self.data / self.run / "clips"

    @property
    def truth(self) -> Path:
        return self.results_c1 / self.run / "truth"

    @property
    def split_csv(self) -> Path:
        return self.results_c1 / f"{self.run}_split.csv"

    def noisy(self, condition: str) -> Path:
        return self.data / self.run / condition

    def cache(self, key: str) -> Path:
        return self.data / self.run / "cache" / key

    @property
    def ledger(self) -> Path:
        return self.results_e2 / "ledger.json"


def load_truths(paths: Paths) -> list[ClipTruth]:
    return [
        ClipTruth.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(paths.truth.glob("*.json"))
    ]


# --- split ---------------------------------------------------------------------------------


def make_split(paths: Paths, ev: EvaluationSettings) -> str:
    scr = scripts.load_scripts(REPO_ROOT / ev.c1.scripts)
    parts = split.split_scripts(scr, ev.e2.split)
    table = split.assign(load_truths(paths), parts)
    counts = split.counts(table, ev.c1)
    text = split.format_counts(counts)
    split.check_counts(counts, ev.c1)  # stops on an empty cell
    paths.split_csv.parent.mkdir(parents=True, exist_ok=True)
    with paths.split_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "script_id", "voice", "part", "wpm_true", "rate_band"])
        for t, part in table:
            band = c1.rate_band(t.wpm, ev.c1.rate_bands) or "outside"
            w.writerow([t.clip_id, t.script_id, t.voice, part, f"{t.wpm:.2f}", band])
    return text


def read_split(paths: Paths) -> dict[str, Part]:
    with paths.split_csv.open(encoding="utf-8") as fh:
        return {r["clip_id"]: r["part"] for r in csv.DictReader(fh)}  # type: ignore[misc]


# --- noise ---------------------------------------------------------------------------------


def snr_of(condition: str) -> float | None:
    if condition == "clean":
        return None
    if not condition.startswith("snr"):
        raise ValueError(f"unknown condition {condition!r}: clean, snr20 or snr10")
    return float(condition[3:])


def noisy_clip(paths: Paths, truth: ClipTruth, index: int, condition: str,
               ev: EvaluationSettings) -> Path:  # fmt: skip
    """The test clip with pink noise at the condition's SNR (written once, seeded per clip)."""
    import soundfile as sf

    snr = snr_of(condition)
    src = paths.clips / f"{truth.clip_id}.flac"
    if snr is None:
        return src
    if snr not in ev.e2.noise.snr_db:
        raise ValueError(f"{condition}: not in configs/evaluation.yaml e2.noise.snr_db")
    out = paths.noisy(condition) / src.name
    if not out.exists():
        x, sr = sf.read(src, dtype="float32")
        rng = np.random.default_rng([ev.e2.noise.seed, index])
        y = c1.mix_noise(x, truth.speech_spans, sr, snr, rng, ev.e2.noise.color)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, np.clip(y, -1, 1), sr, subtype="PCM_16")
    return out


# --- scoring -------------------------------------------------------------------------------


def config_hash(ev: EvaluationSettings, speech: SpeechSettings, asr: AsrSettings,
                model: str) -> str:  # fmt: skip
    blob = json.dumps(
        {
            "e2": ev.e2.model_dump(mode="json"),
            "vad": speech.vad.model_dump(mode="json"),
            "transcript": speech.transcript.model_dump(mode="json"),
            "pauses": speech.pauses.model_dump(mode="json"),
            "rate": speech.rate.model_dump(mode="json"),
            "asr": asr.model_dump(mode="json"),
            "model": model,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _ledger(paths: Paths) -> list[dict[str, Any]]:
    return json.loads(paths.ledger.read_text(encoding="utf-8")) if paths.ledger.exists() else []


def check_ledger(paths: Paths, part: Part, condition: str, cfg_hash: str,
                 rerun_reason: str | None) -> None:  # fmt: skip
    if part != "test" or condition != "clean":
        return
    done = [e for e in _ledger(paths) if e["run"] == paths.run and e["part"] == "test"
            and e["condition"] == "clean" and e["config_hash"] == cfg_hash]  # fmt: skip
    if done and not rerun_reason:
        raise RuntimeError(
            f"the test part was already scored with config {cfg_hash} "
            f"({done[0]['finished']}); ADR 0014 allows one test run per config version. "
            "Pass --rerun-reason to record why a second run is needed."
        )


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    a, b = np.asarray(xs, float), np.asarray(ys, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(
    paths: Paths,
    part: Part,
    condition: str,
    *,
    transcribe: Transcriber,
    vad: Vad,
    ev: EvaluationSettings,
    speech: SpeechSettings,
    progress: Callable[[str], None] = print,
) -> tuple[list[E2ClipScore], list[dict[str, Any]], dict[str, Any]]:
    from presenceai.schemas.speech import CaptureProcessing
    from presenceai.speech import audio, loudness, pauses, rate, trim

    parts = read_split(paths)
    truths = [t for t in load_truths(paths) if parts.get(t.clip_id) == part]
    if not truths:
        raise RuntimeError(f"no clips in the {part} part of run {paths.run}")
    index = {t.clip_id: i for i, t in enumerate(load_truths(paths))}
    scores: list[E2ClipScore] = []
    extras: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for k, truth in enumerate(truths, 1):
        path = noisy_clip(paths, truth, index[truth.clip_id], condition, ev)
        transcript = transcribe(path)
        samples = audio.load_audio(path)
        regions = vad(samples)
        transcript, outside = trim.trim_to_speech(transcript, regions, speech.transcript)
        events: tuple[PauseEvent, ...] = pauses.find_pauses(regions, transcript, speech)
        m_rate, _, _ = rate.word_rate(transcript, regions, speech)
        m_pace, _ = rate.syllable_rate(samples, audio.SAMPLE_RATE, regions, "en", speech)
        wpm = m_rate["rate.wpm"].value if "rate.wpm" in m_rate else None
        score = e2.score_clip(truth, events, wpm, part, condition, ev)
        scores.append(score)
        snr_measured = None
        if condition != "clean":
            _, _, profile = loudness.summarise(
                samples, audio.SAMPLE_RATE, audio.load_native(path), regions,
                capture=CaptureProcessing(), baseline=None, language="en", cfg=speech,
            )  # fmt: skip
            snr_measured = profile.snr_db
        extras.append({
            "words_transcribed": len(transcript.words),
            "words_outside_speech": len(outside),  # removed by ADR 0015, not counted
            "syllables_per_s": m_pace["pace.syllables_per_s"].value
            if "pace.syllables_per_s" in m_pace else None,
            "snr_measured_db": snr_measured,
            "articulation_wpm_true": round(truth.articulation_wpm, 2),
        })  # fmt: skip
        if k % 10 == 0 or k == len(truths):
            progress(f"{k}/{len(truths)} clips, {time.perf_counter() - t0:.0f} s")
    pace = [
        (x["syllables_per_s"], s.wpm_measured, s.wpm_true)
        for s, x in zip(scores, extras, strict=True)
        if x["syllables_per_s"] is not None and s.wpm_measured is not None
    ]
    meta = {
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "syllable_rate_vs_wpm_measured_r": pearson([p[0] for p in pace], [p[1] for p in pace]),
        "syllable_rate_vs_wpm_true_r": pearson([p[0] for p in pace], [p[2] for p in pace]),
        # descriptive (owner, 2026-11-17): against the true articulation rate, pauses removed
        "syllable_rate_vs_articulation_true_r": pearson(
            [x["syllables_per_s"] for x in extras if x["syllables_per_s"] is not None],
            [x["articulation_wpm_true"] for x in extras if x["syllables_per_s"] is not None],
        ),
    }
    snrs = [x["snr_measured_db"] for x in extras if x["snr_measured_db"] is not None]
    if snrs:
        meta["snr_measured_db_median"] = float(np.median(snrs))
    return scores, extras, meta


def _git() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603 (fixed git command)
            ["git", *args],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "-s"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def write_results(
    paths: Paths,
    part: Part,
    condition: str,
    scores: list[E2ClipScore],
    extras: list[dict[str, Any]],
    summary: Any,
    meta: dict[str, Any],
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base = paths.results_e2 / f"{paths.run}_{part}_{condition}_{stamp}"
    base.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s, x in zip(scores, extras, strict=True):
        row: dict[str, Any] = {
            "clip_id": s.clip_id, "voice": s.voice, "rate_band": s.rate_band,
            "has_fillers": s.has_fillers, "wpm_true": round(s.wpm_true, 2),
            "wpm_measured": None if s.wpm_measured is None else round(s.wpm_measured, 2),
            "wpm_error_pct": None if s.wpm_error_pct is None else round(s.wpm_error_pct, 3),
            "true_pauses": s.true_pauses, "detected_pauses": s.detected_pauses,
            "kokoro_pauses": s.kokoro_pauses,
            "duration_mae_s": round(float(np.mean(s.duration_abs_errors_s)), 4)
            if s.duration_abs_errors_s else None,
            "positions_wrong": sum(1 for t, d in s.positions if t != d),
            "positions_matched": len(s.positions),
        }  # fmt: skip
        for t in s.thresholds:
            key = f"{t.threshold_s:g}"
            row |= {f"tp_{key}": t.tp, f"fp_{key}": t.fp, f"fn_{key}": t.fn,
                    f"excluded_{key}": t.excluded, f"tp_all_{key}": t.tp_all,
                    f"fp_all_{key}": t.fp_all, f"fn_all_{key}": t.fn_all}  # fmt: skip
        rows.append(row | x)
    with base.with_name(base.name + "_clips.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = base.with_name(base.name + "_summary.json")
    out.write_text(
        json.dumps({"summary": summary.model_dump(mode="json"), **meta}, indent=1),
        encoding="utf-8",
    )
    return out


def record(paths: Paths, entry: dict[str, Any]) -> None:
    ledger = _ledger(paths)
    ledger.append(entry)
    paths.ledger.parent.mkdir(parents=True, exist_ok=True)
    paths.ledger.write_text(json.dumps(ledger, indent=1), encoding="utf-8")


# --- real model wiring ---------------------------------------------------------------------


def cache_file(paths: Paths, model_name: str, options: AsrSettings, clip: Path) -> Path:
    """Where the transcript of `clip` is cached for this model and these ASR options."""
    key = hashlib.sha256(
        json.dumps(
            {"model": model_name, "asr": options.model_dump(mode="json")}, sort_keys=True
        ).encode()
    ).hexdigest()[:12]
    digest = hashlib.sha256(clip.read_bytes()).hexdigest()[:16]
    return paths.cache(f"{model_name}_{key}") / f"{clip.parent.name}_{clip.stem}_{digest}.json"


def cached_transcriber(paths: Paths, model_name: str, options: AsrSettings) -> Transcriber:
    from presenceai.speech import asr

    model: list[Any] = []

    def run(path: Path) -> Transcript:
        f = cache_file(paths, model_name, options, path)
        if f.exists():
            return Transcript.model_validate_json(f.read_text(encoding="utf-8"))
        if not model:
            model.append(asr.load_model(model_name, options))
        t = asr.transcribe_file(model[0], path, language="en", options=options)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(t.model_dump_json(), encoding="utf-8")
        return t

    return run


def versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version(), "cpu": platform.processor() or platform.machine()}
    try:
        import ctranslate2

        out["cuda_devices"] = str(ctranslate2.get_cuda_device_count())
    except ImportError:
        pass
    for pkg in ("faster-whisper", "ctranslate2", "onnxruntime", "numpy", "librosa"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["split", "score"])
    ap.add_argument("--run", required=True, help="C1 run id (e.g. f1)")
    ap.add_argument("--part", choices=["dev", "test"], default="dev")
    ap.add_argument("--condition", default="clean", help="clean, snr20 or snr10")
    ap.add_argument("--model", default=None, help="ASR model (default: asr.model)")
    ap.add_argument("--rerun-reason", default=None)
    ap.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="ASR device (default: asr.device in configs/default.yaml); compute type unchanged",
    )  # fmt: skip
    ap.add_argument(
        "--dev-min-speech-ms", type=int, default=None,
        help="development part only: try another VAD min_speech_duration_ms (ADR 0011 re-check)",
    )  # fmt: skip
    args = ap.parse_args(argv)
    paths = Paths(args.run)
    ev = load_evaluation()
    if args.action == "split":
        print(make_split(paths, ev))
        print(f"wrote {paths.split_csv.relative_to(REPO_ROOT)}")
        return 0

    from presenceai.settings import get_settings
    from presenceai.speech import vad as vad_mod

    settings = get_settings()
    speech = settings.speech
    asr_opts = settings.asr
    if args.device is not None:  # e.g. a Colab T4: same int8 model, recorded in the results
        asr_opts = asr_opts.model_copy(update={"device": args.device})
    model_name = args.model or asr_opts.model
    part: Part = args.part
    if part == "dev" and args.condition != "clean":
        raise SystemExit("noise conditions are run on the test part only (ADR 0014 §7)")
    if args.dev_min_speech_ms is not None:
        if part != "dev":
            raise SystemExit("--dev-min-speech-ms is for tuning on the development part only")
        vad_cfg = speech.vad.model_copy(update={"min_speech_duration_ms": args.dev_min_speech_ms})
        speech = speech.model_copy(update={"vad": vad_cfg})
    cfg_hash = config_hash(ev, speech, asr_opts, model_name)
    check_ledger(paths, part, args.condition, cfg_hash, args.rerun_reason)
    scores, extras, meta = evaluate(
        paths, part, args.condition,
        transcribe=cached_transcriber(paths, model_name, asr_opts),
        vad=lambda x: vad_mod.detect_speech(x, speech.vad),
        ev=ev, speech=speech,
    )  # fmt: skip
    summary = e2.summarise(scores, ev)
    meta |= {"run": args.run, "part": part, "condition": args.condition, "model": model_name,
             "config_hash": cfg_hash, "vad": speech.vad.model_dump(mode="json"),
             "transcript": speech.transcript.model_dump(mode="json"),
             "asr": asr_opts.model_dump(mode="json"),
             "rerun_reason": args.rerun_reason, "versions": versions(), **_git()}  # fmt: skip
    out = write_results(paths, part, args.condition, scores, extras, summary, meta)
    record(paths, {"run": args.run, "part": part, "condition": args.condition,
                   "config_hash": cfg_hash, "passed": summary.passed, "commit": meta["commit"],
                   "rerun_reason": args.rerun_reason,
                   "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                   "result": out.name})  # fmt: skip
    print(json.dumps(summary.model_dump(mode="json"), indent=1))
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
