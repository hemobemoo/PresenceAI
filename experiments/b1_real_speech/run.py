"""B1 real-speech checks on the owner's own recordings (ADR 0016). Laptop only.

    uv run python experiments/b1_real_speech/run.py status
    uv run python experiments/b1_real_speech/run.py prefill
    uv run python experiments/b1_real_speech/run.py label <clip> | --all
    uv run python experiments/b1_real_speech/run.py listen <clip> <pause-number>
    uv run python experiments/b1_real_speech/run.py score

- Audio: data/b1/audio/b1_<en|ar>_<01-10>_<type>.wav (docs/data/B1_RECORDING_GUIDE.md).
  data/ is git-ignored; nothing here uploads anything.
- `prefill` runs the app's own analysis (`analyze_file`, model from configs/default.yaml) and
  writes data/b1/annotations/<clip>.yaml (to correct) and <clip>.labels.txt (detected pauses,
  all `?`). It never overwrites an annotation; `--force` redoes only those still `prefilled`.
- `label` fills <clip>.labels.txt from the punctuation of the owner's corrected transcript,
  aligned to Whisper's timed words in the cached report. It never changes pause times, never
  adds or deletes pauses, and keeps a file it labelled fully unless `--force`. Each `?` is
  printed with its number, time and the words around it.
- `listen` plays 3 s before to 1 s after one pause (winsound; a temporary WAV slice that is
  deleted afterwards), to decide a `?` by ear.
- `score` re-uses the report cached for the current config (data/b1/reports/), or analyses the
  clip again, and compares it with every annotation marked `status: corrected`. It writes
  numbers only to experiments/results/b1/: per clip (CSV) and per language (JSON). B1 is
  descriptive: no pass/fail, and no threshold is tuned on it.
- Checks it reports: Arabic E2 (FINAL_10 §2), English E2 on real speech, E1 WER on the owner's
  clips, ADR 0015 (real words removed by the end margin), ADR 0014 (pause positions from the
  owner's punctuation; pause F1 and duration only for clips with hand-checked edges), ADR 0011
  (punctuation per sentence mark), ADR 0013 (clipping and the plateau ratio, one microphone) and
  the ADR 0016 language check (detected language and probability per clip).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from presenceai.evalconfig import REPO_ROOT, load_evaluation
from presenceai.evaluation import b1
from presenceai.schemas.b1 import B1ClipScore, LabelledPause, Prefill
from presenceai.schemas.speech import SpeechReport
from presenceai.settings import AsrSettings, Settings, SpeechSettings
from presenceai.speech import analyze, asr, audio, loudness


@dataclass(frozen=True)
class Paths:
    data: Path = REPO_ROOT / "data" / "b1"
    results: Path = REPO_ROOT / "experiments" / "results" / "b1"

    @property
    def audio(self) -> Path:
        return self.data / "audio"

    @property
    def annotations(self) -> Path:
        return self.data / "annotations"

    @property
    def reports(self) -> Path:
        return self.data / "reports"


def config_hash(speech: SpeechSettings, options: AsrSettings, model: str) -> str:
    blob = json.dumps(
        {"speech": speech.model_dump(mode="json"), "asr": options.model_dump(mode="json"),
         "model": model},
        sort_keys=True,
    )  # fmt: skip
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None


def clips(paths: Paths) -> dict[str, Path]:
    """Audio files by stem; a misnamed file stops the run with the expected name."""
    found: dict[str, Path] = {}
    if not paths.audio.is_dir():
        return found
    for f in sorted(paths.audio.iterdir()):
        if f.suffix.lower() in b1.AUDIO_SUFFIXES:
            b1.parse_stem(f.stem)
            found[f.stem] = f
    return found


class Analyser:
    """The app's analysis with one loaded model, cached per clip and config."""

    def __init__(self, paths: Paths, settings: Settings) -> None:
        self.paths, self.settings = paths, settings
        self.model_name = settings.asr.model
        self.hash = config_hash(settings.speech, settings.asr, self.model_name)
        self._model: Any = None

    def report(self, stem: str, clip: Path) -> SpeechReport:
        cache = self.paths.reports / f"{stem}.{self.hash}.json"
        if cache.is_file():
            return SpeechReport.model_validate_json(cache.read_text(encoding="utf-8"))
        if self._model is None:
            self._model = asr.load_model(self.model_name, self.settings.asr)
        language, _ = b1.parse_stem(stem)
        report = analyze.analyze_file(
            clip,
            language=language,
            model=self._model,
            model_name=self.model_name,
            options=self.settings.asr,
            speech=self.settings.speech,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(report.model_dump_json(), encoding="utf-8")
        return report


def cmd_status(paths: Paths) -> int:
    have = clips(paths)
    for stem in b1.expected_stems():
        state = "no audio"
        if stem in have:
            yml = paths.annotations / f"{stem}.yaml"
            state = "recorded"
            if yml.is_file():
                state = b1.load_annotation(yml.read_text(encoding="utf-8")).status
        print(f"{stem:26} {state}")
    return 0


def cmd_prefill(paths: Paths, settings: Settings, force: bool) -> int:
    an = Analyser(paths, settings)
    meta = Prefill(model=an.model_name, config_hash=an.hash, commit=_commit())
    paths.annotations.mkdir(parents=True, exist_ok=True)
    for stem, clip in clips(paths).items():
        yml = paths.annotations / f"{stem}.yaml"
        labels = paths.annotations / f"{stem}.labels.txt"
        if yml.is_file():
            current = b1.load_annotation(yml.read_text(encoding="utf-8"))
            if current.status == "corrected" or not force:
                print(f"{stem}: kept ({current.status})")
                continue
        annotation, label_text = b1.prefill(an.report(stem, clip), stem, meta)
        yml.write_text(b1.dump_annotation(annotation), encoding="utf-8")
        labels.write_text(label_text, encoding="utf-8")
        print(f"{stem}: pre-filled")
    return 0


def _pileup(clip: Path, settings: Settings) -> float | None:
    native, _ = audio.load_native(clip)
    ratios = [loudness.pileup_ratio(ch, settings.speech.loudness) for ch in native]
    values = [r for r in ratios if r is not None]
    return max(values) if values else None


def cmd_score(paths: Paths, settings: Settings) -> int:
    ev = load_evaluation()
    an = Analyser(paths, settings)
    scores: list[B1ClipScore] = []
    waiting: list[str] = []
    not_ready: list[str] = []
    for stem, clip in clips(paths).items():
        yml = paths.annotations / f"{stem}.yaml"
        if not yml.is_file():
            waiting.append(stem)
            continue
        annotation = b1.load_annotation(yml.read_text(encoding="utf-8"))
        if annotation.status != "corrected":
            waiting.append(stem)
            continue
        labelled = b1.read_labels(
            (paths.annotations / f"{stem}.labels.txt").read_text(encoding="utf-8")
        )
        if annotation.labels_method != "transcript_punctuation":
            not_ready.append(f"{stem} (run label)")
            continue
        if any(p.label == b1.UNDECIDED for p in labelled):
            not_ready.append(f"{stem} (a pause is still '?')")
            continue
        scores.append(
            b1.score_clip(
                annotation, labelled, an.report(stem, clip), ev, settings.speech.pauses,
                pileup_ratio_max=_pileup(clip, settings),
            )
        )  # fmt: skip
    if waiting:
        print(f"not corrected yet: {', '.join(waiting)}")
    if not_ready:
        print(f"labels not finished: {', '.join(not_ready)}")
    if not scores:
        print("no corrected clips to score")
        return 1
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    paths.results.mkdir(parents=True, exist_ok=True)
    rows = [s.model_dump(mode="json") for s in scores]
    with (paths.results / f"b1_{stamp}_clips.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, dict | list) else v
                          for k, v in r.items()} for r in rows)  # fmt: skip
    summary = {
        "config_hash": an.hash,
        "model": an.model_name,
        "commit": _commit(),
        "descriptive_only": True,
        "by_language": {
            lang: b1.summarise([s for s in scores if s.language == lang])
            for lang in b1.LANGS
            if any(s.language == lang for s in scores)
        },
    }
    out = paths.results / f"b1_{stamp}_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["by_language"], indent=2, ensure_ascii=False))
    unchecked = [s.clip for s in scores if not s.pause_edges_checked]
    if unchecked:
        print(f"pause edges not hand-checked (no pause F1 or duration): {', '.join(unchecked)}")
    print(f"wrote {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")
    return 0


# --- label and listen (owner's transcript; no extra tools) ---------------------------------


def prefill_report(paths: Paths, stem: str, a: Any) -> SpeechReport:
    """The report the pauses were pre-filled from (its config hash is in the annotation)."""
    cache = paths.reports / f"{stem}.{a.prefill.config_hash}.json"
    if not cache.is_file():
        raise SystemExit(
            f"{stem}: the report this annotation was pre-filled from is missing ({cache.name}); "
            "run `prefill --force` (keeps corrected annotations) and correct again"
        )
    return SpeechReport.model_validate_json(cache.read_text(encoding="utf-8"))


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def label_clip(paths: Paths, stem: str, force: bool) -> int:
    """Fill <clip>.labels.txt from the owner's corrected transcript; returns the number of "?"."""
    yml = paths.annotations / f"{stem}.yaml"
    labels_file = paths.annotations / f"{stem}.labels.txt"
    if not yml.is_file() or not labels_file.is_file():
        raise SystemExit(f"{stem}: no annotation yet; run prefill first")
    a = b1.load_annotation(yml.read_text(encoding="utf-8"))
    existing = b1.read_labels(labels_file.read_text(encoding="utf-8"))
    done = a.labels_method == "transcript_punctuation"
    if done and not force and all(p.label != b1.UNDECIDED for p in existing):
        print(f"{stem}: kept (every pause already labelled; --force to redo)")
        return 0
    if a.transcript == a.whisper_text:
        print(f"{stem}: warning: transcript is still Whisper's text; correct it first")
    tokens = b1.timed_tokens(a.transcript, prefill_report(paths, stem, a), a.language)
    margin = load_evaluation().b1.labels.overlap_margin_s
    result = b1.label_pauses(existing, tokens, margin_s=margin)
    labels_file.write_text(
        b1.write_labels(
            [LabelledPause(start_s=r.pause.start_s, end_s=r.pause.end_s, label=r.label)
             for r in result]
        ),
        encoding="utf-8",
    )  # fmt: skip
    yml.write_text(
        b1.dump_annotation(a.model_copy(update={"labels_method": "transcript_punctuation"})),
        encoding="utf-8",
    )
    counts = {k: sum(1 for r in result if r.label == k) for k in ("sentence", "clause", "mid")}
    undecided = [r for r in result if r.label == b1.UNDECIDED]
    print(f"{stem}: {len(result)} pauses, {counts}, {len(undecided)} to decide by ear")
    for r in undecided:
        print(
            f"  pause {r.number} at {_fmt(r.pause.start_s)}-{_fmt(r.pause.end_s)}: "
            f"... {' '.join(r.before)} | {' '.join(r.after)} ...  ({r.reason})"
        )
    if undecided:
        print(f"  listen: uv run python experiments/b1_real_speech/run.py listen {stem} <pause>")
    return len(undecided)


def cmd_label(paths: Paths, stem: str | None, force: bool) -> int:
    stems = (
        [stem] if stem else [s for s in clips(paths) if (paths.annotations / f"{s}.yaml").is_file()]
    )
    if stem:
        b1.parse_stem(stem)
    for s in stems:
        label_clip(paths, s, force)
    return 0


LISTEN_BEFORE_S, LISTEN_AFTER_S = 3.0, 1.0


def write_slice(clip: Path, start_s: float, end_s: float, out: Path) -> tuple[float, float]:
    """Copy [start_s, end_s) of a WAV file into `out` (16-bit PCM WAV, standard library only).
    Other formats are decoded with the app's own PyAV loader. Returns the range written."""
    try:
        with wave.open(str(clip), "rb") as src:
            rate, n = src.getframerate(), src.getnframes()
            a, b = max(0, round(start_s * rate)), min(n, round(end_s * rate))
            src.setpos(a)
            frames = src.readframes(b - a)
            params = src.getparams()
        with wave.open(str(out), "wb") as dst:
            dst.setparams(params)
            dst.writeframes(frames)
        return a / rate, b / rate
    except wave.Error:
        native, rate = audio.load_native(clip)  # e.g. FLAC or float WAV
        a, b = max(0, round(start_s * rate)), min(native.shape[1], round(end_s * rate))
        pcm = (np.clip(native[:, a:b].T, -1, 1) * 32767).astype("<i2")
        with wave.open(str(out), "wb") as dst:
            dst.setnchannels(native.shape[0])
            dst.setsampwidth(2)
            dst.setframerate(rate)
            dst.writeframes(pcm.tobytes())
        return a / rate, b / rate


def play_wav(path: Path) -> None:
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    else:
        raise SystemExit("listen plays audio with winsound, which exists only on Windows")


def cmd_listen(
    paths: Paths, stem: str, number: int, player: Callable[[Path], None] = play_wav
) -> int:
    b1.parse_stem(stem)
    labels = b1.read_labels((paths.annotations / f"{stem}.labels.txt").read_text(encoding="utf-8"))
    if not 1 <= number <= len(labels):
        raise SystemExit(f"{stem}: pause numbers run from 1 to {len(labels)}")
    p = labels[number - 1]
    clip = clips(paths)[stem]
    fd, name = tempfile.mkstemp(suffix=".wav", prefix="b1_listen_")
    os.close(fd)
    tmp = Path(name)
    try:
        a, b = write_slice(clip, p.start_s - LISTEN_BEFORE_S, p.end_s + LISTEN_AFTER_S, tmp)
        print(
            f"{stem} pause {number} ({p.label}): {_fmt(p.start_s)}-{_fmt(p.end_s)}; "
            f"playing {_fmt(a)}-{_fmt(b)}"
        )
        player(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="list the 20 planned clips and their state")
    pre = sub.add_parser("prefill", help="write annotation files to correct")
    pre.add_argument("--force", action="store_true", help="redo annotations still 'prefilled'")
    lab = sub.add_parser("label", help="fill pause labels from your corrected transcript")
    which = lab.add_mutually_exclusive_group(required=True)
    which.add_argument("clip", nargs="?", help="e.g. b1_en_01_normal")
    which.add_argument("--all", action="store_true", help="every pre-filled clip")
    lab.add_argument("--force", action="store_true", help="redo a fully labelled file")
    lis = sub.add_parser("listen", help="play 3 s before to 1 s after one pause")
    lis.add_argument("clip")
    lis.add_argument("pause", type=int, help="pause number, as `label` prints it")
    sub.add_parser("score", help="compare corrected annotations with the app's reports")
    args = parser.parse_args(argv)
    if isinstance(sys.stdout, io.TextIOWrapper):  # Arabic words, also when redirected
        sys.stdout.reconfigure(encoding="utf-8")
    paths = Paths()
    if args.command == "status":
        return cmd_status(paths)
    if args.command == "label":
        return cmd_label(paths, None if args.all else args.clip, args.force)
    if args.command == "listen":
        return cmd_listen(paths, args.clip, args.pause)
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    if args.command == "prefill":
        return cmd_prefill(paths, settings, args.force)
    return cmd_score(paths, settings)


if __name__ == "__main__":
    sys.exit(main())
