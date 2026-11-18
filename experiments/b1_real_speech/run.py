"""B1 real-speech checks on the owner's own recordings (ADR 0016). Laptop only.

    uv run python experiments/b1_real_speech/run.py status
    uv run python experiments/b1_real_speech/run.py prefill
    uv run python experiments/b1_real_speech/run.py score

- Audio: data/b1/audio/b1_<en|ar>_<01-10>_<type>.wav (docs/data/B1_RECORDING_GUIDE.md).
  data/ is git-ignored; nothing here uploads anything.
- `prefill` runs the app's own analysis (`analyze_file`, model from configs/default.yaml) and
  writes data/b1/annotations/<clip>.yaml and <clip>.labels.txt for the owner to correct. It never
  overwrites an annotation; `--force` redoes only those still `status: prefilled`.
- `score` re-uses the report cached for the current config (data/b1/reports/), or analyses the
  clip again, and compares it with every annotation marked `status: corrected`. It writes
  numbers only to experiments/results/b1/: per clip (CSV) and per language (JSON). B1 is
  descriptive: no pass/fail, and no threshold is tuned on it.
- Checks it reports: Arabic E2 (FINAL_10 §2), English E2 on real speech, E1 WER on the owner's
  clips, ADR 0015 (real words removed by the end margin), ADR 0014 (mid-sentence labels), ADR 0011
  (punctuation per sentence mark), ADR 0013 (clipping and the plateau ratio, one microphone) and
  the ADR 0016 language check (detected language and probability per clip).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from presenceai.evalconfig import REPO_ROOT, load_evaluation
from presenceai.evaluation import b1
from presenceai.schemas.b1 import B1ClipScore, Prefill
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
        scores.append(
            b1.score_clip(
                annotation, labelled, an.report(stem, clip), ev, settings.speech.pauses,
                pileup_ratio_max=_pileup(clip, settings),
            )
        )  # fmt: skip
    if waiting:
        print(f"not corrected yet: {', '.join(waiting)}")
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
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="list the 20 planned clips and their state")
    pre = sub.add_parser("prefill", help="write annotation files to correct")
    pre.add_argument("--force", action="store_true", help="redo annotations still 'prefilled'")
    sub.add_parser("score", help="compare corrected annotations with the app's reports")
    args = parser.parse_args(argv)
    paths = Paths()
    if args.command == "status":
        return cmd_status(paths)
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    if args.command == "prefill":
        return cmd_prefill(paths, settings, args.force)
    return cmd_score(paths, settings)


if __name__ == "__main__":
    sys.exit(main())
