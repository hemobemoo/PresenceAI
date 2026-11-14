"""Spike S3 local check on the laptop CPU: load both CTranslate2 models in faster-whisper,
transcribe the 10 test clips, and apply the pass rule (ADR 0005).

    uv run python experiments/s3_unsloth_whisper/check_local.py --run-id <id>

Reads models/s3/<run>/{base,ft} and data/s3/<run>/clips (git-ignored, written by push.py fetch)
and experiments/results/s3/<run>_train_log.csv. Writes experiments/results/s3/<run>_local.csv
(per clip: reference, both transcripts, filler counts, CPU seconds), <run>_local_info.json and
<run>_decision.csv. Transcripts are of public DisfluencySpeech clips.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common


def transcribe(model: Any, wav: Path, opts: dict[str, Any]) -> tuple[str, float]:
    t0 = time.perf_counter()
    segments, _ = model.transcribe(
        str(wav),
        language=opts["language"],
        beam_size=opts["beam_size"],
        temperature=opts["temperature"],
        condition_on_previous_text=opts["condition_on_previous_text"],
        vad_filter=opts["vad_filter"],
    )
    text = " ".join(s.text.strip() for s in segments).strip()  # generator: decoding runs here
    return text, round(time.perf_counter() - t0, 3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", type=Path, default=common.CONFIG)
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    opts = cfg["local"]
    run = args.run_id
    clips_dir = common.DATA_DIR / run / "clips"
    with (clips_dir / "refs.csv").open(newline="", encoding="utf-8") as f:
        refs = list(csv.DictReader(f))
    with (common.RESULTS / f"{run}_train_log.csv").open(newline="", encoding="utf-8") as f:
        loss = common.loss_check(list(csv.DictReader(f)), int(cfg["pass"]["loss_window"]))

    from faster_whisper import WhisperModel

    hyps: dict[str, list[str]] = {"base": [], "ft": []}
    secs: dict[str, list[float]] = {"base": [], "ft": []}
    loads = True
    load_error = ""
    for which in ("base", "ft"):
        try:
            model = WhisperModel(
                str(common.MODELS_DIR / run / which),
                device="cpu",
                compute_type=opts["compute_type"],
            )
        except Exception as exc:  # recorded: the pass rule needs the fine-tuned model to load
            if which == "ft":
                loads, load_error = False, repr(exc)
                break
            raise
        for r in refs:
            text, s = transcribe(model, clips_dir / r["clip"], opts)
            hyps[which].append(text)
            secs[which].append(s)
        del model

    rows = []
    fillers = cfg["fillers"]
    for i, r in enumerate(refs):
        ft = hyps["ft"][i] if loads else ""
        rows.append(
            {
                "clip": r["clip"],
                "duration_s": r["duration_s"],
                "reference": r["reference"],
                "base": hyps["base"][i],
                "ft": ft,
                "fillers_ref": common.count_fillers(r["reference"], fillers),
                "fillers_base": common.count_fillers(hyps["base"][i], fillers),
                "fillers_ft": common.count_fillers(ft, fillers),
                "cpu_s_base": secs["base"][i],
                "cpu_s_ft": secs["ft"][i] if loads else "",
            }
        )
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    with (common.RESULTS / f"{run}_local.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    references = [r["reference"] for r in refs]
    base_wer = common.corpus_wer(references, hyps["base"])
    ft_wer = common.corpus_wer(references, hyps["ft"]) if loads else float("inf")
    decision = common.decide(
        loss, loads, hyps["ft"], base_wer, ft_wer, float(cfg["pass"]["max_wer_increase_points"])
    )
    decision = {"run_id": run, **decision, "load_error": load_error}
    with (common.RESULTS / f"{run}_decision.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(decision))
        w.writeheader()
        w.writerow(decision)

    import ctranslate2
    import faster_whisper

    info = {
        "run_id": run,
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "faster_whisper": faster_whisper.__version__,
        "ctranslate2": ctranslate2.__version__,
        "options": opts,
        "fillers_total": {
            k: sum(int(r[f"fillers_{k}"]) for r in rows) for k in ("ref", "base", "ft")
        },
        "cpu_s_per_clip_mean": {k: round(sum(v) / len(v), 3) for k, v in secs.items() if v},
        "audio_s_total": round(sum(float(r["duration_s"]) for r in refs), 2),
    }
    (common.RESULTS / f"{run}_local_info.json").write_text(json.dumps(info, indent=2), "utf-8")
    print(json.dumps(decision, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
