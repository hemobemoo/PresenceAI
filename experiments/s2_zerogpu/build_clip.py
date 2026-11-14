"""Spike S2: build the 3-minute test clip and the 10-second probe clip (ADR 0007).

    uv run --with pyarrow python experiments/s2_zerogpu/build_clip.py

Downloads one pinned public file (DisfluencySpeech test split, Apache-2.0), shuffles its rows
with the configured seed and joins whole clips until at least `clip.min_seconds`. Writes
data/s2/clip_full.wav and data/s2/clip_probe.wav (16 kHz mono, git-ignored), and
experiments/results/s2/clip_manifest.csv (rows used, durations), clip_reference.txt (the joined
public transcript) and clip_info.json (SHA-256 of both WAV files). No personal data is involved.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

from presenceai.speech.audio import SAMPLE_RATE, load_audio


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((pcm * 32767.0).round().astype("<i2").tobytes())


def main() -> int:
    cfg = common.load_config()["clip"]
    parquet = hf_hub_download(
        cfg["dataset"], cfg["file"], repo_type="dataset", revision=cfg["revision"]
    )
    df = pd.read_parquet(parquet)
    order = list(range(len(df)))
    random.Random(cfg["seed"]).shuffle(order)  # noqa: S311 (reproducible order, not security)

    pieces: list[np.ndarray] = []
    rows = []
    total = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for i in order:
            raw = df.iloc[i]["audio"]["bytes"]
            src = Path(tmp) / f"row_{i}.audio"
            src.write_bytes(raw)
            samples = load_audio(src)
            seconds = len(samples) / SAMPLE_RATE
            pieces.append(samples)
            rows.append(
                {"row": i, "seconds": round(seconds, 3), "text": df.iloc[i][cfg["text_column"]]}
            )
            total += seconds
            if total >= cfg["min_seconds"]:
                break
    full = np.concatenate(pieces)
    probe = full[: int(cfg["probe_seconds"] * SAMPLE_RATE)]

    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    full_wav, probe_wav = common.DATA_DIR / "clip_full.wav", common.DATA_DIR / "clip_probe.wav"
    write_wav(full_wav, full)
    write_wav(probe_wav, probe)
    with (common.RESULTS / "clip_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row", "seconds", "text"])
        w.writeheader()
        w.writerows(rows)
    reference = " ".join(str(r["text"]).strip() for r in rows)
    (common.RESULTS / "clip_reference.txt").write_text(reference + "\n", encoding="utf-8")
    info = {
        "dataset": cfg["dataset"],
        "revision": cfg["revision"],
        "file": cfg["file"],
        "seed": cfg["seed"],
        "clips": len(rows),
        "full_seconds": round(len(full) / SAMPLE_RATE, 3),
        "probe_seconds": round(len(probe) / SAMPLE_RATE, 3),
        "sha256": {
            "clip_full.wav": common.sha256(full_wav),
            "clip_probe.wav": common.sha256(probe_wav),
        },
    }
    (common.RESULTS / "clip_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
