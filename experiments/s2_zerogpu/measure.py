"""Spike S2: call the private Space and apply the pass rule (ADR 0007).

    uv run python experiments/s2_zerogpu/measure.py env   --space <owner>/<space>
    uv run python experiments/s2_zerogpu/measure.py probe --space <owner>/<space> --run-id <id>
    uv run python experiments/s2_zerogpu/measure.py run   --space <owner>/<space> --run-id <id>
    uv run python experiments/s2_zerogpu/measure.py cpu   --space <owner>/<space> --run-id <id>

Experiment-only network use (AGENTS.md rule 2 exception, ADR 0007): gradio_client calls to the
owner's private S2 Space. HF_TOKEN is read through presenceai.settings by variable name and is
never printed or written. The Space id is passed on the command line and is not stored.

`run` makes the cold call and then the warm calls on the 3-minute clip, one at a time, appending
each to experiments/results/s2/<id>_calls.csv. It stops at the first failed call (for example an
exhausted daily ZeroGPU quota) and can be run again after the reset: calls already recorded are
not repeated. When all calls are in, it writes <id>_decision.csv. Transcripts are of public
DisfluencySpeech audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
from common import Call

FIELDS = ["n", "kind", "ok", "gpu_s", "load_s", "transcribe_s", "audio_s", "client_s",
          "model_cached", "error", "text"]  # fmt: skip


def client(space: str) -> Any:
    from gradio_client import Client

    from presenceai.settings import get_settings

    token = get_settings().hf_token
    if token is None:
        raise SystemExit("HF_TOKEN is not set")
    return Client(space, hf_token=token.get_secret_value(), verbose=False)


def call_gpu(c: Any, wav: Path, mode: str) -> tuple[dict[str, Any], float, str]:
    from gradio_client import handle_file

    t0 = time.perf_counter()
    try:
        out = json.loads(c.predict(handle_file(str(wav)), mode, api_name="/gpu"))
        return out, round(time.perf_counter() - t0, 3), out.get("error", "")
    except Exception as exc:  # recorded: a quota or timeout error is a result, not a crash
        return {}, round(time.perf_counter() - t0, 3), f"{type(exc).__name__}: {exc}"[:500]


def read_calls(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plan(cfg: dict[str, Any]) -> list[str]:
    return ["cold"] * int(cfg["runs"]["cold"]) + ["warm"] * int(cfg["runs"]["warm"])


def to_calls(rows: list[dict[str, str]]) -> list[Call]:
    return [Call(r["kind"], float(r["gpu_s"] or "nan"), r["text"], r["ok"] == "True") for r in rows]


def write_decision(run: str, rows: list[dict[str, str]], cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["pass"]
    d = common.decide(to_calls(rows), p["max_median_warm_gpu_s"], p["max_any_call_gpu_s"])
    ref = (common.RESULTS / "clip_reference.txt").read_text(encoding="utf-8")
    d["wer_vs_reference"] = [round(wer(ref, r["text"]), 2) for r in rows if r["ok"] == "True"]
    d = {"run_id": run, **d}
    with (common.RESULTS / f"{run}_decision.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(d))
        w.writeheader()
        w.writerow(d)
    return d


def wer(reference: str, hypothesis: str) -> float:
    """Information only: same normalisation and WER as spike S3."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "s3_common", common.REPO / "experiments" / "s3_unsloth_whisper" / "common.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load experiments/s3_unsloth_whisper/common.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s3_common"] = mod  # dataclasses need the module registered while it loads
    spec.loader.exec_module(mod)
    return mod.corpus_wer([reference], [hypothesis])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["env", "probe", "run", "cpu"])
    ap.add_argument("--space", required=True, help="<owner>/<space> (not stored)")
    ap.add_argument("--run-id")
    args = ap.parse_args(argv)
    cfg = common.load_config()
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    c = client(args.space)
    if args.action == "env":
        print(json.dumps(json.loads(c.predict(api_name="/env")), indent=2))
        return 0
    if not args.run_id:
        ap.error("--run-id is required")
    run = args.run_id
    if args.action == "probe":
        out, client_s, err = call_gpu(c, common.DATA_DIR / "clip_probe.wav", "probe")
        env = json.loads(c.predict(api_name="/env"))
        rec = {"run_id": run, "client_s": client_s, "error": err, **out, "env": env}
        (common.RESULTS / f"{run}_probe.json").write_text(json.dumps(rec, indent=2), "utf-8")
        print(json.dumps(rec, indent=2))
        return 0 if out.get("ct2_cuda_devices", 0) >= 1 and not err else 1
    if args.action == "cpu":
        t0 = time.perf_counter()
        from gradio_client import handle_file

        out = json.loads(
            c.predict(handle_file(str(common.DATA_DIR / "clip_full.wav")), api_name="/cpu")
        )
        out["client_s"] = round(time.perf_counter() - t0, 3)
        ref = (common.RESULTS / "clip_reference.txt").read_text(encoding="utf-8")
        out["wer_vs_reference"] = round(wer(ref, out["text"]), 2)
        (common.RESULTS / f"{run}_cpu.json").write_text(json.dumps(out, indent=2), "utf-8")
        print(json.dumps({k: v for k, v in out.items() if k != "text"}, indent=2))
        return 0

    path = common.RESULTS / f"{run}_calls.csv"
    rows = [r for r in read_calls(path) if r["ok"] == "True"]  # failed calls are retried
    failed = [r for r in read_calls(path) if r["ok"] != "True"]
    todo = plan(cfg)[len(rows) :]
    for kind in todo:
        out, client_s, err = call_gpu(c, common.DATA_DIR / "clip_full.wav", "full")
        row = {
            "n": len(rows) + len(failed) + 1,
            "kind": kind,
            "ok": str(not err),
            "gpu_s": out.get("gpu_s", ""),
            "load_s": out.get("load_s", ""),
            "transcribe_s": out.get("transcribe_s", ""),
            "audio_s": out.get("audio_s", ""),
            "client_s": client_s,
            "model_cached": out.get("model_cached", ""),
            "error": err,
            "text": out.get("text", ""),
        }
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(json.dumps({k: v for k, v in row.items() if k != "text"}))
        if err:
            print("stopped at a failed call; run again after the quota reset", file=sys.stderr)
            return 1
        rows.append({k: str(v) for k, v in row.items()})
    print(json.dumps(write_decision(run, rows, cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
