"""T1-r1 (PR-4): build the Colab notebook, ingest its zip, compare with base (ADR 0029).

    uv run python training/t1_whisper_verbatim/colab_t1.py build   --run-id <id>
    uv run python training/t1_whisper_verbatim/colab_t1.py ingest  --run-id <id> [--zip <path>]
    uv run python training/t1_whisper_verbatim/colab_t1.py compare --run-id <id>

build writes data/t1/colab/t1_run.py (config, build info and modules injected: the built E3
script itself, so T1 is scored by the same code as base, plus t1train, t1merge, chunks,
t1pick) and
data/t1/colab/T1R1_<id>.ipynb. Needs a **T4 GPU** and the Colab Secret `HF_TOKEN` (write token
for the private T1 repo only). Results go to MyDrive/presenceai_p4/t1r1_<id>/; checkpoints and
converted candidates to its work/ folder, never into the zip.

ingest: hypotheses and item lists (corpus text, research-only annotation times) go to the
git-ignored data/t1/<id>/; everything else to experiments/results/t1/r1_<id>/.

compare runs no model: it applies ADR 0026 §4 to the T1 metrics and run b5's (ADR 0028 §11),
recomputes base's per-unit counts from b5's local hypotheses (they must give b5's totals) and
reports paired bootstrap intervals of the differences.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import colab_e3  # noqa: E402
import colab_ingest as ci  # noqa: E402
import e3score as sc  # noqa: E402
import t1pick  # noqa: E402

REPO = ci.REPO
TEMPLATE = HERE / "colab" / "t1_run.py"
BUILD_DIR = REPO / "data" / "t1" / "colab"
RESULTS = REPO / "experiments" / "results" / "t1"
DATA = REPO / "data" / "t1"
TRAINING = REPO / "configs" / "training.yaml"
DEFAULT = REPO / "configs" / "default.yaml"
FILLERS = REPO / "src" / "presenceai" / "lang" / "en" / "fillers.yaml"
B5 = "20261128-b5"  # the base run (ADR 0028 §11)
B5_RESULTS = REPO / "experiments" / "results" / "e3" / f"base_{B5}"
B5_DATA = REPO / "data" / "e3" / B5
MODULES = {
    "t1train.py": HERE / "colab" / "t1train.py",
    "chunks.py": HERE / "chunks.py",
    "t1pick.py": HERE / "t1pick.py",
    "t1merge.py": HERE / "t1merge.py",
}
TIMEOUT_S = 11 * 3600
SETUP = (
    "# Unsloth for the training process (configs/training.yaml t1.training_env), first.\n"
    "!pip install -q __TRAINING_PACKAGES__\n" + colab_e3.SETUP
)


def load_config() -> dict[str, Any]:
    e3cfg = colab_e3.load_config()
    tr = yaml.safe_load(TRAINING.read_text(encoding="utf-8"))
    vad = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))["speech"]["vad"]
    return {"e3_module": e3cfg, "run": {"training": tr, "e3": e3cfg["e3"], "app_vad": vad}}


def script(cfg: dict[str, Any], build_info: dict[str, Any]) -> str:
    e3_body = colab_e3.script(cfg["e3_module"], build_info)
    mods = {"e3_run.py": base64.b64encode(e3_body.encode("utf-8")).decode("ascii")}
    mods |= {n: base64.b64encode(p.read_bytes()).decode("ascii") for n, p in MODULES.items()}
    text = TEMPLATE.read_text(encoding="utf-8")
    for key in ("__CONFIG_JSON__", "__BUILD_JSON__", "__MODULES_JSON__"):
        if key not in text:
            raise ValueError(f"template has no {key} placeholder")
    return (
        text.replace("__CONFIG_JSON__", json.dumps(cfg["run"], ensure_ascii=True))
        .replace("__BUILD_JSON__", json.dumps(build_info, ensure_ascii=True))
        .replace("__MODULES_JSON__", json.dumps(mods))
    )


def build(run_id: str, out: Path = BUILD_DIR, platform: str = "colab") -> Path:
    """The same script for both platforms (same SHA-256); only the notebook around it differs."""
    cfg = load_config()
    body = script(cfg, {"run_id": run_id, **ci.repo_commit()})
    tr = cfg["run"]["training"]["t1"]
    repo = tr["publish"]["repo"]
    setup = SETUP.replace("__TRAINING_PACKAGES__", " ".join(tr["training_env"]["packages"]))
    if platform == "kaggle":
        return ci.write_build(
            out, "t1_run.py", body, f"T1R1K_{run_id}.ipynb", kaggle(run_id, repo, body, setup)
        )
    if platform != "colab":
        raise ValueError(f"unknown platform {platform!r}")
    nb = ci.notebook(
        title=f"PresenceAI T1-r1 run `{run_id}` (ADR 0029)",
        steps=(
            "1. Runtime > Change runtime type > **T4 GPU**.\n"
            f"2. Secrets (key icon): `HF_TOKEN` = your fine-grained **write** token for `{repo}` "
            "only, with *Notebook access* on.\n"
            "3. Runtime > Run all; allow Google Drive when asked.\n"
            f"4. Results go to `MyDrive/presenceai_p4/t1r1_{run_id}/t1r1/` as each stage "
            "finishes; checkpoints and converted models to `.../work/` (not in the zip). **If "
            "Colab disconnects, reconnect and Run all again**: finished stages, epochs and items "
            "are skipped.\n"
            f"5. Save `t1r1_{run_id}.zip` into `data/colab_results/` when the browser asks, or "
            "take it from the Drive folder.\n"
            "6. Only after the zip is on the laptop: set `DELETE_RUNTIME = True` in the last cell "
            "and run it by hand.\n\n"
            "Order: builder check, model check, training (3 epochs), pick on DisfluencySpeech "
            "val + AMI dev, upload to your private repo, CPU load check, then the E3 test sets "
            "and the per-window PodcastFillers conditions (report-only)."
        ),
        script_name="t1_run.py",
        body=body,
        out_dir="t1r1",
        zip_stem=f"t1r1_{run_id}",
        timeout_s=TIMEOUT_S,
        gpu="T4",
        drive_folder=f"presenceai_p4/t1r1_{run_id}",
        out_env="P4_OUT",
        secrets=("HF_TOKEN",),
        setup=setup,
        cache_folder="presenceai_data",
    )
    return ci.write_build(out, "t1_run.py", body, f"T1R1_{run_id}.ipynb", nb)


def kaggle(run_id: str, repo: str, body: str, setup: str) -> dict[str, Any]:
    """Kaggle Notebooks, the fallback (ADR 0026 §1), used for T1-r1 (owner 2026-11-28, ADR 0029
    §11): a committed run, everything kept in /kaggle/working (the notebook's output)."""
    return ci.kaggle_notebook(
        title=f"PresenceAI T1-r1 run `{run_id}` on Kaggle (ADR 0029 §11)",
        steps=(
            "1. Kaggle > Create > Notebook > File > **Import Notebook**: this file.\n"
            "2. Settings (right panel): Accelerator **GPU T4 x2** (one is used); **Internet on**.\n"
            f"3. Add-ons > Secrets: `HF_TOKEN` = your fine-grained **write** token for `{repo}` "
            "only; tick it for this notebook.\n"
            "4. **Save Version > Save & Run All (Commit)**. It runs in the background (Kaggle's "
            "limit is 12 h; the script stops itself at 11 h); you can close the browser.\n"
            f"5. When it is done: open the version > **Output** > download `t1r1_{run_id}.zip` "
            "into `data/colab_results/`.\n"
            "6. **If it stopped before the end:** Add input > Your Work > this notebook (its "
            "output), then Save & Run All again. The dataset cache and the finished stages, "
            "epochs and items are reused.\n\n"
            "The output keeps the results, the checkpoints (`work/`) and a private cache of the "
            "pinned public files (`presenceai_data/`); delete old versions' output when T1 is "
            "done (data/README.md, delete list)."
        ),
        script_name="t1_run.py",
        body=body,
        out_dir="t1r1",
        zip_stem=f"t1r1_{run_id}",
        timeout_s=TIMEOUT_S,
        secrets=("HF_TOKEN",),
        setup=setup,
    )


def ingest(run_id: str, zip_path: Path | None = None) -> dict[str, Any]:
    res, data = RESULTS / f"r1_{run_id}", DATA / run_id
    return ci.ingest(
        zip_path or ci.INBOX / f"t1r1_{run_id}.zip",
        "t1r1",
        [("hyps/*", data), ("items/*", data), ("*", res)],
        scratch=DATA / f"_ingest_{run_id}",
        info_dest=res,
    )


# ---------------------------------------------------------------- compare (laptop, no model)


def _jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def base_units(hyps_dir: Path, items_dir: Path, cond: str) -> dict[str, dict[str, sc.Counts]]:
    """AMI test (deciding, by meeting) and PodcastFillers (by episode) counts from b5's
    hypotheses, scored exactly as e3_run.score does."""
    ev = yaml.safe_load((REPO / "configs" / "evaluation.yaml").read_text(encoding="utf-8"))
    e3, back = ev["e3"], ev["t1"]["ami"]["backchannels"]
    hes = yaml.safe_load(FILLERS.read_text(encoding="utf-8"))["hesitations"]
    ami = {s["id"]: s for s in json.loads((items_dir / "ami_test.json").read_text("utf-8"))}
    pf_ev = json.loads((items_dir / "podcastfillers_test_events.json").read_text("utf-8"))
    pf = e3["podcastfillers"]
    out: dict[str, dict[str, sc.Counts]] = {"ami_test": {}, "podcastfillers_test": {}}
    for h in _jsonl(hyps_dir / f"{cond}.jsonl"):
        if h["dataset"] == "ami":
            s = ami[h["id"]]
            if s["deciding"]:
                c = sc.filler_counts(s["text"], h["text"], hes, back)
                m = out["ami_test"]
                m[s["meeting"]] = m.get(s["meeting"], sc.Counts()) + c
        elif h["dataset"] == "podcastfillers":
            hypev = sc.hyp_filler_events(h["words"], pf["hypothesis_family"])
            refev = [sc.Event(a, b) for a, b in pf_ev[h["id"]]]
            out["podcastfillers_test"][h["id"]] = sc.time_match(
                hypev, refev, pf["match_tolerance_s"]
            )
    return out


def _counts(table: dict[str, list[int]]) -> dict[str, sc.Counts]:
    return {k: sc.Counts(*v) for k, v in table.items()}


def compare(run_id: str) -> dict[str, Any]:
    res, data = RESULTS / f"r1_{run_id}", DATA / run_id
    ev = yaml.safe_load((REPO / "configs" / "evaluation.yaml").read_text(encoding="utf-8"))
    b = ev["e3"]["bootstrap"]
    t1m = json.loads((res / "e3_metrics.json").read_text("utf-8"))["t1"]
    b5m = json.loads((B5_RESULTS / "e3_metrics.json").read_text("utf-8"))
    for name in ("ami_test.json", "podcastfillers_test_events.json"):  # like for like
        if (data / "items" / name).read_bytes() != (B5_DATA / "items" / name).read_bytes():
            raise ValueError(f"items/{name} differs from run b5's: not like for like")
    t1u = json.loads((res / "units" / "t1.json").read_text("utf-8"))
    paired: dict[str, Any] = {}
    for cond in ("base", "base_initial_prompt"):
        bu = base_units(B5_DATA / "hyps", B5_DATA / "items", cond)
        for s in t1pick.DECIDING:
            tot = sum(bu[s].values(), start=sc.Counts())
            want = b5m[cond][s]
            if (tot.tp, tot.fp, tot.fn) != (want["tp"], want["fp"], want["fn"]):
                raise ValueError(f"{cond} {s}: recomputed counts differ from run b5's totals")
            tu = _counts(t1u[s])
            paired[f"t1_minus_{cond}_{s}"] = {
                "f1_ci95": list(
                    sc.paired_bootstrap(tu, bu[s], sc.f1_of, b["resamples"], b["seed"])
                ),
                "precision_ci95": list(
                    sc.paired_bootstrap(tu, bu[s], sc.precision_of, b["resamples"], b["seed"])
                ),
            }
    rule = t1pick.pass_rule(t1m, b5m["base"], b5m["base_initial_prompt"])
    pw_path = res / "per_window_metrics.json"
    out = {
        "run": f"t1_r1_{run_id}",
        "base_run": f"e3_base_{B5}",
        "pass_rule": rule,
        "paired_bootstrap_95": paired,
        "per_window_report_only": json.loads(pw_path.read_text("utf-8"))
        if pw_path.exists()
        else None,
        "note": "ADR 0026 §4 on point estimates; paired intervals beside them (ADR 0029 §10)",
    }
    (res / "pass_rule.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["build", "ingest", "compare"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--zip", type=Path, help="default: data/colab_results/t1r1_<id>.zip")
    ap.add_argument("--platform", choices=["colab", "kaggle"], default="colab", help="build only")
    args = ap.parse_args(argv)
    if args.action == "build":
        print(f"built {build(args.run_id, platform=args.platform)}")
    elif args.action == "ingest":
        print(json.dumps(ingest(args.run_id, args.zip), indent=2))
    else:
        print(json.dumps(compare(args.run_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
