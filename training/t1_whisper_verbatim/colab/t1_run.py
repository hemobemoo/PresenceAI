"""T1-r1 on Google Colab, T4 (ADR 0029). Built by colab_t1.py; do not run locally.

Stages (each resumable; results to P4_OUT on the owner's Drive as they finish; checkpoints and
converted candidates to the Drive work folder next to it, never into the zip):
0. ds_check   - the DisfluencySpeech builder check (ADR 0027 §6), as in E3.
1. model      - the app's whisper-small, checked and converted exactly as in E3 (ADR 0028 §1);
                the pinned HF files are kept for training and merging.
2. train_data - DisfluencySpeech train and validation (T1 text builder) to local parquet.
3. train      - Unsloth LoRA, configs/training.yaml t1.r1 (ADR 0029 §2), own process.
4. dev_data   - AMI dev segments and headsets (pinned on first download, ADR 0029 §6).
5. pick       - every epoch merged, converted with the app's converter, decoded with the app's
                settings on DisfluencySpeech val + AMI dev; pick on dev only (ADR 0029 §5).
6. publish    - the picked model to the owner's private HF repo; commit SHA and file hashes.
7. load_check - faster-whisper loads it on CPU (int8) and decodes one clip.
8. e3_t1      - E3 test sets with condition t1, by the E3 code itself (ADR 0029 §7).
9. per_window - PodcastFillers per-window conditions, report-only (ADR 0028 §12, ADR 0029 §8).
HF_TOKEN (Colab Secret, write token for the T1 repo only) is used by name and never printed.
"""

import base64
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

CONFIG = json.loads(r"""__CONFIG_JSON__""")
BUILD = json.loads(r"""__BUILD_JSON__""")
MODULES = json.loads(r"""__MODULES_JSON__""")

LIB = Path(os.environ.get("P4_WORK", "/content/e3work")) / "lib"  # as e3_run.WORK
LIB.mkdir(parents=True, exist_ok=True)
for name, b64 in MODULES.items():
    (LIB / name).write_bytes(base64.b64decode(b64))
sys.path.insert(0, str(LIB))

import chunks  # noqa: E402
import e3_run as e3  # noqa: E402  (the built E3 script: same protocol, same code)
import t1merge  # noqa: E402
import t1pick  # noqa: E402

sc, fetch, t1text, ami = e3.sc, e3.fetch, e3.t1text, e3.ami
OUT, WORK = e3.OUT, e3.WORK
# Checkpoints and converted candidates: next to the results folder (Drive on Colab,
# /kaggle/working on Kaggle), never inside it, so never in the zip (ADR 0029 §2, §11).
WORKD = Path(os.environ["P4_WORKD"]) if os.environ.get("P4_WORKD") else OUT.parent / "work"
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
TR, E3 = CONFIG["training"], CONFIG["e3"]
RUN = TR["t1"]["r1"]
PICK, PUB = TR["t1"]["pick"], TR["t1"]["publish"]
HES, BACK = e3.HES, e3.BACK
log, write_json, read_json, read_jsonl, append_jsonl = (
    e3.log,
    e3.write_json,
    e3.read_json,
    e3.read_jsonl,
    e3.append_jsonl,
)


def progress():
    p = read_json("progress.json") or {"script_sha256": SCRIPT_SHA, "done": []}
    if p["script_sha256"] != SCRIPT_SHA:
        raise RuntimeError(f"{OUT} holds results of another build; use a new folder")
    write_json("progress.json", p)
    return p


def mark(p, stage):
    if stage not in p["done"]:
        p["done"].append(stage)
        write_json("progress.json", p)
    log(f"stage done: {stage}")


def sub(mode, args):
    """Run t1train.py in its own process (its output streams to the cell)."""
    path = WORK / f"{mode}_args.json"
    path.write_text(json.dumps(args), encoding="utf-8")
    subprocess.run([sys.executable, str(LIB / "t1train.py"), mode, str(path)], check=True)  # noqa: S603


# ---------------------------------------------------------------- stage 2: training data


def stage_train_data():
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables, _ = e3.ds_tables(["audio", "transcript_annotated"], splits={"train", "validation"})
    counts = {}
    for split in ("train", "validation"):
        rows = [r for t in tables[split] for r in t.to_pylist()]
        tab = pa.table(
            {
                "id": [f"{split}_{k:04d}" for k in range(len(rows))],
                "audio": [r["audio"]["bytes"] for r in rows],
                "text": [t1text.build_ds_text(r["transcript_annotated"]) for r in rows],
            }
        )
        (WORK / "ds").mkdir(parents=True, exist_ok=True)
        pq.write_table(tab, WORK / "ds" / f"{split}.parquet")
        counts[split] = len(rows)
    log(f"DisfluencySpeech: {counts}")
    return counts


# ---------------------------------------------------------------- stage 3: training


def stage_train():
    ckpt = WORKD / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    sub(
        "train",
        {
            "src": str(WORK / "model_src"),
            "data": str(WORK / "ds" / "train.parquet"),
            "ckpt_dir": str(ckpt),
            "hp": RUN,
            "report": str(OUT / "train_report.json"),
        },
    )
    rep = read_json("train_report.json")
    if [c["epoch"] for c in rep["checkpoints"]] != list(range(1, RUN["epochs"] + 1)):
        raise ValueError(f"expected one checkpoint per epoch, got {rep['checkpoints']}")


def checkpoint(epoch):
    rep = read_json("train_report.json")
    name = next(c["name"] for c in rep["checkpoints"] if c["epoch"] == epoch)
    return WORKD / "checkpoints" / name


# ---------------------------------------------------------------- stage 4: AMI dev


def dev_sessions():
    pool, test = E3["ami_split"]["pool"], set(E3["scoring"]["ami_test_sessions"])
    return [s for s in pool if s not in test]


def stage_dev_data():
    segs, _ = e3.ami_test_segments(sessions=dev_sessions(), probe_rel=None)
    dev = [s for s in segs if s["deciding"]]  # the deciding crosstalk filter (ADR 0029 §5)
    write_json("items/ami_dev.json", dev)
    pins = read_json("pins/ami_dev_headsets.json") or {}
    known = e3.PD["ami_headset"].get("sha256") or {}
    for meeting, channel in sorted({(s["meeting"], s["channel"]) for s in dev}):
        name = f"{meeting}.Headset-{channel}.wav"
        dest = WORK / "ami_audio" / name
        if dest.exists():
            continue
        pin = known.get(name) or (pins.get(name) or {}).get("sha256")
        url = e3.PD["ami_headset"]["url"].format(meeting=meeting, channel=channel)
        rec = fetch.download(url, dest, pin, cache_dir=e3.cache("ami_headsets"))
        if not pin:  # ADR 0029 §6: pinned on first download, then cached under that hash
            fetch.cache_put(e3.cache("ami_headsets"), name, rec["sha256"], dest, url)
            pins[name] = {**rec, "pinned_on_first_download": True}
            write_json("pins/ami_dev_headsets.json", pins)
            log(f"AMI dev {name}: downloaded, pinned and cached")
    log(f"AMI dev: {len(dev)} segments in {dev_sessions()}")


def dev_items():
    import pyarrow.parquet as pq

    out = []
    for r in pq.read_table(WORK / "ds" / "validation.parquet").to_pylist():
        out.append(
            ("disfluencyspeech_val", r["id"], (lambda b=r["audio"]: e3.decode_audio(b)), r["text"])
        )
    cache = {}

    def headset(meeting, channel):
        if (meeting, channel) not in cache:
            cache.clear()
            cache[(meeting, channel)] = e3.decode_audio(
                WORK / "ami_audio" / f"{meeting}.Headset-{channel}.wav"
            )
        return cache[(meeting, channel)]

    for s in read_json("items/ami_dev.json"):
        out.append(
            (
                "ami_dev",
                s["id"],
                (
                    lambda s=s: headset(s["meeting"], s["channel"])[
                        int(s["start_s"] * 16000) : int(s["end_s"] * 16000)
                    ]
                ),
                s["text"],
            )
        )
    return out


# ---------------------------------------------------------------- stage 5: pick


def merge_checkpoint(epoch, out):
    """LoRA merged into the base's own fp32 weights; every other file is the base's (§3)."""
    from safetensors.numpy import load_file, save_file

    ckpt, src = checkpoint(epoch), WORK / "model_src"
    cfg = json.loads((ckpt / "adapter_config.json").read_text(encoding="utf-8"))
    lora = RUN["lora"]
    scaling = t1merge.check_config(cfg, lora["r"], lora["alpha"], lora["target_modules"])
    base = load_file(str(src / "model.safetensors"))
    merged, changed = t1merge.merge(
        base, load_file(str(ckpt / "adapter_model.safetensors")), scaling, lora["target_modules"]
    )
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy2(f, out / f.name)
    save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
    write_json(
        f"merge/epoch_{epoch}.json",
        {
            "checkpoint": ckpt.name,
            "scaling": scaling,
            "changed_tensors": len(changed),
            "tensors": len(merged),
            "changed_examples": changed[:4],
        },
    )


def build_candidate(epoch):
    ct2 = WORKD / "ct2" / f"epoch_{epoch}"
    if (ct2 / "model.bin").exists():
        return ct2
    merged = WORK / "merged" / f"epoch_{epoch}"
    merge_checkpoint(epoch, merged)
    tmp = ct2.with_name(ct2.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    e3.convert(merged, tmp)
    shutil.rmtree(ct2, ignore_errors=True)
    shutil.move(str(tmp), str(ct2))
    shutil.rmtree(merged, ignore_errors=True)
    return ct2


def load(ct2, device="cuda"):
    from faster_whisper import WhisperModel

    return WhisperModel(str(ct2), device=device, compute_type=E3["decoding"]["compute_type"])


def decode_all(rel, model, todo_items, cond):
    done = {(r["dataset"], r["id"]) for r in read_jsonl(rel)}
    todo = [it for it in todo_items if (it[0], it[1]) not in done]
    log(f"{rel}: {len(todo_items)} items, {len(done)} done")
    t0 = time.time()
    for n, (ds, iid, get, _ref) in enumerate(todo, 1):
        text, words = e3.transcribe(model, get(), cond)
        append_jsonl(rel, {"dataset": ds, "id": iid, "text": text, "words": words})
        if n % 50 == 0 or n == len(todo):
            log(f"{rel}: {n}/{len(todo)} new items, {time.time() - t0:.0f} s")


def dev_scores(rel, items_):
    refs = {(ds, iid): ref for ds, iid, _, ref in items_}
    hyps = read_jsonl(rel)
    f1, wer = {}, {}
    for ds in ("disfluencyspeech_val", "ami_dev"):
        c, err, words = sc.Counts(), 0, 0
        for h in hyps:
            if h["dataset"] != ds:
                continue
            ref = refs[(ds, h["id"])]
            c = c + sc.filler_counts(ref, h["text"], HES, BACK)
            r = sc.normalise_wer(ref, E3["wer_normaliser"])
            err += sc.word_errors(r, sc.normalise_wer(h["text"], E3["wer_normaliser"]))
            words += len(r)
        name = {"disfluencyspeech_val": "disfluencyspeech_validation"}.get(ds, ds)
        f1[name] = sc.prf(c)["f1"]
        wer[name] = 100 * err / words if words else None
    return f1, wer


def stage_pick():
    items_ = dev_items()
    cands = []
    for epoch in range(1, RUN["epochs"] + 1):
        ct2 = build_candidate(epoch)
        rel = f"hyps/pick_epoch_{epoch}.jsonl"
        model = load(ct2)
        decode_all(rel, model, items_, "t1")
        del model
        f1, wer = dev_scores(rel, items_)
        if sorted(f1) != sorted(PICK["sets"]):  # dev only, the owner's two sets (ADR 0029 §5)
            raise ValueError(f"pick sets {sorted(f1)} differ from {PICK['sets']}")
        cands.append({"epoch": epoch, "f1": f1, "wer": wer})
        log(f"epoch {epoch}: dev F1 {f1}, WER {wer}")
    res = t1pick.pick(cands, PICK["tie_within"])
    res["candidates"] = cands
    res["rule"] = PICK
    write_json("pick.json", res)
    log(f"picked epoch {res['picked_epoch']}")
    return res["picked_epoch"]


# ---------------------------------------------------------------- stage 6: publish


def stage_publish(epoch):
    from huggingface_hub import HfApi

    tok = os.environ.get(PUB["token_env"], "").strip()
    if not tok:
        raise RuntimeError(f"Colab Secret {PUB['token_env']} is missing (write token for the repo)")
    api = HfApi(token=tok)
    if not api.repo_info(PUB["repo"], repo_type="model").private:
        raise RuntimeError(f"{PUB['repo']} is not private; nothing uploaded")
    stage = WORK / "publish"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(WORKD / "ct2" / f"epoch_{epoch}", stage / PUB["folders"]["ct2"])
    (stage / PUB["folders"]["lora"]).mkdir(parents=True)
    for f in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(checkpoint(epoch) / f, stage / PUB["folders"]["lora"] / f)
    files = {
        p.relative_to(stage).as_posix(): {"bytes": p.stat().st_size, "sha256": e3.fetch_sha(p)}
        for p in sorted(stage.rglob("*"))
        if p.is_file()
    }
    info = api.upload_folder(
        repo_id=PUB["repo"],
        repo_type="model",
        folder_path=str(stage),
        commit_message=f"T1-r1 run {BUILD['run_id']}: epoch {epoch} (ADR 0029)",
    )
    rec = {"repo": PUB["repo"], "commit": info.oid, "epoch": epoch, "files": files}
    write_json("publish.json", rec)
    log(f"published to {PUB['repo']} at commit {info.oid}")
    return rec


# ---------------------------------------------------------------- stage 7: load check


def stage_load_check(epoch):
    items_ = dev_items()
    t0 = time.time()
    model = load(WORKD / "ct2" / f"epoch_{epoch}", device="cpu")
    text, _ = e3.transcribe(model, items_[0][2](), "t1")
    write_json(
        "load_check.json",
        {
            "device": "cpu",
            "compute_type": E3["decoding"]["compute_type"],
            "ok": bool(text),
            "item": items_[0][1],
            "words": len(text.split()),
            "seconds": round(time.time() - t0, 1),
        },
    )
    del model


# ---------------------------------------------------------------- stage 9: per window


def vad_regions(audio):
    from faster_whisper import vad as fw_vad

    v = CONFIG["app_vad"]
    opts = fw_vad.VadOptions(
        threshold=v["threshold"],
        min_speech_duration_ms=v["min_speech_duration_ms"],
        min_silence_duration_ms=v["min_silence_duration_ms"],
        speech_pad_ms=v["speech_pad_ms"],
    )
    stamps = fw_vad.get_speech_timestamps(audio, opts, sampling_rate=16000)
    return [(s["start"] / 16000, s["end"] / 16000) for s in stamps]


def stage_per_window(t1_model):
    pw = E3["per_window"]
    eps = sorted(read_json("items/podcastfillers_test_events.json"))
    plans = read_json("items/pf_chunks.json") or {}
    base = load(WORK / "model")
    models = {
        "base_initial_prompt_per_window": (base, "base_initial_prompt"),
        "t1_per_window": (t1_model, "t1"),
    }
    for ep in eps:
        audio = None
        if ep not in plans:
            audio = e3.decode_audio(WORK / "pf_wav" / f"{ep}.wav")
            regs = vad_regions(audio)
            plans[ep] = {
                "regions": len(regs),
                "chunks": chunks.plan_chunks(regs, len(audio) / 16000, pw["max_chunk_s"]),
            }
            write_json("items/pf_chunks.json", plans)
        for cond, (model, how) in models.items():
            rel = f"hyps/{cond}.jsonl"
            if any(r["id"] == ep for r in read_jsonl(rel)):
                continue
            if audio is None:
                audio = e3.decode_audio(WORK / "pf_wav" / f"{ep}.wav")
            words = []
            for s, e in plans[ep]["chunks"]:
                _, w = e3.transcribe(model, audio[int(s * 16000) : int(e * 16000)], how)
                words += [(t, round(a + s, 3), round(b + s, 3)) for t, a, b in w]
            append_jsonl(
                rel,
                {
                    "dataset": "podcastfillers",
                    "id": ep,
                    "text": " ".join(t for t, _, _ in words),
                    "words": words,
                },
            )
        log(f"per window {ep}: {len(plans[ep]['chunks'])} chunks")
    pf, b = E3["podcastfillers"], E3["bootstrap"]
    events = read_json("items/podcastfillers_test_events.json")
    out = {}
    for cond in models:
        units = {}
        for h in read_jsonl(f"hyps/{cond}.jsonl"):
            hypev = sc.hyp_filler_events(h["words"], pf["hypothesis_family"])
            refev = [sc.Event(s, e) for s, e in events[h["id"]]]
            units[h["id"]] = sc.time_match(hypev, refev, pf["match_tolerance_s"])
        out[cond] = {
            "podcastfillers_test": {
                **e3.summarise(units, b, "episode", deciding=False),
                "precision_is_lower_bound": True,
                "report_only": True,
            }
        }
        write_json(f"units/{cond}.json", {"podcastfillers_test": e3.counts_table(units)})
    write_json("per_window_metrics.json", out)


# ---------------------------------------------------------------- main


def main():
    t0 = time.time()
    fetch.require_cloud()
    OUT.mkdir(parents=True, exist_ok=True)
    WORKD.mkdir(parents=True, exist_ok=True)
    p = progress()
    info = {
        "run_id": BUILD["run_id"],
        "build": BUILD,
        "status": "error",
        "steps": {},
        "script_sha256": SCRIPT_SHA,
        "python": sys.version,
        "platform": platform.platform(),
        "resumed_stages": list(p["done"]),
        "run": "t1_r1",
    }
    log(f"results folder: {OUT}; work folder: {WORKD}; already done: {p['done'] or 'nothing'}")
    try:
        if "ds_check" not in p["done"]:
            if not e3.stage_ds_check():
                info["status"] = "ds_text_check_failed"
                return info
            mark(p, "ds_check")
        missing = not (WORK / "model" / "model.bin").exists() or not (WORK / "model_src").exists()
        if missing and not e3.stage_model(keep_source=True):
            info["status"] = "model_differs_from_app"
            return info
        mark(p, "model")
        if not (WORK / "ds" / "validation.parquet").exists():
            info["steps"]["train_data"] = stage_train_data()
        mark(p, "train_data")
        if "train" not in p["done"]:
            stage_train()
            mark(p, "train")
        if "dev_data" not in p["done"] or not (WORK / "ami_audio").exists():
            stage_dev_data()
        mark(p, "dev_data")
        if "pick" not in p["done"]:
            stage_pick()
            mark(p, "pick")
        epoch = read_json("pick.json")["picked_epoch"]
        build_candidate(epoch)  # rebuilt after a reset if the Drive copy is gone
        if "publish" not in p["done"]:
            stage_publish(epoch)
            mark(p, "publish")
        if "load_check" not in p["done"]:
            stage_load_check(epoch)
            mark(p, "load_check")
        # test data only now, after the pick is written (ADR 0029 §5)
        if not (WORK / "librispeech").exists() or "test_data" not in p["done"]:
            why = e3.stage_data()
            if why != "ok":
                info["status"] = why
                return info
        mark(p, "test_data")
        import ctranslate2
        import faster_whisper

        info["versions"] = {
            "faster_whisper": faster_whisper.__version__,
            "ctranslate2": ctranslate2.__version__,
        }
        t1_model = load(WORKD / "ct2" / f"epoch_{epoch}")
        all_items = e3.items()
        e3.stage_asr("t1", t1_model, all_items)
        e3.CONDITIONS = ("t1",)
        e3.score(all_items)
        mark(p, "e3_t1")
        stage_per_window(t1_model)
        mark(p, "per_window")
        info["status"] = "ok"
    except Exception:
        info["error"] = traceback.format_exc()
    finally:
        info["elapsed_s"] = round(time.time() - t0, 1)
    return info


if __name__ == "__main__":
    info = main()
    info["sha256"] = {
        q.relative_to(OUT).as_posix(): {"bytes": q.stat().st_size, "sha256": e3.fetch_sha(q)}
        for q in sorted(OUT.rglob("*"))
        if q.is_file() and q.name != "run_info.json" and not q.name.endswith(".part")
    }
    pub = read_json("publish.json")
    if pub:  # PR-5's download_models.py pins these (ADR 0029 §9)
        info["published"] = {"repo": pub["repo"], "commit": pub["commit"], "files": pub["files"]}
    (OUT / "run_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("T1-R1 DONE: status=" + info["status"])
    sys.exit(0 if info["status"] == "ok" else 1)
