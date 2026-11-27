"""P4 data run on Google Colab (ADR 0026 §3, ADR 0027). Built by colab_data.py; do not run locally.

Order (DATA_SOURCES §4):
1. download the AMI manual annotations zip, record its SHA-256;
2. read ONLY its licence file; if it does not state CC BY 4.0, stop before any other AMI file;
3. extract; format probe; speaker check (ADR 0012); T1 manifest (prepare_data.build);
4. download each selected headset (2 at a time, retried when the server is busy), record SHA-256
   and duration, check every segment fits, delete the audio;
5. DisfluencySpeech (pinned revision): read only its two text columns, build the T1 text from
   transcript_annotated for every row and compare it with transcript_a (they must differ only at
   partial words);
6. LibriSpeech test-clean: download, SHA-256, count files, delete.

**Resumable** (owner, 2026-11-27, after a Colab disconnect): results go to `P4_OUT` (the notebook
sets it to a folder on the owner's Google Drive; results only, never datasets, ADR 0026 D2). Every
stage writes its files there as it finishes; a rerun of the same build skips finished stages and
finished headsets. A folder written by another build is refused.

`HF_TOKEN` (optional): read from the environment for the Hugging Face download only; never
printed, never written. Without it, or if Hugging Face refuses it, the public dataset is read
anonymously.
"""

import base64
import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import threading
import time
import traceback
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CONFIG = json.loads(r"""__CONFIG_JSON__""")
BUILD = json.loads(r"""__BUILD_JSON__""")
MODULES = json.loads(r"""__MODULES_JSON__""")

OUT = Path(os.environ.get("P4_OUT", "/content/p4data"))
WORK = Path("/content/p4work")  # datasets: local disk only, deleted at the end
LIB = WORK / "lib"
HEADSET_WORKERS = 2  # the AMI site asks for limited bandwidth; run d2 got 503 with 3

LIB.mkdir(parents=True, exist_ok=True)
for name, b64 in MODULES.items():
    (LIB / name).write_bytes(base64.b64decode(b64))
sys.path.insert(0, str(LIB))

import fetch_public_data as fetch  # noqa: E402
import prepare_data  # noqa: E402
import t1text  # noqa: E402

SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_lock = threading.Lock()


def write_json(rel, data):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(p)  # a disconnect never leaves a half-written result


def read_json(rel):
    p = OUT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def progress():
    p = read_json("progress.json")
    if p is None:
        p = {"script_sha256": SCRIPT_SHA, "done": []}
        write_json("progress.json", p)
    if p["script_sha256"] != SCRIPT_SHA:
        raise RuntimeError(f"{OUT} holds results of another build; use a new folder")
    return p


def mark(p, stage):
    if stage not in p["done"]:
        p["done"].append(stage)
        write_json("progress.json", p)
    print(f"stage done: {stage}", flush=True)


def hf_token():
    tok = os.environ.get("HF_TOKEN", "").strip()
    return tok or None


def ds_tables(d):
    """Only the two text columns, read from the pinned revision's parquet files."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    cols = ["transcript_annotated", "transcript_a"]
    last = None
    for token in (hf_token(), False):  # False = anonymous (public dataset)
        if token is None:
            continue
        try:
            fs = HfFileSystem(token=token)
            files = sorted(fs.glob(f"datasets/{d['repo']}@{d['revision']}/**/*.parquet"))
            if not files:
                raise FileNotFoundError("no parquet files at the pinned revision")
            out = {}
            for f in files:
                split = Path(f).name.split("-")[0]
                with fs.open(f, "rb") as fh:
                    out.setdefault(split, []).append(pq.read_table(fh, columns=cols))
            return out, ("token" if token else "anonymous"), files
        except Exception as e:  # a refused token falls back to anonymous; message has no token
            last = f"{type(e).__name__}: {e}"
            print("Hugging Face read failed with", "token" if token else "anonymous", flush=True)
    raise RuntimeError(f"DisfluencySpeech could not be read: {last}")


def ds_check(t1cfg, hes, back):
    d = t1cfg["disfluencyspeech"]
    tables, auth, files = ds_tables(d)
    report = {
        "repo": d["repo"],
        "revision": d["revision"],
        "read_with": auth,
        "files": [Path(f).name for f in files],
        "splits": {},
    }
    for split in sorted(tables):
        rows = [r for t in tables[split] for r in t.to_pylist()]
        kinds = {"same": 0, "punctuation_only": 0, "partials_only": 0, "words_differ": 0}
        fillers = partials = 0
        examples = []
        for row in rows:
            ours = t1text.build_ds_text(row["transcript_annotated"])
            a = row["transcript_a"]
            fillers += t1text.count_fillers(ours, hes, back)
            partials += t1text.count_partials(ours, hes)
            kind = t1text.compare_with_transcript_a(ours, a)
            kinds[kind] += 1
            if kind == "words_differ" and len(examples) < 30:  # punctuation-only rows are fine
                examples.append({"ours": ours, "transcript_a": a})
        report["splits"][split] = {
            "rows": len(rows),
            **kinds,
            "fillers": fillers,
            "partials": partials,
            "words_differ_examples": examples,
        }
    return report


def _headset(meeting, channel, end_s, url_tpl, pins):
    name = f"{meeting}.Headset-{channel}.wav"
    dest = WORK / "audio" / name
    rec = fetch.download(url_tpl.format(meeting=meeting, channel=channel), dest, pins.get(name))
    with wave.open(str(dest), "rb") as w:
        dur = w.getnframes() / w.getframerate()
        rec |= {
            "sample_rate": w.getframerate(),
            "channels": w.getnchannels(),
            "duration_s": round(dur, 3),
        }
    dest.unlink()
    rec["file"] = name
    rec["problem"] = f"segment ends at {end_s} s > {dur:.1f} s" if end_s > dur + 0.5 else None
    with _lock, (OUT / "pins" / "headsets.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"headset {name}: {rec['bytes']} bytes", flush=True)
    return rec


def headsets(url_tpl, pins):
    max_end = {}
    for line in (OUT / "t1" / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        key = (s["meeting"], s["channel"])
        max_end[key] = max(max_end.get(key, 0.0), s["end_s"])
    done_file = OUT / "pins" / "headsets.jsonl"
    done_file.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if done_file.exists():
        done = {json.loads(x)["file"] for x in done_file.read_text(encoding="utf-8").splitlines()}
    todo = [k for k in sorted(max_end) if f"{k[0]}.Headset-{k[1]}.wav" not in done]
    print(f"headsets: {len(max_end)} needed, {len(done)} already done", flush=True)
    with ThreadPoolExecutor(HEADSET_WORKERS) as ex:
        for fut in [ex.submit(_headset, m, c, max_end[(m, c)], url_tpl, pins) for m, c in todo]:
            fut.result()
    rows = [json.loads(x) for x in done_file.read_text(encoding="utf-8").splitlines()]
    return rows, [f"{r['file']}: {r['problem']}" for r in rows if r.get("problem")]


def librispeech(src):
    dest = WORK / "librispeech" / "test-clean.tar.gz"
    rec = fetch.download(src["url"], dest, src.get("sha256"))
    flac = trans = 0
    with tarfile.open(dest, "r:gz") as t:
        for m in t:
            flac += m.name.endswith(".flac")
            trans += m.name.endswith(".trans.txt")
    dest.unlink()
    return {**rec, "flac_files": flac, "transcript_files": trans}


def stage_ami(pd_cfg, info):
    ann = pd_cfg["ami_annotations"]
    zpath = WORK / "ami" / Path(ann["url"]).name
    write_json("pins/ami_annotations.json", fetch.download(ann["url"], zpath, ann.get("sha256")))
    member, text = fetch.licence_text(zpath)
    (OUT / "licence").mkdir(parents=True, exist_ok=True)
    (OUT / "licence" / "AMI_LICENCE.txt").write_text(text, encoding="utf-8")
    ok = fetch.licence_states(text, ann["licence_must_state"])
    write_json(
        "licence/licence_check.json",
        {"member": member, "must_state": ann["licence_must_state"], "passed": ok},
    )
    if not ok:
        return False  # DATA_SOURCES §4: stop before any other AMI file
    root = WORK / "ami" / "extracted"
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():  # official zip, hash recorded; still refuse unsafe paths
            if n.startswith("/") or ".." in n.split("/") or ":" in n or "\\" in n:
                raise ValueError(f"unsafe path in the AMI zip: {n!r}")
        z.extractall(root)  # noqa: S202 (paths checked above)
    prepare_data.build(root, CONFIG, OUT / "t1")
    return True


def main():
    t0 = time.time()
    fetch.require_colab()
    OUT.mkdir(parents=True, exist_ok=True)
    p = progress()
    pd_cfg, t1cfg = CONFIG["public_data"], CONFIG["t1"]
    info = {
        "run_id": BUILD["run_id"],
        "build": BUILD,
        "status": "error",
        "steps": {},
        "script_sha256": SCRIPT_SHA,
        "python": sys.version,
        "platform": platform.platform(),
        "resumed_stages": list(p["done"]),
    }
    print(f"results folder: {OUT}; already done: {p['done'] or 'nothing'}", flush=True)
    try:
        if "ami" not in p["done"]:
            if not stage_ami(pd_cfg, info):
                info["status"] = "licence_failed"
                return info
            mark(p, "ami")
        info["steps"]["licence"] = "passed"
        summary = read_json("t1/manifest_summary.json")
        info["steps"]["manifest"] = {
            k: summary[k] for k in ("speech_hours_kept", "segments", "manifest_sha256")
        }
        info["steps"]["speaker_check_passed"] = read_json("t1/speaker_check.json")["passed"]
        pinned = t1cfg["ami"].get("manifest_sha256")
        if pinned and summary["manifest_sha256"] != pinned:  # run d3 pinned it (ADR 0027 §6)
            info["status"] = "manifest_changed"
            info["steps"]["manifest_pinned"] = pinned
            return info
        probe = read_json("t1/format_probe.json")
        if summary["segments"] == 0 or probe["unknown_elements"]:
            info["steps"]["format_warning"] = probe["unknown_elements"] or "no segments"

        rows, problems = headsets(
            pd_cfg["ami_headset"]["url"], pd_cfg["ami_headset"].get("sha256") or {}
        )
        mark(p, "headsets")
        info["steps"]["headsets"] = len(rows)
        info["steps"]["headset_problems"] = problems

        if "disfluencyspeech" not in p["done"]:
            write_json(
                "ds_text_check.json",
                ds_check(t1cfg, CONFIG["hesitations"], t1cfg["ami"]["backchannels"]),
            )
            mark(p, "disfluencyspeech")
        info["steps"]["disfluencyspeech"] = "checked"

        if "librispeech" not in p["done"]:
            write_json(
                "pins/librispeech_test_clean.json", librispeech(pd_cfg["librispeech_test_clean"])
            )
            mark(p, "librispeech")
        info["steps"]["librispeech"] = "checked"
        info["status"] = "ok" if not problems else "headset_problems"
    except Exception:
        info["error"] = traceback.format_exc()
    finally:
        info["elapsed_s"] = round(time.time() - t0, 1)
    return info


if __name__ == "__main__":
    info = main()
    info["sha256"] = {
        q.relative_to(OUT).as_posix(): {
            "bytes": q.stat().st_size,
            "sha256": hashlib.sha256(q.read_bytes()).hexdigest(),
        }
        for q in sorted(OUT.rglob("*"))
        if q.is_file() and q.name != "run_info.json" and not q.name.endswith(".part")
    }
    (OUT / "run_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(WORK, ignore_errors=True)
    print("P4 DATA DONE: status=" + info["status"])
    sys.exit(0 if info["status"] == "ok" else 1)
