"""E3 base run on Google Colab, T4 (ADR 0028). Built by colab_e3.py; do not run locally.

Stages (each resumable; results written to P4_OUT on the owner's Drive as they finish):
0. ds_check   - fixed DisfluencySpeech builder on all 5,000 rows; pass = words_differ <= 50
                (ADR 0027 §6). Fails -> stop.
1. model      - openai/whisper-small at the app's pinned revision, every source file checked
                against models/MANIFEST.json, converted with the app's converter versions; every
                converted file must match the app's SHA-256 (the same model as the laptop).
2. data       - AMI annotations (pinned) + disfluency-layer probe + test headsets (hashed);
                PodcastFillers licence files read first, then only the test WAVs by range
                requests (CRC32-checked, hashed; full download as fallback); LibriSpeech
                test-clean (pinned).
3. asr_<cond> - whisper-small, the app's decoding (ADR 0028), cond = base | base_initial_prompt;
                one hypothesis per item, appended as it finishes (resumes mid-dataset).
4. score      - fillers (alignment: DisfluencySpeech, AMI; time: PodcastFillers), AMI
                repetitions kept, LibriSpeech WER, bootstrap 95 % intervals.
HF_TOKEN (optional, Colab Secret) is used only for Hugging Face reads and never printed.
"""

import base64
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import zipfile
from pathlib import Path

CONFIG = json.loads(r"""__CONFIG_JSON__""")
BUILD = json.loads(r"""__BUILD_JSON__""")
MODULES = json.loads(r"""__MODULES_JSON__""")

OUT = Path(os.environ.get("P4_OUT", "/content/e3base"))
WORK = Path("/content/e3work")  # datasets and the model: local disk only
LIB = WORK / "lib"
LIB.mkdir(parents=True, exist_ok=True)
for name, b64 in MODULES.items():
    (LIB / name).write_bytes(base64.b64decode(b64))
sys.path.insert(0, str(LIB))

import ami_words as ami  # noqa: E402
import e3score as sc  # noqa: E402
import fetch_public_data as fetch  # noqa: E402
import splitzip  # noqa: E402
import t1text  # noqa: E402

SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
E3, T1, PD = CONFIG["e3"], CONFIG["t1"], CONFIG["public_data"]
HES, BACK = CONFIG["hesitations"], T1["ami"]["backchannels"]
CONDITIONS = ("base", "base_initial_prompt")
_lock = threading.Lock()


# ---------------------------------------------------------------- results folder and progress


def write_json(rel, data):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(p)


def read_json(rel):
    p = OUT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def append_jsonl(rel, row):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock, p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(rel):
    p = OUT / rel
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(json.JSONDecodeError):  # a line cut by a disconnect
            rows.append(json.loads(line))
    return rows


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
    print(f"stage done: {stage}", flush=True)


def log(*a):
    print(*a, flush=True)


def hf_token():
    tok = os.environ.get("HF_TOKEN", "").strip()
    return tok or None


# ---------------------------------------------------------------- stage 0: DisfluencySpeech check


def ds_tables(columns, splits=None):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    d = T1["disfluencyspeech"]
    last = None
    for token in (hf_token(), False):
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
                if splits and split not in splits:
                    continue
                with fs.open(f, "rb") as fh:
                    out.setdefault(split, []).append(pq.read_table(fh, columns=columns))
            return out, ("token" if token else "anonymous")
        except Exception as e:  # message carries no token
            last = f"{type(e).__name__}: {e}"
            log("Hugging Face read failed with", "token" if token else "anonymous")
    raise RuntimeError(f"DisfluencySpeech could not be read: {last}")


def stage_ds_check():
    tables, auth = ds_tables(["transcript_annotated", "transcript_a"])
    rep = {
        "read_with": auth,
        "max_words_differ": T1["disfluencyspeech"]["text_check_max_words_differ"],
        "splits": {},
    }
    total_bad = 0
    for split in sorted(tables):
        kinds = {"same": 0, "punctuation_only": 0, "partials_only": 0, "words_differ": 0}
        examples = []
        for t in tables[split]:
            for row in t.to_pylist():
                ours = t1text.build_ds_text(row["transcript_annotated"])
                k = t1text.compare_with_transcript_a(ours, row["transcript_a"])
                kinds[k] += 1
                if k == "words_differ" and len(examples) < 60:
                    examples.append({"ours": ours, "transcript_a": row["transcript_a"]})
        total_bad += kinds["words_differ"]
        rep["splits"][split] = {
            **kinds,
            "rows": sum(kinds.values()),
            "words_differ_examples": examples,
        }
    rep["words_differ_total"] = total_bad
    rep["passed"] = total_bad <= rep["max_words_differ"]
    write_json("ds_text_check.json", rep)
    return rep["passed"]


# ---------------------------------------------------------------- stage 1: the app's model


def stage_model():
    from huggingface_hub import hf_hub_download

    m = CONFIG["model"]
    src = WORK / "model_src"
    src.mkdir(parents=True, exist_ok=True)
    checks = {"source": [], "converted": []}
    for f in m["source_files"]:
        path = None
        for token in (hf_token(), False):
            if token is None:
                continue
            try:
                path = hf_hub_download(
                    m["source"]["repo"],
                    f["path"],
                    revision=m["source"]["revision"],
                    token=token,
                    local_dir=str(src),
                )
                break
            except Exception:
                log("model download failed with", "token" if token else "anonymous")
        if path is None:
            raise RuntimeError(f"could not download {f['path']}")
        got = fetch_sha(Path(path))
        checks["source"].append({"path": f["path"], "ok": got == f["sha256"]})
        if got != f["sha256"]:
            raise ValueError(f"source {f['path']} differs from the app's MANIFEST.json")
    # Converter environment made with uv, as scripts/download_models.py does on the laptop.
    # Run b1 (2026-11-28): Colab's /usr/bin/python3.12 has no ensurepip, so `python -m venv`
    # failed; uv needs no ensurepip and fetches a CPython of the pinned minor version if needed.
    venv = WORK / "conv"
    uv_pin = f"uv=={CONFIG['uv_version']}"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", uv_pin], check=True)  # noqa: S603
    uv = [sys.executable, "-m", "uv"]
    py_minor = str(m["converter"]["python"])
    subprocess.run([*uv, "venv", "--python", py_minor, str(venv)], check=True)  # noqa: S603
    pkgs = list(m["converter"]["packages"])
    cmd = [*uv, "pip", "install", "--python", str(venv / "bin" / "python"), *pkgs]
    if any(p.startswith("torch==") for p in pkgs):
        cmd += ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
    subprocess.run(cmd, check=True)  # noqa: S603
    dst = WORK / "model"
    subprocess.run(  # noqa: S603 (fixed converter command)
        [
            str(venv / "bin" / "ct2-transformers-converter"),
            "--model",
            str(src),
            "--output_dir",
            str(dst),
            "--quantization",
            m["quantization"],
            "--copy_files",
            *m["copy_files"],
        ],
        check=True,
    )
    same = True
    text_lf = E3["model_text_files_lf_sha256"]  # ADR 0028 §7 (run b2: CRLF on the laptop)
    for f in m["files"]:
        if f["path"] in text_lf:
            data = (dst / f["path"]).read_bytes().replace(b"\r\n", b"\n")
            got, want, how = hashlib.sha256(data).hexdigest(), text_lf[f["path"]], "lf_text"
        else:
            got, want, how = fetch_sha(dst / f["path"]), f["sha256"], "bytes"
        checks["converted"].append(
            {"path": f["path"], "ok": got == want, "sha256": got, "compared": how}
        )
        same &= got == want
    checks["model_identical_to_app"] = same
    write_json("model_check.json", checks)
    shutil.rmtree(src, ignore_errors=True)
    return same


def fetch_sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        str(WORK / "model"), device="cuda", compute_type=E3["decoding"]["compute_type"]
    )


# ---------------------------------------------------------------- stage 2: data


def ami_root():
    root = WORK / "ami" / "extracted"
    if root.exists():
        return root
    ann = PD["ami_annotations"]
    z = WORK / "ami" / Path(ann["url"]).name
    fetch.download(ann["url"], z, ann["sha256"])  # pinned in PR-2
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if n.startswith("/") or ".." in n.split("/") or ":" in n or "\\" in n:
                raise ValueError(f"unsafe path in the AMI zip: {n!r}")
        zf.extractall(root)  # noqa: S202 (paths checked above)
    return root


def ami_ontology(root):
    """Every AMI ontology node, id -> label (ADR 0028 §8: disfluency types are pointer targets)."""
    onto, files = {}, []
    for p in sorted(root.rglob("*.xml")):
        if "ontolog" in p.as_posix().lower() or "type" in p.name.lower():
            try:
                onto.update(ami.load_ontology(p.read_bytes()))
                files.append(p.relative_to(root).as_posix())
            except Exception as e:  # a file that is not an ontology is listed and skipped
                files.append(f"{p.relative_to(root).as_posix()} (skipped: {type(e).__name__})")
    return onto, files


def ami_test_segments():
    """Deciding (crosstalk filter on) and secondary (crosstalk kept) segments, with word ids and
    the repetition events inside each segment."""
    root = ami_root()
    meetings = ami.parse_meetings(next(root.rglob("meetings.xml")).read_bytes())
    test = set(CONFIG["e3"]["scoring"]["ami_test_sessions"])
    seg_cfg = dict(T1["ami"]["segment"])
    loose = {**seg_cfg, "crosstalk_max_share": 1.0}
    words = {(p.name.split(".")[0], p.name.split(".")[1]): p for p in root.rglob("*.words.xml")}
    disf = {(p.name.split(".")[0], p.name.split(".")[1]): p for p in root.rglob("*.disfluency.xml")}
    onto, onto_files = ami_ontology(root)
    probe = ami.DisfluencyProbe()
    rows, n_reps = [], 0
    for m in sorted(x for x in meetings if ami.session(x) in test):
        per = {}
        for spk in meetings[m]:
            if (m, spk.agent) in words:
                per[spk.agent] = ami.parse_words(words[(m, spk.agent)].read_bytes())
        for spk in meetings[m]:
            if spk.agent not in per:
                continue
            others = [w for ag, it in per.items() if ag != spk.agent for w in ami.timed_words(it)]
            reps = []
            if (m, spk.agent) in disf:
                order = [i.id for i in per[spk.agent] if i.id]
                dis = ami.parse_disfluency(disf[(m, spk.agent)].read_bytes(), order, probe, onto)
                reps = [set(d.word_ids) for d in ami.repetitions(dis)]
                n_reps += len(reps)
            args = (m, spk.agent, spk.channel, per[spk.agent], others)
            strict = {
                (s.start_s, s.end_s)
                for s in ami.segments(*args, seg_cfg, T1["ami"]["spelling"], BACK)
            }
            for s in ami.segments(*args, loose, T1["ami"]["spelling"], BACK):
                idx = {w: i for i, w in enumerate(s.word_ids)}
                events = [sorted(idx[w] for w in r) for r in reps if r and r <= set(idx)]
                rows.append(
                    {
                        "id": f"{m}.{spk.agent}.{s.start_s:.2f}",
                        "meeting": m,
                        "channel": s.channel,
                        "start_s": s.start_s,
                        "end_s": s.end_s,
                        "text": s.text,
                        "deciding": (s.start_s, s.end_s) in strict,
                        "repetitions": events,
                        "crosstalk_share": s.crosstalk_share,
                    }
                )
    write_json(
        "probes/ami_disfluency_probe.json",
        {
            "files": len(disf),
            "ontology_files": onto_files,
            "ontology_nodes": len(onto),
            "elements": dict(probe.elements.most_common()),
            "attributes": dict(probe.attributes.most_common()),
            "pointer_roles": dict(probe.pointer_roles.most_common()),
            "pointer_targets": dict(probe.pointer_targets.most_common(20)),
            "type_values": dict(probe.type_values.most_common(60)),
            "unresolved_links": probe.unresolved_links,
            "unresolved_pointers": probe.unresolved_pointers,
            "repetitions_found": n_reps,
        },
    )
    return rows, n_reps


def pf_events(csv_path):
    """(test episodes, scored events per episode) or None if a column cannot be found; writes a
    probe with the header and value counts only (no annotation rows). Rules: e3score.pf_columns."""
    from collections import Counter

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        cols = sc.pf_columns(header)
        rows = list(reader) if all(cols.values()) else []
    labels = Counter(r[cols["label"]] for r in rows) if rows else Counter()
    subsets = Counter(r[cols["subset"]] for r in rows) if rows else Counter()
    write_json(
        "probes/podcastfillers_csv_probe.json",
        {
            "columns": header,
            "mapped": cols,
            "label_values": dict(labels.most_common()),
            "subset_values": dict(subsets.most_common()),
            "rows": len(rows),
        },
    )
    if not all(cols.values()):
        return None
    return sc.pf_test_events(rows, cols, E3["podcastfillers"]["scored_labels"])


def stage_data():
    """Cheap checks first (PodcastFillers columns, AMI disfluency types), then the downloads.
    Returns "ok", or the reason to stop."""
    # 1. PodcastFillers annotation CSV (21 MB): its columns are checked before anything large
    pf = PD["podcastfillers"]
    csv_path = WORK / "pf" / "PodcastFillers.csv"
    a = pf["annotations_csv"]
    rec = fetch.download(pf["files_api"].format(key=a["key"]), csv_path)
    if hashlib.md5(csv_path.read_bytes()).hexdigest() != a["md5"]:  # noqa: S324 (Zenodo's md5)
        raise ValueError("PodcastFillers.csv differs from Zenodo's md5")
    pf_pins = {"annotations_csv": rec}
    got = pf_events(csv_path)
    if got is None:
        return "podcastfillers_columns_unknown"
    test_eps, events = got
    write_json(
        "items/podcastfillers_test_events.json",
        {ep: sorted(events.get(ep, [])) for ep in sorted(test_eps)},
    )
    log(
        f"PodcastFillers: {len(test_eps)} test episodes, "
        f"{sum(len(v) for v in events.values())} scored events"
    )

    # 2. AMI test segments and disfluency types (annotations only)
    segs, n_reps = ami_test_segments()
    write_json("items/ami_test.json", segs)
    log(f"AMI: {len(segs)} test segments, {n_reps} repetitions found in the disfluency layer")

    # 3. PodcastFillers: licence files first, then only the test WAVs
    sizes = [p["bytes"] for p in pf["parts"]]
    urls = [pf["files_api"].format(key=p["key"]) for p in pf["parts"]]
    try:
        parts = splitzip.Parts(sizes, splitzip.http_fetch(urls))
        idx = splitzip.index(parts)
        pf_pins["access"] = "range_requests"
    except splitzip.RangeNotSupportedError:
        log("Zenodo refused range requests: downloading all four parts (about 25 GB)")
        local = []
        for p, u in zip(pf["parts"], urls, strict=True):
            dest = WORK / "pf" / p["key"]
            fetch.download(u, dest)
            if hashlib.md5(dest.read_bytes()).hexdigest() != p["md5"]:  # noqa: S324
                raise ValueError(f"{p['key']} differs from Zenodo's md5") from None
            local.append(dest)
        parts = splitzip.Parts(sizes, splitzip.file_fetch(local))
        idx = splitzip.index(parts)
        pf_pins["access"] = "full_download"
    for lic in pf["licence_members"]:
        name = next((n for n in idx if n.endswith(lic)), None)
        if name is None:
            raise ValueError(f"PodcastFillers licence member {lic} not found")
        r = splitzip.extract(parts, idx[name], OUT / "licence" / "podcastfillers" / lic)
        pf_pins.setdefault("licence_members", {})[lic] = r
    wavs = {}
    for ep in sorted(test_eps):
        name = next((n for n in idx if n.endswith(f"{pf['test_wav_dir']}{ep}.wav")), None)
        if name is None:
            raise ValueError(f"no test WAV for {ep}; first members: {sorted(idx)[:5]}")
        dest = WORK / "pf_wav" / f"{ep}.wav"
        if not dest.exists():
            wavs[ep] = splitzip.extract(parts, idx[name], dest)
            log(f"PodcastFillers {ep}: {wavs[ep]['bytes']} bytes, CRC32 ok")
    pf_pins["test_wavs"] = wavs
    write_json("pins/podcastfillers.json", pf_pins)

    # 4. AMI test headsets (hashed; the AMI server is slow and answers 503 at times)
    pins = []
    for meeting, channel in sorted({(s["meeting"], s["channel"]) for s in segs}):
        name = f"{meeting}.Headset-{channel}.wav"
        dest = WORK / "ami_audio" / name
        if not dest.exists():
            url = PD["ami_headset"]["url"].format(meeting=meeting, channel=channel)
            rec = fetch.download(url, dest, (PD["ami_headset"].get("sha256") or {}).get(name))
            pins.append({"file": name, **rec})
    write_json("pins/ami_test_headsets.json", pins)

    # 5. LibriSpeech test-clean
    lib = PD["librispeech_test_clean"]
    tgz = WORK / "librispeech" / "test-clean.tar.gz"
    if not (WORK / "librispeech" / "LibriSpeech").exists():
        fetch.download(lib["url"], tgz, lib["sha256"])
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(WORK / "librispeech", filter="data")
        tgz.unlink()
    return "ok"


# ---------------------------------------------------------------- stage 3: ASR


def decode_audio(data, sr_out=16000):
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(data) if isinstance(data, bytes) else str(data), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != sr_out:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, sr_out)
        x = resample_poly(x, sr_out // g, sr // g).astype(np.float32)
    return x


def transcribe(model, audio, cond):
    d = E3["decoding"]
    segs, _ = model.transcribe(
        audio,
        language=d["language"],
        beam_size=d["beam_size"],
        temperature=d["temperature"],
        condition_on_previous_text=d["condition_on_previous_text"],
        word_timestamps=d["word_timestamps"],
        vad_filter=d["vad_filter"],
        initial_prompt=E3["initial_prompt"] if cond == "base_initial_prompt" else None,
    )
    words = [
        (w.word.strip(), round(w.start, 3), round(w.end, 3)) for s in segs for w in (s.words or [])
    ]
    return " ".join(w for w, _, _ in words), words


def items():
    """(dataset, item id, loader, reference) for every item, in a fixed order."""
    out = []
    tables, _ = ds_tables(["audio", "transcript_annotated"], splits={"test"})
    for t in tables["test"]:
        for k, row in enumerate(t.to_pylist()):
            out.append(
                (
                    "disfluencyspeech",
                    f"test_{k:04d}",
                    (lambda b=row["audio"]["bytes"]: decode_audio(b)),
                    t1text.build_ds_text(row["transcript_annotated"]),
                )
            )
    cache = {}

    def headset(meeting, channel):
        key = (meeting, channel)
        if key not in cache:
            cache.clear()
            cache[key] = decode_audio(WORK / "ami_audio" / f"{meeting}.Headset-{channel}.wav")
        return cache[key]

    for s in read_json("items/ami_test.json"):
        out.append(
            (
                "ami",
                s["id"],
                (
                    lambda s=s: headset(s["meeting"], s["channel"])[
                        int(s["start_s"] * 16000) : int(s["end_s"] * 16000)
                    ]
                ),
                s["text"],
            )
        )
    for ep in sorted(read_json("items/podcastfillers_test_events.json")):
        out.append(
            (
                "podcastfillers",
                ep,
                (lambda ep=ep: decode_audio(WORK / "pf_wav" / f"{ep}.wav")),
                None,
            )
        )
    for trans in sorted((WORK / "librispeech").rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, text = line.split(" ", 1)
            flac = trans.parent / f"{uid}.flac"
            out.append(("librispeech", uid, (lambda f=flac: decode_audio(f)), text))
    return out


def stage_asr(cond, model, all_items):
    rel = f"hyps/{cond}.jsonl"
    done = {(r["dataset"], r["id"]) for r in read_jsonl(rel)}
    todo = [it for it in all_items if (it[0], it[1]) not in done]
    log(f"{cond}: {len(all_items)} items, {len(done)} done")
    t0 = time.time()
    for n, (ds, iid, load, _ref) in enumerate(todo, 1):
        text, words = transcribe(model, load(), cond)
        append_jsonl(rel, {"dataset": ds, "id": iid, "text": text, "words": words})
        if n % 50 == 0:
            log(f"{cond}: {n}/{len(todo)} new items, {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- stage 4: scoring


def score(all_items):
    refs = {(ds, iid): ref for ds, iid, _, ref in all_items}
    ami_items = {s["id"]: s for s in read_json("items/ami_test.json")}
    pf_events = read_json("items/podcastfillers_test_events.json")
    b = E3["bootstrap"]
    out = {}
    for cond in CONDITIONS:
        hyps = {(r["dataset"], r["id"]): r for r in read_jsonl(f"hyps/{cond}.jsonl")}
        res = {}
        # DisfluencySpeech (reported, not deciding)
        units = {
            iid: sc.filler_counts(refs[(ds, iid)], h["text"], HES, BACK)
            for (ds, iid), h in hyps.items()
            if ds == "disfluencyspeech"
        }
        res["disfluencyspeech_test"] = summarise(units, b, "clip", deciding=False)
        # AMI: deciding (crosstalk filter) and secondary (crosstalk kept); by meeting
        for label, keep in (("ami_test", True), ("ami_test_crosstalk_kept", None)):
            by_meeting, kept, total = {}, 0, 0
            for (ds, iid), h in hyps.items():
                if ds != "ami":
                    continue
                s = ami_items[iid]
                if keep and not s["deciding"]:
                    continue
                c = sc.filler_counts(s["text"], h["text"], HES, BACK)
                by_meeting[s["meeting"]] = by_meeting.get(s["meeting"], sc.Counts()) + c
                k, t = sc.repetitions_kept(s["text"].split(), h["text"].split(), s["repetitions"])
                kept, total = kept + k, total + t
            res[label] = summarise(by_meeting, b, "meeting", deciding=bool(keep))
            # no repetition resolved from the disfluency layer -> "not scored", never a 0 or 1
            res[label]["repetitions_kept"] = (
                {"kept": kept, "total": total, "recall": kept / total}
                if total
                else {
                    "status": "not_scored",
                    "reason": "no repetition resolved; see probes/ami_disfluency_probe.json",
                }
            )
        dec = [s for s in ami_items.values() if s["deciding"]]
        all_s = sum(s["end_s"] - s["start_s"] for s in ami_items.values())
        res["ami_test"]["excluded_share_by_crosstalk_filter"] = (
            1 - sum(s["end_s"] - s["start_s"] for s in dec) / all_s if all_s else None
        )
        # PodcastFillers by time (precision is a lower bound)
        pf = E3["podcastfillers"]
        units = {}
        for (ds, iid), h in hyps.items():
            if ds == "podcastfillers":
                hypev = sc.hyp_filler_events(h["words"], pf["hypothesis_family"])
                refev = [sc.Event(s, e) for s, e in pf_events[iid]]
                units[iid] = sc.time_match(hypev, refev, pf["match_tolerance_s"])
        res["podcastfillers_test"] = summarise(units, b, "episode", deciding=True)
        res["podcastfillers_test"]["precision_is_lower_bound"] = True
        # LibriSpeech WER by speaker
        spk = {}
        for (ds, iid), h in hyps.items():
            if ds == "librispeech":
                r = sc.normalise_wer(refs[(ds, iid)], E3["wer_normaliser"])
                e = sc.word_errors(r, sc.normalise_wer(h["text"], E3["wer_normaliser"]))
                s = iid.split("-")[0]
                spk[s] = (spk.get(s, (0, 0))[0] + e, spk.get(s, (0, 0))[1] + len(r))
        err, wds = sum(v[0] for v in spk.values()), sum(v[1] for v in spk.values())
        lo, hi = sc.bootstrap(spk, sc.wer_of, b["resamples"], b["seed"])
        res["librispeech_test_clean"] = {
            "wer": 100 * err / wds if wds else None,
            "ci95": [lo, hi],
            "words": wds,
            "speakers": len(spk),
        }
        out[cond] = res
    write_json("e3_metrics.json", out)


def summarise(units, b, unit, deciding):
    tot = sum(units.values(), start=sc.Counts())
    lo, hi = sc.bootstrap(units, sc.f1_of, b["resamples"], b["seed"])
    return {
        **sc.prf(tot),
        "tp": tot.tp,
        "fp": tot.fp,
        "fn": tot.fn,
        "f1_ci95": [lo, hi],
        "units": len(units),
        "bootstrap_unit": unit,
        "deciding": deciding,
    }


# ---------------------------------------------------------------- main


def main():
    t0 = time.time()
    fetch.require_colab()
    OUT.mkdir(parents=True, exist_ok=True)
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
    }
    log(f"results folder: {OUT}; already done: {p['done'] or 'nothing'}")
    try:
        if "ds_check" not in p["done"]:
            if not stage_ds_check():
                info["status"] = "ds_text_check_failed"  # ADR 0027 §6: fix, repeat before PR-4
                return info
            mark(p, "ds_check")
        info["steps"]["ds_check"] = "passed"
        if not (WORK / "model" / "model.bin").exists() and not stage_model():
            info["status"] = "model_differs_from_app"
            return info
        mark(p, "model")
        if not (WORK / "librispeech").exists() or "data" not in p["done"]:
            why = stage_data()
            if why != "ok":
                info["status"] = why  # e.g. podcastfillers_columns_unknown: see probes/
                return info
        mark(p, "data")
        import ctranslate2
        import faster_whisper

        info["versions"] = {
            "faster_whisper": faster_whisper.__version__,
            "ctranslate2": ctranslate2.__version__,
        }
        model = load_model()
        all_items = items()
        for cond in CONDITIONS:
            stage_asr(cond, model, all_items)
            mark(p, f"asr_{cond}")
        score(all_items)
        mark(p, "score")
        info["status"] = "ok"
    except Exception:
        info["error"] = traceback.format_exc()
    finally:
        info["elapsed_s"] = round(time.time() - t0, 1)
    return info


if __name__ == "__main__":
    info = main()
    info["sha256"] = {
        q.relative_to(OUT).as_posix(): {"bytes": q.stat().st_size, "sha256": fetch_sha(q)}
        for q in sorted(OUT.rglob("*"))
        if q.is_file() and q.name != "run_info.json" and not q.name.endswith(".part")
    }
    (OUT / "run_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("E3 BASE DONE: status=" + info["status"])
    sys.exit(0 if info["status"] == "ok" else 1)
