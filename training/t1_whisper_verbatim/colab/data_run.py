"""P4 data run on Google Colab (ADR 0026 §3, ADR 0027). Built by colab_data.py; do not run locally.

Order (DATA_SOURCES §4):
1. download the AMI manual annotations zip, record its SHA-256;
2. read ONLY its licence file; if it does not state CC BY 4.0, stop before any other AMI file;
3. extract; format probe; speaker check (ADR 0012); T1 manifest (prepare_data.build);
4. download each selected headset, record SHA-256 and duration, check every segment fits, delete;
5. DisfluencySpeech (pinned revision): build the T1 text from transcript_annotated for every row and
   compare it with transcript_a (they must differ only at partial words);
6. LibriSpeech test-clean: download, SHA-256, count files, delete.
Writes /content/p4data/ with run_info.json (SHA-256 of every file). No secret, no Drive.
"""

import base64
import hashlib
import json
import platform
import shutil
import sys
import tarfile
import time
import traceback
import wave
import zipfile
from pathlib import Path

CONFIG = json.loads(r"""__CONFIG_JSON__""")
BUILD = json.loads(r"""__BUILD_JSON__""")
MODULES = json.loads(r"""__MODULES_JSON__""")

OUT = Path("/content/p4data")
WORK = Path("/content/p4work")
LIB = WORK / "lib"

LIB.mkdir(parents=True, exist_ok=True)
for name, b64 in MODULES.items():
    (LIB / name).write_bytes(base64.b64decode(b64))
sys.path.insert(0, str(LIB))

import fetch_public_data as fetch  # noqa: E402
import prepare_data  # noqa: E402
import t1text  # noqa: E402


def write_json(rel, data):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def ds_check(t1cfg, hes, back):
    from datasets import load_dataset

    d = t1cfg["disfluencyspeech"]
    ds = load_dataset(d["repo"], revision=d["revision"])
    report = {"repo": d["repo"], "revision": d["revision"], "splits": {}}
    for split in sorted(ds):
        part = ds[split].select_columns(["transcript_annotated", "transcript_a"])
        same = len_diff = word_diff = fillers = partials = 0
        examples = []
        for row in part:
            ours = t1text.build_ds_text(row["transcript_annotated"])
            a = row["transcript_a"]
            ot, at = ours.split(), a.split()
            fillers += t1text.count_fillers(ours, hes, back)
            partials += t1text.count_partials(ours, hes)
            if len(ot) != len(at):
                len_diff += 1
                bad = True
            else:
                bad = any(x != y and not t1text.is_partial(x) for x, y in zip(ot, at, strict=True))
                word_diff += bad
            same += not bad
            if bad and len(examples) < 20:
                examples.append({"ours": ours, "transcript_a": a})
        report["splits"][split] = {
            "rows": len(part),
            "equal_except_partials": same,
            "length_differs": len_diff,
            "word_differs": word_diff,
            "fillers": fillers,
            "partials": partials,
            "examples": examples,
        }
    return report


def headsets(chosen, url_tpl, pins):
    rows, problems = [], []
    max_end = {}
    for line in (OUT / "t1" / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        key = (s["meeting"], s["channel"])
        max_end[key] = max(max_end.get(key, 0.0), s["end_s"])
    for meeting, channel in sorted(max_end):
        url = url_tpl.format(meeting=meeting, channel=channel)
        dest = WORK / "audio" / f"{meeting}.Headset-{channel}.wav"
        name = dest.name
        rec = fetch.download(url, dest, pins.get(name))
        with wave.open(str(dest), "rb") as w:
            dur = w.getnframes() / w.getframerate()
            rec |= {
                "sample_rate": w.getframerate(),
                "channels": w.getnchannels(),
                "duration_s": round(dur, 3),
            }
        if max_end[(meeting, channel)] > dur + 0.5:
            problems.append(
                f"{name}: segment ends at {max_end[(meeting, channel)]} s > {dur:.1f} s"
            )
        rows.append({"file": name, **rec})
        dest.unlink()
    return rows, problems


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


def main():
    t0 = time.time()
    fetch.require_colab()
    OUT.mkdir(parents=True, exist_ok=True)
    pd_cfg, t1cfg = CONFIG["public_data"], CONFIG["t1"]
    info = {
        "run_id": BUILD["run_id"],
        "build": BUILD,
        "status": "error",
        "steps": {},
        "python": sys.version,
        "platform": platform.platform(),
    }
    pins = {}
    try:
        ann = pd_cfg["ami_annotations"]
        zpath = WORK / "ami" / Path(ann["url"]).name
        pins["ami_annotations"] = fetch.download(ann["url"], zpath, ann.get("sha256"))
        member, text = fetch.licence_text(zpath)
        (OUT / "licence").mkdir(parents=True, exist_ok=True)
        (OUT / "licence" / "AMI_LICENCE.txt").write_text(text, encoding="utf-8")
        ok = fetch.licence_states(text, ann["licence_must_state"])
        write_json(
            "licence/licence_check.json",
            {"member": member, "must_state": ann["licence_must_state"], "passed": ok},
        )
        if not ok:
            info["status"] = "licence_failed"  # DATA_SOURCES §4: stop before any other AMI file
            return info, pins
        info["steps"]["licence"] = "passed"

        root = WORK / "ami" / "extracted"
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():  # official zip, hash recorded; still refuse unsafe paths
                if n.startswith("/") or ".." in n.split("/") or ":" in n or "\\" in n:
                    raise ValueError(f"unsafe path in the AMI zip: {n!r}")
            z.extractall(root)  # noqa: S202 (paths checked above)
        summary = prepare_data.build(root, CONFIG, OUT / "t1")
        info["steps"]["manifest"] = {
            k: summary[k] for k in ("speech_hours_kept", "segments", "manifest_sha256")
        }
        check = json.loads((OUT / "t1" / "speaker_check.json").read_text(encoding="utf-8"))
        info["steps"]["speaker_check_passed"] = check["passed"]
        probe = json.loads((OUT / "t1" / "format_probe.json").read_text(encoding="utf-8"))
        if summary["segments"] == 0 or probe["unknown_elements"]:
            info["steps"]["format_warning"] = probe["unknown_elements"] or "no segments"

        rows, problems = headsets(
            summary, pd_cfg["ami_headset"]["url"], pd_cfg["ami_headset"].get("sha256") or {}
        )
        pins["ami_headset"] = rows
        info["steps"]["headset_problems"] = problems

        write_json(
            "ds_text_check.json",
            ds_check(t1cfg, CONFIG["hesitations"], t1cfg["ami"]["backchannels"]),
        )
        info["steps"]["disfluencyspeech"] = "checked"

        pins["librispeech_test_clean"] = librispeech(pd_cfg["librispeech_test_clean"])
        info["steps"]["librispeech"] = "checked"
        info["status"] = "ok" if not problems else "headset_problems"
    except Exception:
        info["error"] = traceback.format_exc()
    finally:
        info["elapsed_s"] = round(time.time() - t0, 1)
    return info, pins


if __name__ == "__main__":
    info, pins = main()
    write_json("source_pins.json", pins)
    info["sha256"] = {
        p.relative_to(OUT).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(OUT.rglob("*"))
        if p.is_file() and p.name != "run_info.json"
    }
    (OUT / "run_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(WORK, ignore_errors=True)
    print("P4 DATA DONE: status=" + info["status"])
    sys.exit(0 if info["status"] == "ok" else 1)
