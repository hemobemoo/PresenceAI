"""T1 manifest from the AMI manual annotations (ADR 0026 §3, ADR 0027).

    python prepare_data.py --annotations <unzipped ami_public_manual_1.6.2> --config <eval.json>
        --out <dir>

Writes `manifest.jsonl` (one segment per line: meeting, agent, channel, start_s, end_s, text,
n_words, n_fillers, n_partials, crosstalk_share), `manifest_summary.json` (hours kept, drops by
reason, meetings used, pool exclusions, the manifest's SHA-256) and `format_probe.json`, plus
`speaker_check.json` (ADR 0012). Pure and deterministic: the same annotations and config give
the same manifest hash, which the T1 training run checks before it trains. Runs on Colab; the
laptop runs it only in tests, on hand-written fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ami_words as ami
import t1text


def find_one(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {pattern} under {root}, found {len(hits)}")
    return hits[0]


def word_files(root: Path) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for p in sorted(root.rglob("*.words.xml")):
        parts = p.name.split(".")  # ES2002a.A.words.xml
        if len(parts) == 4:
            out[(parts[0], parts[1])] = p
    if not out:
        raise FileNotFoundError(f"no *.words.xml under {root}")
    return out


def build(root: Path, cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    t1, e3 = cfg["t1"], cfg["e3"]
    a = t1["ami"]
    hes = cfg["hesitations"]
    out.mkdir(parents=True, exist_ok=True)

    meetings = ami.parse_meetings(find_one(root, "meetings.xml").read_bytes())
    files = word_files(root)
    dev = list(e3["tuning"]["ami_dev_sessions"])
    test = list(e3["scoring"]["ami_test_sessions"])
    check = ami.speaker_check(meetings, dev, test)
    pool, excluded = ami.training_pool(
        meetings, e3["ami_split"]["pool"], a["exclude_shared_speakers"]
    )
    order = ami.meeting_order(pool, a["sites"], t1["seed"])

    probe = ami.Probe()
    drops: Counter[str] = Counter()
    segs: dict[str, list[ami.Segment]] = defaultdict(list)
    missing: list[str] = []
    for m in order:
        per_agent: dict[str, list[ami.Item]] = {}
        for spk in meetings[m]:
            f = files.get((m, spk.agent))
            if f is None:
                missing.append(f"{m}.{spk.agent}")
                continue
            per_agent[spk.agent] = ami.parse_words(f.read_bytes(), probe)
        for spk in meetings[m]:
            if spk.agent not in per_agent:
                continue
            others = [
                w
                for ag, items in per_agent.items()
                if ag != spk.agent
                for w in ami.timed_words(items)
            ]
            segs[m] += ami.segments(
                m,
                spk.agent,
                spk.channel,
                per_agent[spk.agent],
                others,
                a["segment"],
                a["spelling"],
                a["backchannels"],
                drops,
            )

    chosen = ami.select(order, segs, float(a["train_speech_hours"]))
    lines = []
    for s in chosen:
        row = asdict(s)
        row["n_fillers"] = t1text.count_fillers(s.text, hes, a["backchannels"])
        row["n_partials"] = t1text.count_partials(s.text, hes)
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    body = ("\n".join(lines) + "\n").encode("utf-8")
    (out / "manifest.jsonl").write_bytes(body)

    used = sorted({s.meeting for s in chosen})
    summary = {
        "speech_hours_kept": round(sum(s.duration_s for s in chosen) / 3600.0, 4),
        "target_hours": float(a["train_speech_hours"]),
        "segments": len(chosen),
        "meetings_used": used,
        "headsets": sorted({f"{s.meeting}.Headset-{s.channel}" for s in chosen}),
        "pool_meetings": len(pool),
        "pool_excluded": excluded,
        "segments_dropped": dict(sorted(drops.items())),
        "segments_considered": sum(len(v) for v in segs.values()) + sum(drops.values()),
        "missing_word_files": missing,
        "fillers": sum(json.loads(x)["n_fillers"] for x in lines),
        "partials": sum(json.loads(x)["n_partials"] for x in lines),
        "words": sum(s.n_words for s in chosen),
        "manifest_sha256": hashlib.sha256(body).hexdigest(),
        "seed": t1["seed"],
    }
    probe_out = {
        "elements": dict(probe.elements.most_common()),
        "w_attributes": dict(probe.w_attributes.most_common()),
        "filler_like_spellings": dict(probe.short_tokens.most_common(40)),
        "untimed_words": probe.untimed_words,
        "unknown_elements": sorted(set(probe.elements) - ami.KNOWN),
    }
    for name, data in (
        ("manifest_summary", summary),
        ("format_probe", probe_out),
        ("speaker_check", check),
    ):
        (out / f"{name}.json").write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True, help="JSON: t1, e3 and hesitations")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(build(args.annotations, cfg, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
