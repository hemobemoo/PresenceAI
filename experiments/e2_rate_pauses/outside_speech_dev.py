"""ADR 0015 evidence: transcript words outside VAD speech on the C1 development part.

    uv run python experiments/e2_rate_pauses/outside_speech_dev.py --run f1
    uv run python experiments/e2_rate_pauses/outside_speech_dev.py --run f1 --data <other>/data/c1

Uses only cached transcripts (written by run.py; no ASR is run, a missing transcript stops the
script) and the app's VAD. Development part only (ADR 0014). For each rule and tolerance it
counts the words the rule would remove, split into
- real: aligned to the clip's script by difflib on lower-case letters (fillers count as real),
- extra: not in the script (invented or misheard),
and the mean wpm error with the removed words left out (rate.word_rate, as the app computes it).

Rules (a word is kept when ...):
- end:       it starts at or before the last speech offset + tol (ADR 0015, as changed)
- ends:      its span touches [first speech onset - tol, last speech offset + tol] (first rule)
- regions:   its span touches some speech region widened by tol (also checks pauses)
- mid_ends:  its midpoint lies in [first onset - tol, last offset + tol]
- mid_regions: its midpoint lies in some region widened by tol

Writes experiments/results/e2/<run>_dev_outside_speech.csv and prints the words the rule in use
(end, configs/default.yaml tolerance) removes.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import importlib.util
import json
import re
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

from presenceai.evalconfig import REPO_ROOT
from presenceai.schemas.speech import SpeechRegion, Transcript, Word
from presenceai.settings import get_settings
from presenceai.speech import audio, rate, vad

TOLERANCES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
RULES = ("end", "ends", "regions", "mid_ends", "mid_regions")
FILLERS = {"um", "uh", "er", "erm", "hmm", "mm"}

_spec = importlib.util.spec_from_file_location("e2_run", Path(__file__).with_name("run.py"))
if _spec is None or _spec.loader is None:
    raise ImportError("experiments/e2_rate_pauses/run.py not found")
run = importlib.util.module_from_spec(_spec)
sys.modules["e2_run"] = run
_spec.loader.exec_module(run)


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def keeper(regions: tuple[SpeechRegion, ...], tol: float, rule: str) -> Callable[[Word], bool]:
    first, last = regions[0].start_s - tol, regions[-1].end_s + tol

    def keep(w: Word) -> bool:
        mid = (w.start_s + w.end_s) / 2
        if rule == "end":
            return w.start_s <= last
        if rule == "ends":
            return w.end_s >= first and w.start_s <= last
        if rule == "regions":
            return any(w.end_s >= r.start_s - tol and w.start_s <= r.end_s + tol for r in regions)
        if rule == "mid_ends":
            return first <= mid <= last
        return any(r.start_s - tol <= mid <= r.end_s + tol for r in regions)

    return keep


def without(t: Transcript, keep: Callable[[Word], bool]) -> Transcript:
    """Same transcript minus the removed words (offsets unchanged; only counts are used)."""
    segs = []
    for s in t.segments:
        words = tuple(w for w in s.words if keep(w))
        if words:
            segs.append(s.model_copy(update={"words": words}))
    return t.model_copy(update={"segments": tuple(segs)})


def in_script(words: tuple[Word, ...], script_text: str) -> list[bool]:
    hyp = [norm(w.text) for w in words]
    ref = [norm(x) for x in re.findall(r"[A-Za-z]+", script_text)]
    real = [h in FILLERS for h in hyp]
    sm = difflib.SequenceMatcher(None, hyp, ref, autojunk=False)
    for a, _, n in sm.get_matching_blocks():
        for i in range(a, a + n):
            real[i] = True
    return real


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--data", type=Path, default=REPO_ROOT / "data" / "c1",
        help="folder holding <run>/clips and <run>/cache (default: this checkout)",
    )  # fmt: skip
    ap.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="which cached transcripts (the ASR device is part of the cache key)",
    )  # fmt: skip
    args = ap.parse_args(argv)
    paths = run.Paths(args.run, data=args.data)
    settings = get_settings()
    speech, asr_opts = settings.speech, settings.asr
    if args.device is not None:
        asr_opts = asr_opts.model_copy(update={"device": args.device})
    parts = run.read_split(paths)
    truths = [t for t in run.load_truths(paths) if parts.get(t.clip_id) == "dev"]

    tally = {(r, t): {"real": 0, "extra": 0, "clips": set(), "err": []}
             for r in RULES for t in TOLERANCES}  # fmt: skip
    base_err: list[float] = []
    extra_total = 0
    chosen: list[tuple[str, str, float, float, float, str]] = []
    tol_cfg = speech.transcript.end_tolerance_s
    cache_dirs: set[str] = set()
    for truth in truths:
        clip = paths.clips / f"{truth.clip_id}.flac"
        f = run.cache_file(paths, asr_opts.model, asr_opts, clip)
        if not f.exists():
            raise SystemExit(f"no cached transcript for {truth.clip_id} ({f}); run run.py first")
        cache_dirs.add(f.parent.name)
        t = Transcript.model_validate_json(f.read_text(encoding="utf-8"))
        regions = vad.detect_speech(audio.load_audio(clip), speech.vad)
        real = in_script(t.words, truth.text)
        extra_total += real.count(False)
        m, _, _ = rate.word_rate(t, regions, speech)
        base_err.append(abs(m["rate.wpm"].value - truth.wpm) / truth.wpm * 100)
        for rule in RULES:
            for tol in TOLERANCES:
                keep = keeper(regions, tol, rule)
                cell = tally[(rule, tol)]
                for w, is_real in zip(t.words, real, strict=True):
                    if keep(w):
                        continue
                    cell["real" if is_real else "extra"] += 1
                    cell["clips"].add(truth.clip_id)
                    if rule == "end" and tol == tol_cfg:
                        chosen.append((truth.clip_id, w.text, round(w.start_s, 2),
                                       round(w.end_s, 2), round(w.probability, 2),
                                       "real" if is_real else "extra"))  # fmt: skip
                m, _, _ = rate.word_rate(without(t, keep), regions, speech)
                cell["err"].append(abs(m["rate.wpm"].value - truth.wpm) / truth.wpm * 100)

    out = paths.results_e2 / f"{args.run}_dev_outside_speech.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "tolerance_s", "real_removed", "extra_removed", "clips_affected",
                    "wpm_error_mean_pct", "wpm_error_max_pct", "clips_over_5pct"])  # fmt: skip
        w.writerow(["none", "", 0, 0, 0, f"{statistics.mean(base_err):.3f}",
                    f"{max(base_err):.2f}", sum(e > 5 for e in base_err)])  # fmt: skip
        for (rule, tol), c in tally.items():
            w.writerow([rule, tol, c["real"], c["extra"], len(c["clips"]),
                        f"{statistics.mean(c['err']):.3f}", f"{max(c['err']):.2f}",
                        sum(e > 5 for e in c["err"])])  # fmt: skip
    print(json.dumps({"dev_clips": len(truths), "words_not_in_script": extra_total,
                      "cache": sorted(cache_dirs)}, indent=1))  # fmt: skip
    print(out.read_text(encoding="utf-8"))
    print(f"removed by 'end' at {tol_cfg} s (configs/default.yaml):")
    for row in chosen:
        print("  ", row)
    print(f"wrote {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
