"""ADR 0016 open item 6 (B1-informed): which side of a pause a stretched word is on, C1 dev only.

    uv run python experiments/e2_rate_pauses/word_side_dev.py --run f1

B1 round 1 showed that Whisper often stretches a word into the silence next to it, so its
midpoint falls inside the VAD pause and `pauses.find_pauses` (midpoint rule) skips it. This
script checks, on the C1 development part only (ADR 0014; never the test part, never B1),
whether putting such a word on a side changes pause positions:

- midpoint:  the app's rule (`pauses.find_pauses`), unchanged;
- overlap:   a word whose midpoint is inside the pause goes to the side where more of it lies
  outside the pause, if longer by more than the margin (the B1 `label` rule,
  `evaluation.b1.side_of_pause`). A word that cannot be placed is either skipped, as now
  (`skip`), or makes the position `unknown` (`unknown`).

Uses cached transcripts only (written by run.py); VAD re-runs. Writes
experiments/results/e2/<run>_dev_word_side.csv: per rule, the position table counts, both
ADR 0014 position errors and how many pauses changed word_before.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import sys
from collections.abc import Callable
from pathlib import Path

from presenceai.evalconfig import REPO_ROOT, load_evaluation
from presenceai.evaluation import b1, e2
from presenceai.schemas.assessment import Confidence
from presenceai.schemas.b1 import LabelledPause
from presenceai.schemas.speech import PauseEvent, PausePosition, SpeechRegion, Transcript
from presenceai.settings import SpeechSettings, get_settings
from presenceai.speech import pauses, vad
from presenceai.speech.confidence import band

MARGINS = (0.05, 0.10, 0.20)
FindPauses = Callable[
    [tuple[SpeechRegion, ...], Transcript, SpeechSettings], tuple[PauseEvent, ...]
]

_spec = importlib.util.spec_from_file_location("e2_run", Path(__file__).with_name("run.py"))
if _spec is None or _spec.loader is None:
    raise ImportError("experiments/e2_rate_pauses/run.py not found")
run = importlib.util.module_from_spec(_spec)
sys.modules["e2_run"] = run
_spec.loader.exec_module(run)


def overlap_rule(margin_s: float, ambiguous: str, moved: list[int]) -> FindPauses:
    """`find_pauses` with the overlap rule; counts pauses whose word_before differs into
    `moved`."""

    def find(
        regions: tuple[SpeechRegion, ...], transcript: Transcript, cfg: SpeechSettings
    ) -> tuple[PauseEvent, ...]:
        shortest = min(cfg.pauses.thresholds_s)
        words = transcript.words
        tokens = [b1.Token(raw=w.text, norm=w.text, start_s=w.start_s, end_s=w.end_s)
                  for w in words]  # fmt: skip
        reliable = pauses.punctuation_reliable(transcript, cfg.pauses)
        mine = {(e.start_s, e.end_s): e for e in pauses.find_pauses(regions, transcript, cfg)}
        out: list[PauseEvent] = []
        for a, b in itertools.pairwise(regions):
            if b.start_s - a.end_s < shortest - 1e-9:
                continue
            p = LabelledPause(start_s=a.end_s, end_s=b.start_s, label="?")
            sides = [b1.side_of_pause(t, p, margin_s) for t in tokens]
            before = max((i for i, s in enumerate(sides) if s == "before"), default=None)
            after = min((i for i, s in enumerate(sides) if s == "after"), default=None)
            unplaced = (
                before is not None
                and after is not None
                and any(sides[i] is None for i in range(before + 1, after))
            )
            if mine[(a.end_s, b.start_s)].word_before != before:
                moved[0] += 1
            if (
                not reliable
                or before is None
                or after is None
                or (unplaced and ambiguous == "unknown")
            ):
                position, conf = PausePosition.UNKNOWN, Confidence.LOW
            else:
                position = pauses.position_after(words[before].text, cfg.pauses)
                q = min(words[before].probability, words[after].probability)
                conf = band(cfg.confidence, probability=q, speaking_time_s=None)
            out.append(
                PauseEvent(start_s=a.end_s, end_s=b.start_s, position=position,
                           confidence=conf, word_before=before, word_after=after)
            )  # fmt: skip
        return tuple(out)

    return find


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    paths = run.Paths(args.run)
    settings = get_settings()
    speech, asr_opts = settings.speech, settings.asr
    ev = load_evaluation()
    rules: list[tuple[str, float | None, str, FindPauses | None, list[int]]] = [
        ("midpoint", None, "skip", None, [0])
    ]
    for m in MARGINS:
        for amb in ("skip", "unknown"):
            moved = [0]
            rules.append(("overlap", m, amb, overlap_rule(m, amb, moved), moved))
    keys = [p.value for p in PausePosition]
    out = paths.results_e2 / f"{args.run}_dev_word_side.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "margin_s", "unplaced_word", "pauses", "agree", "unknown",
                    "mid_as_boundary", "boundary_as_mid", "word_before_changed",
                    *[f"{t}->{d}" for t in keys for d in keys]])  # fmt: skip
        for name, margin, amb, find, moved in rules:
            scores, _, _ = run.evaluate(
                paths, "dev", "clean",
                transcribe=run.cached_transcriber(paths, asr_opts.model, asr_opts),
                vad=lambda x: vad.detect_speech(x, speech.vad),
                ev=ev, speech=speech, progress=lambda _: None, find_pauses=find,
            )  # fmt: skip
            s = e2.summarise(scores, ev)
            t = s.position_table
            cells = [t[a][d] for a in keys for d in keys]
            row = [name, "" if margin is None else margin, amb, sum(cells),
                   sum(t[k][k] for k in keys), sum(t[a]["unknown"] for a in keys),
                   f"{s.mid_as_boundary_rate:.4f}", f"{s.boundary_as_mid_rate:.4f}",
                   moved[0], *cells]  # fmt: skip
            w.writerow(row)
            print(*row[:9], sep="\t")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
