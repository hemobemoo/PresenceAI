"""E2 scoring: speech rate, pause detection and pause duration on C1 (ADR 0009, 0011, 0014 §6).

All rules here were fixed before the first E2 run:

- a detected pause matches a true pause if they overlap and their onsets differ by at most
  `match.onset_tolerance_s`; one-to-one, largest overlap first. Only true pauses of at least
  `min_pause_s` take part;
- at each F1 threshold T, a true pause within `f1_exclusion_s` of T counts neither as a hit nor
  as a miss, and a detection matched to it is not a false alarm. F1 without the exclusion and
  the number of excluded true pauses are reported next to it;
- duration error: mean absolute error over all matched pauses;
- wpm error: mean absolute percentage error over **all clips**; a clip without a measured rate
  counts as a 100 % error. The filler / no-filler breakdown is extra information;
- pass/fail against ADR 0009's targets only on the test part; the development part is for
  tuning and gets no verdict.
"""

from __future__ import annotations

from collections.abc import Sequence

from presenceai.evalconfig import E2Settings, EvaluationSettings
from presenceai.schemas.c1 import (
    ClipTruth,
    E2ClipScore,
    E2Summary,
    Part,
    PauseMatch,
    ThresholdScore,
    TruePause,
    f1,
)
from presenceai.schemas.speech import PauseEvent, PausePosition
from presenceai.synth.c1 import rate_band

MID = PausePosition.MID_SENTENCE.value
UNKNOWN = PausePosition.UNKNOWN.value


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def match(
    true: Sequence[TruePause], detected: Sequence[PauseEvent], cfg: E2Settings
) -> list[PauseMatch]:
    pairs: list[tuple[float, int, int]] = []
    for i, t in enumerate(true):
        for j, d in enumerate(detected):
            ov = _overlap(t.start_s, t.end_s, d.start_s, d.end_s)
            if cfg.match.require_overlap and ov <= 0:
                continue
            if abs(d.start_s - t.start_s) > cfg.match.onset_tolerance_s:
                continue
            pairs.append((ov, i, j))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    used_t: set[int] = set()
    used_d: set[int] = set()
    out: list[PauseMatch] = []
    for _, i, j in pairs:
        if i in used_t or j in used_d:
            continue
        used_t.add(i)
        used_d.add(j)
        t, d = true[i], detected[j]
        out.append(
            PauseMatch(
                true_index=i,
                detected_index=j,
                onset_error_s=d.start_s - t.start_s,
                duration_error_s=(d.end_s - d.start_s) - t.duration_s,
            )
        )
    return sorted(out, key=lambda m: m.true_index)


def _counts(
    true_d: Sequence[float],
    det_d: Sequence[float],
    matches: Sequence[PauseMatch],
    threshold: float,
    exclude: set[int],
) -> tuple[int, int, int]:
    tp = fp = fn = 0
    matched_t = {m.true_index: m.detected_index for m in matches}
    matched_d = set(matched_t.values())
    for i, dur in enumerate(true_d):
        if i in exclude:
            continue
        pos_t = dur >= threshold
        if i in matched_t:
            pos_d = det_d[matched_t[i]] >= threshold
            tp += pos_t and pos_d
            fn += pos_t and not pos_d
            fp += pos_d and not pos_t
        else:
            fn += pos_t
    fp += sum(1 for j, dur in enumerate(det_d) if j not in matched_d and dur >= threshold)
    return tp, fp, fn


def threshold_score(
    true: Sequence[TruePause],
    detected: Sequence[PauseEvent],
    matches: Sequence[PauseMatch],
    threshold: float,
    exclusion_s: float,
) -> ThresholdScore:
    true_d = [t.duration_s for t in true]
    det_d = [d.end_s - d.start_s for d in detected]
    zone = {i for i, dur in enumerate(true_d) if abs(dur - threshold) <= exclusion_s + 1e-9}
    tp, fp, fn = _counts(true_d, det_d, matches, threshold, zone)
    tp_a, fp_a, fn_a = _counts(true_d, det_d, matches, threshold, set())
    return ThresholdScore(
        threshold_s=threshold, tp=tp, fp=fp, fn=fn, excluded=len(zone),
        tp_all=tp_a, fp_all=fp_a, fn_all=fn_a,
    )  # fmt: skip


def score_clip(
    truth: ClipTruth,
    detected: Sequence[PauseEvent],
    wpm_measured: float | None,
    part: Part,
    condition: str,
    ev: EvaluationSettings,
) -> E2ClipScore:
    cfg = ev.e2
    true = [p for p in truth.pauses if p.duration_s >= cfg.min_pause_s]
    det = [d for d in detected if d.end_s - d.start_s >= cfg.min_pause_s]
    matches = match(true, det, cfg)
    err = 100.0 if wpm_measured is None else abs(wpm_measured - truth.wpm) / truth.wpm * 100.0
    return E2ClipScore(
        clip_id=truth.clip_id,
        script_id=truth.script_id,
        voice=truth.voice,
        part=part,
        condition=condition,
        has_fillers=truth.fillers > 0,
        rate_band=rate_band(truth.wpm, ev.c1.rate_bands),
        wpm_true=truth.wpm,
        wpm_measured=wpm_measured,
        wpm_error_pct=err,
        true_pauses=len(true),
        detected_pauses=len(det),
        thresholds=tuple(
            threshold_score(true, det, matches, t, cfg.f1_exclusion_s) for t in cfg.thresholds
        ),
        duration_abs_errors_s=tuple(abs(m.duration_error_s) for m in matches),
        positions=tuple(
            (true[m.true_index].position.value, det[m.detected_index].position.value)
            for m in matches
        ),
        true_positions=tuple(p.position.value for p in true),
        detected_positions=tuple(d.position.value for d in det),
        kokoro_pauses=sum(1 for p in true if p.source == "kokoro"),
    )


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _share(xs: Sequence[str], value: str) -> float | None:
    return sum(1 for x in xs if x == value) / len(xs) if xs else None


BOUNDARY = {PausePosition.SENTENCE_BOUNDARY.value, PausePosition.CLAUSE_BOUNDARY.value}


def position_table(pairs: Sequence[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Matched pauses as true position x detected label."""
    labels = [p.value for p in PausePosition]
    table = {t: dict.fromkeys(labels, 0) for t in labels}
    for t, d in pairs:
        table[t][d] += 1
    return table


def boundary_mid_errors(pairs: Sequence[tuple[str, str]]) -> tuple[float | None, float | None]:
    """The headline position measure (ADR 0014 §6): (true mid-sentence pauses labelled as a
    boundary, true boundary pauses labelled mid-sentence), each as a share of the matched true
    pauses of that kind with a known label. Sentence <-> clause swaps are not errors here;
    `unknown` labels are left out and reported apart."""
    mid = [d for t, d in pairs if t == MID and d != UNKNOWN]
    bnd = [d for t, d in pairs if t in BOUNDARY and d != UNKNOWN]
    mid_as_boundary = sum(1 for d in mid if d in BOUNDARY) / len(mid) if mid else None
    boundary_as_mid = sum(1 for d in bnd if d == MID) / len(bnd) if bnd else None
    return mid_as_boundary, boundary_as_mid


def summarise(scores: Sequence[E2ClipScore], ev: EvaluationSettings) -> E2Summary:
    cfg = ev.e2
    parts: set[Part] = {s.part for s in scores}
    conditions = {s.condition for s in scores}
    if len(parts) > 1 or len(conditions) > 1:
        raise ValueError("summarise one part and one condition at a time")
    part: Part = next(iter(parts)) if parts else "test"
    condition = next(iter(conditions)) if conditions else "clean"

    def wpm(sel: Sequence[E2ClipScore]) -> float | None:
        return _mean([s.wpm_error_pct for s in sel if s.wpm_error_pct is not None])

    f1s: dict[float, float | None] = {}
    f1_all: dict[float, float | None] = {}
    excluded: dict[float, int] = {}
    for t in cfg.thresholds:
        ts = [x for s in scores for x in s.thresholds if x.threshold_s == t]
        f1s[t] = f1(sum(x.tp for x in ts), sum(x.fp for x in ts), sum(x.fn for x in ts))
        f1_all[t] = f1(
            sum(x.tp_all for x in ts), sum(x.fp_all for x in ts), sum(x.fn_all for x in ts)
        )
        excluded[t] = sum(x.excluded for x in ts)
    durations = [e for s in scores for e in s.duration_abs_errors_s]
    positions = [p for s in scores for p in s.positions]
    wpm_all = wpm(scores)
    dur = _mean(durations)
    if part == "test":
        passes: dict[str, bool | None] = {
            "wpm": None if wpm_all is None else wpm_all <= cfg.wpm_error_pct_max,
            "duration": None if dur is None else dur <= cfg.duration_error_s_max,
        }
        for key, target in cfg.pause_f1_min.items():
            v = f1s[float(key)]
            passes[f"f1_{key}"] = None if v is None else v >= target
        passed = None if any(v is None for v in passes.values()) else all(passes.values())
    else:
        passes = {}
        passed = None
    return E2Summary(
        part=part,
        condition=condition,
        clips=len(scores),
        wpm_mape_pct=wpm_all,
        wpm_mape_fillers_pct=wpm([s for s in scores if s.has_fillers]),
        wpm_mape_no_fillers_pct=wpm([s for s in scores if not s.has_fillers]),
        f1=f1s,
        f1_all=f1_all,
        excluded=excluded,
        true_pauses=sum(s.true_pauses for s in scores),
        detected_pauses=sum(s.detected_pauses for s in scores),
        matched=len(durations),
        duration_mae_s=dur,
        true_mid_share=_share([p for s in scores for p in s.true_positions], MID),
        detected_mid_share=_share([p for s in scores for p in s.detected_positions], MID),
        position_error_rate=(
            sum(1 for t, d in positions if t != d) / len(positions) if positions else None
        ),
        position_unknown_share=_share([d for _, d in positions], UNKNOWN),
        position_table=position_table(positions),
        mid_as_boundary_rate=boundary_mid_errors(positions)[0],
        boundary_as_mid_rate=boundary_mid_errors(positions)[1],
        kokoro_pauses=sum(s.kokoro_pauses for s in scores),
        passes=passes,
        passed=passed,
    )
