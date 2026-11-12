"""Spike S1 metrics: per-second counters, meter latency samples, CSV I/O and the PASS rule.

Pure Python + PyYAML so it can be unit-tested without a browser, camera or microphone.
The PASS rule (docs/adr/0002-live-capture-transport.md):

    processed fps >= min_fps in >= min_share of post-warm-up seconds
    AND meter_ms p95 < latency budget            (max is reported, not used)

min_fps and the budget come from configs/live.yaml; warm-up and share from
experiments/configs/s1_live_capture.yaml.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------- small maths


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 1]); None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def slope_per_min(ts_ms: Sequence[float], ys: Sequence[float]) -> float | None:
    """Least-squares slope of ys against time, in y-units per minute."""
    n = len(ts_ms)
    if n < 2:
        return None
    mx = sum(ts_ms) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in ts_ms)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(ts_ms, ys, strict=True))
    return sxy / sxx * 60_000.0


# ---------------------------------------------------------------- meter text

_METER = re.compile(r"level=(?P<level>[0-9.]+) seq=(?P<seq>\d+) t_server_ms=(?P<t>[0-9.]+)")


@dataclass(frozen=True)
class MeterReading:
    level: float
    seq: int
    t_server_ms: float


def format_meter(level: float, seq: int, t_server_ms: float) -> str:
    return f"level={level:.4f} seq={seq} t_server_ms={t_server_ms:.3f}"


def parse_meter(text: str) -> MeterReading | None:
    match = _METER.search(text or "")
    if match is None:
        return None
    try:
        return MeterReading(float(match["level"]), int(match["seq"]), float(match["t"]))
    except ValueError:
        return None


def now_ms() -> float:
    """Wall-clock ms; the server and the browser on the same machine share this clock."""
    return time.time() * 1000.0


# ---------------------------------------------------------------- per-second rows


@dataclass(frozen=True)
class SecondRow:
    run_id: str
    transport: str
    t_s: int
    warmup: bool
    video_frames_received: int
    video_frames_processed: int
    audio_chunks: int
    audio_seconds: float
    meter_updates: int
    process_ms_p50: float | None
    process_ms_max: float | None
    cpu_percent: float
    # Landmark follow-up columns (defaults keep the first S1 CSVs readable).
    landmarks: str = "none"
    frames_dropped: int = 0  # received frames the landmark worker skipped because it was busy
    pose_checked: int = 0
    pose_found: int = 0
    face_checked: int = 0
    face_found: int = 0
    frame_w_min: int | None = None  # smallest received frame size in the second
    frame_h_min: int | None = None


SECOND_FIELDS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(SecondRow))


@dataclass
class _Bucket:
    video_received: int = 0
    video_processed: int = 0
    audio_chunks: int = 0
    audio_seconds: float = 0.0
    meter_updates: int = 0
    process_ms: list[float] = field(default_factory=list[float])
    frames_dropped: int = 0
    pose_checked: int = 0
    pose_found: int = 0
    face_checked: int = 0
    face_found: int = 0
    w_min: int | None = None
    h_min: int | None = None


class RunRecorder:
    """Thread-safe per-second counters for one transport run.

    The run starts at the first event, so browser start-up time is not counted.
    `flush()` returns (and sends to `sink`) one row per completed second, including
    idle seconds with zero counts.
    """

    def __init__(
        self,
        run_id: str,
        transport: str,
        *,
        warmup_s: float,
        clock: Callable[[], float] = time.monotonic,
        cpu: Callable[[], float],
        sink: Callable[[SecondRow], None] | None = None,
        landmarks: str = "none",
    ) -> None:
        self.run_id = run_id
        self.transport = transport
        self.landmarks = landmarks
        self.warmup_s = warmup_s
        self._clock = clock
        self._cpu = cpu
        self._sink = sink
        self._lock = threading.Lock()
        self._start: float | None = None
        self._last = 0.0
        self._next_second = 0
        self._buckets: dict[int, _Bucket] = {}

    def _bucket(self, *, media: bool = True) -> _Bucket:
        now = self._clock()
        if self._start is None:
            self._start = now
        if media:
            self._last = now
        index = int(now - self._start)
        return self._buckets.setdefault(index, _Bucket())

    def video_frame(
        self,
        *,
        processed: bool,
        process_ms: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        with self._lock:
            b = self._bucket()
            b.video_received += 1
            if width is not None:
                b.w_min = width if b.w_min is None else min(b.w_min, width)
            if height is not None:
                b.h_min = height if b.h_min is None else min(b.h_min, height)
            if processed:
                b.video_processed += 1
                if process_ms is not None:
                    b.process_ms.append(process_ms)

    def frame_processed(
        self,
        *,
        process_ms: float,
        pose_found: bool | None = None,
        face_found: bool | None = None,
    ) -> None:
        """A frame finished landmark processing; None = that model is not running."""
        with self._lock:
            b = self._bucket()
            b.video_processed += 1
            b.process_ms.append(process_ms)
            if pose_found is not None:
                b.pose_checked += 1
                b.pose_found += int(pose_found)
            if face_found is not None:
                b.face_checked += 1
                b.face_found += int(face_found)

    def frame_dropped(self) -> None:
        with self._lock:
            self._bucket().frames_dropped += 1

    def audio_chunk(self, seconds: float) -> None:
        with self._lock:
            b = self._bucket()
            b.audio_chunks += 1
            b.audio_seconds += seconds

    def meter_published(self) -> None:
        with self._lock:
            self._bucket(media=False).meter_updates += 1

    def flush(self) -> list[SecondRow]:
        with self._lock:
            if self._start is None:
                return []
            # Complete seconds before the one holding the latest audio/video event. Gaps inside
            # the run are kept as zero rows; the second cut short by Stop and the idle time
            # after it are not reported (it would look like a stall that never happened).
            complete = min(int(self._clock() - self._start), int(self._last - self._start))
            if complete <= self._next_second:
                return []
            cpu = self._cpu()
            rows: list[SecondRow] = []
            for t in range(self._next_second, complete):
                b = self._buckets.pop(t, _Bucket())
                rows.append(
                    SecondRow(
                        run_id=self.run_id,
                        transport=self.transport,
                        t_s=t,
                        warmup=t < self.warmup_s,
                        video_frames_received=b.video_received,
                        video_frames_processed=b.video_processed,
                        audio_chunks=b.audio_chunks,
                        audio_seconds=round(b.audio_seconds, 6),
                        meter_updates=b.meter_updates,
                        process_ms_p50=percentile(b.process_ms, 0.5),
                        process_ms_max=max(b.process_ms) if b.process_ms else None,
                        cpu_percent=cpu,
                        landmarks=self.landmarks,
                        frames_dropped=b.frames_dropped,
                        pose_checked=b.pose_checked,
                        pose_found=b.pose_found,
                        face_checked=b.face_checked,
                        face_found=b.face_found,
                        frame_w_min=b.w_min,
                        frame_h_min=b.h_min,
                    )
                )
            self._next_second = complete
        if self._sink is not None:
            for row in rows:
                self._sink(row)
        return rows


# ---------------------------------------------------------------- latency samples


@dataclass(frozen=True)
class LatencySample:
    """One meter value: server publish time and browser render time (same machine clock)."""

    transport: str
    seq: int
    t_server_ms: float
    t_render_ms: float

    @property
    def meter_ms(self) -> float:
        return self.t_render_ms - self.t_server_ms


LATENCY_FIELDS: tuple[str, ...] = (*(f.name for f in dataclasses.fields(LatencySample)), "meter_ms")


# ---------------------------------------------------------------- CSV


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(round(value, 3))
    return str(value)


def write_csv(path: Path, rows: Iterable[Any], *, append: bool = False) -> None:
    """Write dataclass rows (SecondRow, LatencySample, Summary) with a header."""
    rows = list(rows)
    if not rows:
        return
    first = rows[0]
    names = [f.name for f in dataclasses.fields(first)]
    extra = ["meter_ms"] if isinstance(first, LatencySample) else []
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not (append and path.exists() and path.stat().st_size > 0)
    with path.open("a" if append else "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow([*names, *extra])
        for row in rows:
            values = [getattr(row, n) for n in names]
            if extra:
                values.append(row.meter_ms)
            writer.writerow([_csv_value(v) for v in values])


def _parse(text: str, kind: Any) -> Any:
    if text == "":
        return None
    if kind is bool or kind == "bool":
        return text == "True"
    if kind in (int, "int"):
        return int(text)
    if kind in (str, "str"):
        return text
    return float(text)


def read_second_rows(path: Path) -> list[SecondRow]:
    kinds = {f.name: str(f.type).split(" ")[0] for f in dataclasses.fields(SecondRow)}
    with path.open(encoding="utf-8", newline="") as fh:
        return [
            SecondRow(**{k: _parse(v, kinds[k]) for k, v in rec.items()})
            for rec in csv.DictReader(fh)
        ]


def read_latency_samples(path: Path) -> list[LatencySample]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [
            LatencySample(
                rec["transport"],
                int(rec["seq"]),
                float(rec["t_server_ms"]),
                float(rec["t_render_ms"]),
            )
            for rec in csv.DictReader(fh)
        ]


# ---------------------------------------------------------------- summary and PASS rule


@dataclass(frozen=True)
class Summary:
    transport: str
    seconds_total: int
    seconds_used: int
    fps_p50: float | None
    fps_min: float | None
    share_fps_ok: float | None
    audio_chunk_rate_p50: float | None
    audio_realtime_ratio: float | None
    meter_n: int
    meter_ms_p50: float | None
    meter_ms_p95: float | None
    meter_ms_max: float | None
    lag_slope_ms_per_min: float | None
    cpu_p50: float | None
    cpu_max: float | None
    min_fps: float
    budget_ms: float
    passed: bool
    reason: str
    # Landmark follow-up: reported, not part of PASS.
    landmarks: str = "none"
    pose_success_rate: float | None = None  # share of processed frames with a pose found
    face_success_rate: float | None = None
    frames_dropped_share: float | None = None  # dropped / received frames
    frame_w_min: int | None = None
    frame_h_min: int | None = None
    settings: str = ""


def _rate(found: int, checked: int) -> float | None:
    return found / checked if checked else None


def _min_or_none(values: Iterable[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def summarise(
    transport: str,
    rows: Sequence[SecondRow],
    latencies: Sequence[LatencySample],
    *,
    min_fps: float,
    budget_ms: float,
    warmup_s: float,
    min_share: float,
    settings: str = "",
) -> Summary:
    used = [r for r in rows if not r.warmup]
    received = sum(r.video_frames_received for r in used)
    fps = [float(r.video_frames_processed) for r in used]
    share = sum(f >= min_fps for f in fps) / len(fps) if fps else None
    audio_secs = sum(r.audio_seconds for r in used)

    lat_used: list[LatencySample] = []
    if latencies:
        t0 = min(s.t_server_ms for s in latencies)
        lat_used = [s for s in latencies if s.t_server_ms - t0 >= warmup_s * 1000.0]
    meter = [s.meter_ms for s in lat_used]
    p95 = percentile(meter, 0.95)
    cpu = [r.cpu_percent for r in used]

    problems: list[str] = []
    if not used:
        problems.append("no post-warm-up seconds")
    elif share is not None and share < min_share:
        problems.append(f"fps >= {min_fps:g} in {share:.0%} of seconds (< {min_share:.0%})")
    if p95 is None:
        problems.append("no meter latency samples")
    elif p95 >= budget_ms:
        problems.append(f"meter_ms p95 {p95:.0f} >= {budget_ms:g}")

    return Summary(
        transport=transport,
        seconds_total=len(rows),
        seconds_used=len(used),
        fps_p50=percentile(fps, 0.5),
        fps_min=min(fps) if fps else None,
        share_fps_ok=share,
        audio_chunk_rate_p50=percentile([float(r.audio_chunks) for r in used], 0.5),
        audio_realtime_ratio=audio_secs / len(used) if used else None,
        meter_n=len(meter),
        meter_ms_p50=percentile(meter, 0.5),
        meter_ms_p95=p95,
        meter_ms_max=max(meter) if meter else None,
        lag_slope_ms_per_min=slope_per_min([s.t_server_ms for s in lat_used], meter),
        cpu_p50=percentile(cpu, 0.5),
        cpu_max=max(cpu) if cpu else None,
        min_fps=min_fps,
        budget_ms=budget_ms,
        passed=not problems,
        reason="; ".join(problems) if problems else "PASS",
        landmarks=rows[0].landmarks if rows else "none",
        pose_success_rate=_rate(sum(r.pose_found for r in used), sum(r.pose_checked for r in used)),
        face_success_rate=_rate(sum(r.face_found for r in used), sum(r.face_checked for r in used)),
        frames_dropped_share=sum(r.frames_dropped for r in used) / received if received else None,
        frame_w_min=_min_or_none(r.frame_w_min for r in used),
        frame_h_min=_min_or_none(r.frame_h_min for r in used),
        settings=settings,
    )


# ---------------------------------------------------------------- config


@dataclass(frozen=True)
class SpikeConfig:
    warmup_s: float
    min_share_fps_ok: float
    meter_interval_ms: float
    process_every: int
    process_size: tuple[int, int]
    echo_size: tuple[int, int]
    fallback_stream_every_s: float
    time_limit_s: float
    track_constraints: dict[str, Any]
    landmark_target_fps: float  # time-based sampling, used when --landmarks is not none
    landmark_sampling_tolerance: float  # fraction of the interval a frame may be early
    rtp_params: dict[str, Any]  # FastRTC sender parameters (degradationPreference)


def _yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data  # pyright: ignore[reportUnknownVariableType]


def load_thresholds(live_yaml: Path) -> tuple[float, float]:
    """(min_fps, latency_budget_ms) from configs/live.yaml."""
    data = _yaml(live_yaml)
    return float(data["video"]["min_fps"]), float(data["latency_budget_ms"])


def load_spike_config(path: Path) -> SpikeConfig:
    d = _yaml(path)
    return SpikeConfig(
        warmup_s=float(d["warmup_s"]),
        min_share_fps_ok=float(d["min_share_fps_ok"]),
        meter_interval_ms=float(d["meter_interval_ms"]),
        process_every=int(d["process_every"]),
        process_size=(int(d["process_size"][0]), int(d["process_size"][1])),
        echo_size=(int(d["echo_size"][0]), int(d["echo_size"][1])),
        fallback_stream_every_s=float(d["fallback_stream_every_s"]),
        time_limit_s=float(d["time_limit_s"]),
        track_constraints=dict(d["track_constraints"]),
        landmark_target_fps=float(d["landmark_target_fps"]),
        landmark_sampling_tolerance=float(d["landmark_sampling_tolerance"]),
        rtp_params=dict(d.get("rtp_params") or {}),
    )
