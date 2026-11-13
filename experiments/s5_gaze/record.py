"""Spike S5 recorder: one prompted-target gaze session on http://127.0.0.1:7861 (ADR 0004).

    uv run python experiments/s5_gaze/record.py --session A
    uv run python experiments/s5_gaze/record.py --session B
    uv run python experiments/s5_gaze/record.py --session C      # dim light
    uv run python experiments/s5_gaze/record.py --session D      # glasses on

Press Start, then follow the screen and the beeps (one beep: start holding; two beeps: block
over; three beeps: session over). The webcam image is never written: each sampled frame is
turned into gaze features at once, and only those numbers are saved.

Writes:
    data/s5/<run>_<session>_features.csv              per-frame features (git-ignored)
    experiments/results/s5/<run>_<session>_blocks.csv  schedule, order used, actual start times
    experiments/results/s5/features_manifest.csv       SHA-256 and row count of each features file

Spike code: UI strings are inline on purpose; product UI strings live in src/presenceai/lang/.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import argparse
import csv
import hashlib
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import numpy.typing as npt
from fastrtc import WebRTC
from gradio.themes import Default, Font

sys.path.insert(0, str(Path(__file__).resolve().parent))

import protocol
from features import FEATURES, FaceEngine, FrameFeatures

MODELS_DIR = protocol.REPO / "models"
FRAME_FIELDS = [
    "t_s",
    "segment",
    "phase",
    "label",
    "rep",
    "face_found",
    "eyes_closed",
    "process_ms",
    *FEATURES,
]
MANIFEST_FIELDS = ["run_id", "session", "file", "rows", "sha256", "written_at"]
TEXT = {
    "calibration_camera": "Look into the CAMERA LENS",
    "calibration_notes": "Look at your NOTES (desk, right of the laptop)",
    "calibration_neutral": "Look at the screen and relax",
    "at_camera": "CAMERA LENS",
    "at_screen": "the ● DOT on this page",
    "down_notes": "your NOTES (desk, right of the laptop)",
    "away_left": "the wall point on your LEFT",
    "away_right": "the wall point on your RIGHT",
}


def beep(times: int, freq: int = 880) -> None:
    """Laptop-speaker beep on Windows; terminal bell elsewhere."""
    try:
        import winsound

        for _ in range(times):
            winsound.Beep(freq, 140)
            time.sleep(0.06)
    except (ImportError, RuntimeError):
        print("\a" * times, end="", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def append_manifest(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def target_text(seg: protocol.Segment) -> str:
    if seg.phase.startswith("calibration"):
        return TEXT[seg.phase]
    key = f"away_{seg.away_side}" if seg.label == "away" else seg.label
    return TEXT[key]


class Session:
    """Holds the schedule, the feature worker and the output files for one recording."""

    def __init__(self, cfg: dict[str, Any], session: str, run_id: str, data_dir: Path, out: Path):
        self.cfg = cfg
        self.session = session
        self.run_id = run_id
        self.schedule = protocol.build_schedule(cfg, session)
        self.total_s = self.schedule[-1].end_s
        self.features_path = data_dir / f"{run_id}_{session}_features.csv"
        self.blocks_path = out / f"{run_id}_{session}_blocks.csv"
        self.manifest_path = out / "features_manifest.csv"
        self.t0: float | None = None
        self.done = False
        self.rows = 0
        self.unknown = 0
        self.received = 0  # camera frames that reached the server (before and after Start)
        self.actual_start: dict[int, float] = {}
        self._cond = threading.Condition()
        self._pending: tuple[npt.NDArray[np.uint8], float] | None = None
        self._last_take = -1.0
        self._interval = 1.0 / float(cfg["target_fps"])
        self.error: BaseException | None = None
        self.worker_done = threading.Event()
        data_dir.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)

    # ---- capture side (FastRTC handler thread): never blocks
    def offer(self, frame: npt.NDArray[np.uint8]) -> None:
        self.received += 1
        if self.t0 is None or self.done:
            return
        t = time.monotonic() - self.t0
        if t - self._last_take < self._interval * 0.75:
            return
        self._last_take = t
        with self._cond:
            self._pending = (frame, t)  # keep only the latest frame
            self._cond.notify_all()

    # ---- worker thread: features and CSV rows
    def run_worker(self) -> None:
        try:
            self._work()
        except BaseException as exc:  # shown on the page and stops the manifest entry
            self.error = exc
        finally:
            self.worker_done.set()

    def _work(self) -> None:
        engine = FaceEngine(MODELS_DIR, float(self.cfg["eyes_closed_blink"]))
        w, h = self.cfg["process_size"]
        last_ts = 0
        with self.features_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
            writer.writeheader()
            while True:
                with self._cond:
                    while self._pending is None and not self.done:
                        self._cond.wait(0.5)
                    if self._pending is None and self.done:
                        break
                    item, self._pending = self._pending, None
                if item is None:
                    continue
                frame, t = item
                seg = protocol.segment_at(self.schedule, t)
                if seg is None:
                    continue
                p0 = time.perf_counter()
                rows = np.linspace(0, frame.shape[0] - 1, h).astype(np.intp)
                cols = np.linspace(0, frame.shape[1] - 1, w).astype(np.intp)
                small = frame[np.ix_(rows, cols)]
                rgb = np.ascontiguousarray(small[..., ::-1] if self.cfg["frames_bgr"] else small)
                last_ts = max(last_ts + 1, int(t * 1000))
                ff: FrameFeatures = engine(rgb, last_ts)
                ms = round((time.perf_counter() - p0) * 1000, 1)
                writer.writerow(
                    {
                        "t_s": round(t, 3),
                        "segment": seg.index,
                        "phase": seg.phase,
                        "label": seg.label,
                        "rep": seg.rep,
                        "face_found": ff.face_found,
                        "eyes_closed": ff.eyes_closed,
                        "process_ms": ms,
                        **{k: round(v, 6) for k, v in ff.values.items()},
                    }
                )
                self.rows += 1
                self.unknown += ff.unknown
        engine.close()

    # ---- clock thread: beeps and finishing
    def run_clock(self) -> None:
        assert self.t0 is not None  # noqa: S101 (set by start())
        for seg in self.schedule:
            while time.monotonic() - self.t0 < seg.start_s:
                time.sleep(0.005)
            self.actual_start[seg.index] = round(time.monotonic() - self.t0, 3)
            if seg.phase in ("settle", "calibration_camera", "calibration_notes"):
                threading.Thread(target=beep, args=(1,), daemon=True).start()
            elif (
                seg.phase == "cue"
                and seg.index > 0
                and self.schedule[seg.index - 1].phase == "hold"
            ):
                threading.Thread(target=beep, args=(2, 660), daemon=True).start()
        while time.monotonic() - self.t0 < self.total_s:
            time.sleep(0.01)
        self.finish()
        beep(3, 990)

    def start(self) -> None:
        if self.t0 is not None:
            return
        threading.Thread(target=self.run_worker, name="s5-features", daemon=True).start()
        self.t0 = time.monotonic()
        threading.Thread(target=self.run_clock, name="s5-clock", daemon=True).start()

    def finish(self) -> None:
        with self._cond:
            self.done = True
            self._cond.notify_all()
        self.worker_done.wait(30)  # the worker closes the features file
        s = self.cfg["sessions"][self.session]
        meta = {
            "run_id": self.run_id,
            "session": self.session,
            "condition": s["condition"],
            "glasses": s["glasses"],
            "notes_position": self.cfg["notes_position"],
        }
        protocol.write_timing(self.blocks_path, self.schedule, meta, self.actual_start)
        if self.error is not None or not self.features_path.exists():
            print(f"S5 session {self.session} failed; no manifest entry: {self.error}", flush=True)
            return
        append_manifest(
            self.manifest_path,
            {
                "run_id": self.run_id,
                "session": self.session,
                "file": self.features_path.name,
                "rows": self.rows,
                "sha256": sha256(self.features_path),
                "written_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(
            f"S5 {self.run_id} session {self.session}: {self.rows} frames "
            f"({self.unknown} unknown); wrote {self.features_path} and {self.blocks_path.name}",
            flush=True,
        )

    def status(self) -> tuple[str, str]:
        if self.error is not None:
            return "## Error", f"Face Landmarker failed: {self.error}"
        if self.t0 is None:
            return (
                f"## Session {self.session}: press Start when ready",
                f"About {self.total_s / 60:.1f} minutes. Sit as you would present; "
                "follow the text and the beeps. Camera: "
                + (f"{self.received} frames received" if self.received else "not started yet")
                + ".",
            )
        t = time.monotonic() - self.t0
        if self.done or t >= self.total_s:
            return "## Session finished. Thank you.", f"{self.rows} frames saved as features."
        seg = protocol.segment_at(self.schedule, t)
        if seg is None:
            return "## …", ""
        left = seg.end_s - t
        if seg.phase == "cue":
            head = f"## Next: look at {target_text(seg)}"
        elif seg.phase in ("settle", "hold"):
            hold_end = seg.end_s if seg.phase == "hold" else self.schedule[seg.index + 1].end_s
            left = hold_end - t
            head = f"# HOLD: {target_text(seg)}"
        else:
            head = f"# {target_text(seg)}"
        n_blocks = sum(1 for s in self.schedule if s.phase == "hold")
        done_blocks = sum(1 for s in self.schedule if s.phase == "hold" and s.end_s <= t)
        unk = f"{self.unknown / self.rows:.0%}" if self.rows else "–"
        return head, (
            f"{left:4.1f} s left in this step · block {done_blocks}/{n_blocks} · "
            f"{self.rows} frames · unknown {unk}"
        )


def build_demo(sess: Session) -> gr.Blocks:
    cfg = sess.cfg
    theme = Default(font=[Font("system-ui"), Font("sans-serif")])
    echo_w, echo_h = cfg["echo_size"]

    def on_frame(frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        sess.offer(frame)
        rows = np.linspace(0, frame.shape[0] - 1, echo_h).astype(np.intp)
        cols = np.linspace(0, frame.shape[1] - 1, echo_w).astype(np.intp)
        return frame[np.ix_(rows, cols)]

    with gr.Blocks(
        title=f"S5 gaze session {sess.session}", theme=theme, analytics_enabled=False
    ) as demo:
        head = gr.Markdown(sess.status()[0], elem_id="s5-head")
        info = gr.Markdown(sess.status()[1], elem_id="s5-info")
        gr.Markdown("<div style='text-align:center;font-size:64px'>●</div>")
        with gr.Row():
            start = gr.Button("Start", variant="primary", elem_id="s5-start")
        with gr.Accordion("Camera (not recorded)", open=True):
            rtc = WebRTC(
                label="Camera",
                modality="video",
                mode="send-receive",
                rtc_configuration=None,
                track_constraints=cfg["track_constraints"],
                height=180,
                full_screen=False,  # as in S1: a full-screen video covers the Start button
                elem_id="s5-camera",
            )
            rtc.stream(on_frame, inputs=[rtc], outputs=[rtc], time_limit=int(sess.total_s) + 300)
        timer = gr.Timer(0.25)
        # Both are instant calls; outside the queue so the video stream never holds them back.
        timer.tick(lambda: sess.status(), outputs=[head, info], show_progress="hidden", queue=False)
        start.click(lambda: sess.start(), outputs=None, queue=False)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--session", choices=protocol.SESSIONS, required=True)
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--config", type=Path, default=protocol.CONFIG)
    ap.add_argument("--data-dir", type=Path, default=protocol.DATA_DIR)
    ap.add_argument("--out", type=Path, default=protocol.RESULTS)
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()
    if not (MODELS_DIR / "face_landmarker.task").is_file():
        raise SystemExit(
            "missing face_landmarker.task: run `uv run python scripts/download_models.py --pin`"
        )
    cfg = protocol.load_config(args.config)
    sess = Session(cfg, args.session, args.run_id, args.data_dir, args.out)
    s = cfg["sessions"][args.session]
    print(
        f"S5 run {args.run_id} session {args.session} ({s['condition']}, glasses={s['glasses']}), "
        f"{sess.total_s:.0f} s; open http://127.0.0.1:{args.port}",
        flush=True,
    )
    build_demo(sess).queue().launch(
        server_name="127.0.0.1", server_port=args.port, share=False, show_api=False
    )


if __name__ == "__main__":
    main()
