"""Spike S1: live mic + webcam capture page (FastRTC and Gradio fallback).

    uv run python experiments/s1_live_capture/app.py                 # both transports, port 7861
    uv run python experiments/s1_live_capture/app.py --transport fastrtc --landmarks pose-lite

Shows a live audio level meter and a frame counter per transport and appends one row per
second to experiments/results/s1/<run-id>_<transport>_server.csv. Only counters are written:
no audio, no frames, nothing leaves 127.0.0.1 (no TURN/STUN, no analytics, no share link).

--landmarks none|pose-lite|pose-full|pose-lite+face picks what runs on sampled frames (none =
the first runs' grey-mean stand-in). Landmarks run on a worker thread that keeps only the latest
frame (landmarks.py); only found/not-found counts are recorded.

Spike code: UI strings are inline on purpose; product UI strings live in src/presenceai/lang/.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import argparse
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import numpy.typing as npt
import psutil
from fastrtc import AdditionalOutputs, AudioVideoStreamHandler, WebRTC
from gradio.themes import Default, Font
from landmarks import MODES, LandmarkWorker, build_engine, missing_models, resize_nearest
from metrics import (
    RunRecorder,
    SecondRow,
    SpikeConfig,
    format_meter,
    load_spike_config,
    now_ms,
    write_csv,
)
from sampling import SamplingSpec, choose_sampling

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "experiments" / "configs" / "s1_live_capture.yaml"
DEFAULT_OUT = REPO / "experiments" / "results" / "s1"
MODELS_DIR = REPO / "models"
TRANSPORTS = ("fastrtc", "fallback")
DEGRADATION = ("config", "browser-default")


class CpuSampler:
    """Process CPU% sampled once per second on its own thread (can exceed 100 on multi-core).

    psutil reports usage since the previous call, so a single sampler is shared by all
    recorders instead of each recorder calling psutil and resetting the interval.
    """

    def __init__(self, period_s: float = 1.0) -> None:
        self._process = psutil.Process()
        self._process.cpu_percent(None)
        self.value = 0.0
        self._period_s = period_s
        threading.Thread(target=self._loop, name="s1-cpu", daemon=True).start()

    def _loop(self) -> None:
        while True:
            time.sleep(self._period_s)
            self.value = float(self._process.cpu_percent(None))


def rms_level(samples: npt.NDArray[Any]) -> float:
    """RMS level in [0, 1] for int16 or float audio of any shape."""
    if samples.size == 0:
        return 0.0
    x = samples.astype(np.float32)
    if np.issubdtype(samples.dtype, np.integer):
        x /= 32768.0
    return float(np.sqrt(np.mean(np.square(x))))


def n_samples(samples: npt.NDArray[Any]) -> int:
    """Samples per channel for (n,), (n, ch) or (ch, n) arrays."""
    if samples.ndim == 1:
        return samples.shape[0]
    return max(samples.shape)


@dataclass
class Shared:
    cfg: SpikeConfig
    recorders: dict[str, RunRecorder]
    workers: dict[str, LandmarkWorker]
    sampling: SamplingSpec
    rtp_params: dict[str, Any] | None


class S1Handler(AudioVideoStreamHandler):
    """FastRTC audio-video handler (send-receive is the only audio-video mode)."""

    def __init__(self, shared: Shared) -> None:
        super().__init__(expected_layout="mono", input_sample_rate=48000)
        self.shared = shared
        self.rec = shared.recorders["fastrtc"]
        self.worker = shared.workers["fastrtc"]
        self.frames = shared.sampling.make()
        w, h = shared.cfg.echo_size
        self.echo: npt.NDArray[np.uint8] = np.zeros((h, w, 3), dtype=np.uint8)
        self.level = 0.0
        self.seq = 0
        self.interval_s = shared.cfg.meter_interval_ms / 1000.0
        self.next_emit = time.perf_counter()

    def receive(self, frame: tuple[int, npt.NDArray[np.int16]]) -> None:
        sample_rate, samples = frame
        self.level = rms_level(samples)
        self.rec.audio_chunk(n_samples(samples) / sample_rate)

    def emit(self) -> Any:
        # Block until the next meter slot: FastRTC calls emit() in a loop.
        delay = self.next_emit - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        self.next_emit = max(self.next_emit + self.interval_s, time.perf_counter())
        self.seq += 1
        self.rec.meter_published()
        meter = format_meter(self.level, self.seq, now_ms())
        return AdditionalOutputs(meter, f"frames={self.frames.count}")

    # FastRTC annotates VideoFrame but passes frame.to_ndarray(format="bgr24") (fastrtc/tracks.py).
    def video_receive(self, frame: npt.NDArray[np.uint8]) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.rec.video_frame(processed=False, width=frame.shape[1], height=frame.shape[0])
        if self.frames.take(time.monotonic()):
            self.worker.submit(frame)  # never blocks: the worker keeps only the latest frame
        self.echo = resize_nearest(frame, self.shared.cfg.echo_size)

    def video_emit(self) -> npt.NDArray[np.uint8]:
        return self.echo

    def copy(self) -> S1Handler:
        return S1Handler(self.shared)


def build_demo(shared: Shared, transports: tuple[str, ...]) -> gr.Blocks:
    cfg = shared.cfg
    # Plain Font objects: no Google Fonts fetch and no local font files to 404 on.
    theme = Default(
        font=[Font("system-ui"), Font("sans-serif")],
        font_mono=[Font("ui-monospace"), Font("monospace")],
    )
    with gr.Blocks(title="S1 live capture spike", theme=theme, analytics_enabled=False) as demo:
        gr.Markdown("## S1 live capture spike\nCounters only. Nothing is recorded or uploaded.")
        with gr.Tabs():
            if "fastrtc" in transports:
                with gr.Tab("FastRTC", id="fastrtc", elem_id="s1-tab-fastrtc"):
                    rtc = WebRTC(
                        label="Camera + microphone (FastRTC)",
                        modality="audio-video",
                        mode="send-receive",
                        rtc_configuration=None,
                        track_constraints=cfg.track_constraints,
                        rtp_params=shared.rtp_params,
                        height=320,
                        full_screen=False,
                        elem_id="s1-fastrtc-stream",
                    )
                    meter = gr.Textbox(label="Meter", elem_id="s1-fastrtc-meter", interactive=False)
                    frames = gr.Textbox(
                        label="Frames", elem_id="s1-fastrtc-frames", interactive=False
                    )
                    rtc.stream(
                        S1Handler(shared),
                        inputs=[rtc],
                        outputs=[rtc],
                        time_limit=int(cfg.time_limit_s),
                    )
                    rtc.on_additional_outputs(
                        lambda m, f: (m, f),
                        outputs=[meter, frames],
                        show_progress="hidden",
                    )
            if "fallback" in transports:
                with gr.Tab("Fallback", id="fallback", elem_id="s1-tab-fallback"):
                    rec = shared.recorders["fallback"]
                    fb_worker = shared.workers["fallback"]
                    sampler = shared.sampling.make()
                    seq = {"n": 0}
                    mic = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="Microphone (gr.Audio streaming)",
                        elem_id="s1-fallback-audio",
                    )
                    cam = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label="Webcam (gr.Image streaming)",
                        elem_id="s1-fallback-video",
                    )
                    fb_meter = gr.Textbox(
                        label="Meter", elem_id="s1-fallback-meter", interactive=False
                    )
                    fb_frames = gr.Textbox(
                        label="Frames", elem_id="s1-fallback-frames", interactive=False
                    )

                    def on_audio(chunk: tuple[int, npt.NDArray[Any]] | None) -> str:
                        if chunk is None:
                            return ""
                        sample_rate, samples = chunk
                        rec.audio_chunk(n_samples(samples) / sample_rate)
                        seq["n"] += 1
                        rec.meter_published()
                        return format_meter(rms_level(samples), seq["n"], now_ms())

                    def on_frame(frame: npt.NDArray[Any] | None) -> str:
                        if frame is None:
                            return ""
                        rec.video_frame(
                            processed=False, width=frame.shape[1], height=frame.shape[0]
                        )
                        if sampler.take(time.monotonic()):
                            fb_worker.submit(frame)
                        return f"frames={sampler.count}"

                    mic.stream(
                        on_audio,
                        inputs=[mic],
                        outputs=[fb_meter],
                        stream_every=cfg.fallback_stream_every_s,
                        time_limit=int(cfg.time_limit_s),
                        show_progress="hidden",
                    )
                    cam.stream(
                        on_frame,
                        inputs=[cam],
                        outputs=[fb_frames],
                        stream_every=cfg.fallback_stream_every_s,
                        time_limit=int(cfg.time_limit_s),
                        show_progress="hidden",
                    )
    return demo


def start_flusher(
    recorders: dict[str, RunRecorder], workers: dict[str, LandmarkWorker], period_s: float = 0.25
) -> None:
    reported: set[str] = set()

    def loop() -> None:
        while True:
            time.sleep(period_s)
            for r in recorders.values():
                r.flush()
            for name, w in workers.items():
                if w.error is not None and name not in reported:
                    reported.add(name)
                    print(f"landmark worker for {name} failed: {w.error}", flush=True)

    threading.Thread(target=loop, name="s1-flusher", daemon=True).start()


def make_recorders(
    run_id: str, out: Path, transports: tuple[str, ...], warmup_s: float, landmarks: str
) -> dict[str, RunRecorder]:
    recorders: dict[str, RunRecorder] = {}
    sampler = CpuSampler()
    for t in TRANSPORTS:
        path = out / f"{run_id}_{t}_server.csv"

        def sink(row: SecondRow, path: Path = path) -> None:
            write_csv(path, [row], append=True)

        recorders[t] = RunRecorder(
            run_id,
            t,
            warmup_s=warmup_s,
            cpu=lambda: sampler.value,
            sink=sink if t in transports else None,
            landmarks=landmarks,
        )
    return recorders


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--transport", choices=[*TRANSPORTS, "both"], default="both")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--landmarks", choices=MODES, default="none")
    parser.add_argument(
        "--process-every",
        type=int,
        default=None,
        help="send every Nth frame to the landmark step (default: time-based, spike config)",
    )
    parser.add_argument(
        "--degradation",
        choices=DEGRADATION,
        default="config",
        help="FastRTC rtp_params from the spike config, or the browser's default behaviour",
    )
    args = parser.parse_args()

    transports = TRANSPORTS if args.transport == "both" else (args.transport,)
    cfg = load_spike_config(args.config)
    missing = missing_models(args.landmarks, MODELS_DIR)
    if missing:
        names = ", ".join(p.name for p in missing)
        raise SystemExit(f"missing {names}: run `uv run python scripts/download_models.py --pin`")
    sampling = choose_sampling(
        mode=args.landmarks,
        process_every_override=args.process_every,
        stand_in_every=cfg.process_every,
        target_fps=cfg.landmark_target_fps,
        tolerance=cfg.landmark_sampling_tolerance,
    )
    rtp = cfg.rtp_params if args.degradation == "config" and cfg.rtp_params else None

    recorders = make_recorders(args.run_id, args.out, transports, cfg.warmup_s, args.landmarks)
    mode = args.landmarks
    workers = {
        t: LandmarkWorker(
            lambda: build_engine(mode, MODELS_DIR),
            recorders[t],
            process_size=cfg.process_size,
            bgr=(t == "fastrtc"),  # FastRTC hands BGR arrays; gr.Image hands RGB
        )
        for t in TRANSPORTS
    }
    for w in workers.values():
        w.start()
    start_flusher(recorders, workers)
    print(
        f"S1 run {args.run_id}: landmarks={mode} {sampling.describe()} rtp_params={rtp}; "
        f"writing per-second rows to {args.out}",
        flush=True,
    )
    demo = build_demo(Shared(cfg, recorders, workers, sampling, rtp), transports)
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port, share=False, show_api=False)


if __name__ == "__main__":
    main()
