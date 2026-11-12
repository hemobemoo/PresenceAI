"""Spike S1 measurement driver: runs app.py and measures it in Playwright's Chromium.

    # your laptop, real camera + microphone, both transports, 60 s each (window must stay in front)
    uv run python experiments/s1_live_capture/measure.py --devices real --headed --seconds 60

    # landmark load step by step, FastRTC only, one app start per mode
    uv run python experiments/s1_live_capture/measure.py --devices real --headed --seconds 60 \
        --transport fastrtc --landmarks none,pose-lite,pose-full,pose-lite+face

    # no hardware: Chromium fake camera + a generated tone as the microphone
    uv run python experiments/s1_live_capture/measure.py --devices fake --seconds 20

For each transport it opens the tab, starts capture, and records the moment each new meter value
is visible in the page (polled once per animation frame). meter_ms = browser render time minus
the server's publish timestamp; both use this machine's clock. It covers server -> render only,
NOT camera/microphone -> server.

Outputs in --out (default experiments/results/s1 for real devices, a temp dir for fake ones):
    <run>_<transport>_server.csv    one row per second (written by app.py)
    <run>_<transport>_browser.csv   one row per meter value
    <run>_summary.csv               PASS/FAIL per transport (rule in docs/adr/0002)
    <run>_network.csv               every request to a host other than 127.0.0.1
"""

from __future__ import annotations

import argparse
import csv
import math
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

from landmarks import MODES, missing_models
from metrics import (
    LatencySample,
    Summary,
    load_spike_config,
    load_thresholds,
    parse_meter,
    read_second_rows,
    summarise,
    write_csv,
)
from playwright.sync_api import Page, sync_playwright
from sampling import choose_sampling

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
APP = HERE / "app.py"
LIVE_YAML = REPO / "configs" / "live.yaml"
SPIKE_YAML = REPO / "experiments" / "configs" / "s1_live_capture.yaml"
DEFAULT_OUT = REPO / "experiments" / "results" / "s1"
TRANSPORTS = ("fastrtc", "fallback")
LOCAL_PREFIXES = ("http://127.0.0.1", "ws://127.0.0.1", "data:", "blob:", "about:")

# Polls each meter textarea once per animation frame and stores [render_ms, text] on change.
_POLLER = """
(ids) => {
  window.__s1 = window.__s1 || {};
  const last = {};
  for (const id of ids) { window.__s1[id] = []; }
  const tick = () => {
    const now = Date.now();
    for (const id of ids) {
      const el = document.querySelector('#' + id + ' textarea');
      if (el && el.value && el.value !== last[id]) {
        last[id] = el.value;
        window.__s1[id].push([now, el.value]);
      }
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def write_tone(path: Path, seconds: float = 4.0, rate: int = 48000) -> Path:
    """Fake microphone input: 0.5 s of 440 Hz, 0.5 s of silence, repeated (16-bit mono WAV)."""
    frames = bytearray()
    for i in range(int(seconds * rate)):
        on = (i // (rate // 2)) % 2 == 0
        value = int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / rate)) if on else 0
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return path


def wait_for(url: str, timeout_s: float = 120.0, proc: subprocess.Popen[Any] | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"app.py exited early (code {proc.returncode}); see its log")
        try:
            urllib.request.urlopen(url, timeout=1).close()  # noqa: S310  (local http only)
            return
        except OSError:
            time.sleep(0.3)
    raise TimeoutError(f"app did not start at {url}")


def start_capture(page: Page, transport: str) -> None:
    page.get_by_role("tab", name="FastRTC" if transport == "fastrtc" else "Fallback").click()
    page.wait_for_timeout(500)
    if transport == "fastrtc":
        root = page.locator("#s1-fastrtc-stream")
        root.locator("button", has_text="Access Webcam").click()
        root.locator("button[aria-label='start stream']").click()
    else:
        video = page.locator("#s1-fallback-video")
        video.locator("button", has_text="Access Webcam").click()
        video.locator("button[aria-label='capture photo']").click()
        page.locator("#s1-fallback-audio button", has_text="Record").click()


def stop_capture(page: Page, transport: str) -> None:
    if transport == "fastrtc":
        page.locator("#s1-fastrtc-stream button[aria-label='start stream']").click()
    else:
        page.locator("#s1-fallback-audio button", has_text="Stop").click()
        page.locator("#s1-fallback-video button[aria-label='capture photo']").click()


def latency_samples(transport: str, raw: list[list[Any]]) -> list[LatencySample]:
    seen: set[int] = set()
    out: list[LatencySample] = []
    for render_ms, text in raw:
        reading = parse_meter(str(text))
        if reading is None or reading.seq in seen:
            continue
        seen.add(reading.seq)
        out.append(LatencySample(transport, reading.seq, reading.t_server_ms, float(render_ms)))
    return out


def print_summary(s: Summary) -> None:
    def f(v: float | None, spec: str = ".1f") -> str:
        return "n/a" if v is None else format(v, spec)

    print(
        f"[{s.transport}] {s.reason}\n"
        f"  seconds used {s.seconds_used}/{s.seconds_total}"
        f"  fps p50 {f(s.fps_p50)} min {f(s.fps_min)}"
        f"  share fps>={s.min_fps:g}: {f(s.share_fps_ok, '.0%')}\n"
        f"  meter_ms p50 {f(s.meter_ms_p50)} p95 {f(s.meter_ms_p95)} max {f(s.meter_ms_max)}"
        f" (n={s.meter_n})  lag trend {f(s.lag_slope_ms_per_min)} ms/min\n"
        f"  audio chunks/s p50 {f(s.audio_chunk_rate_p50)}  audio real-time ratio"
        f" {f(s.audio_realtime_ratio, '.2f')}  CPU% p50 {f(s.cpu_p50)} max {f(s.cpu_max)}\n"
        f"  landmarks {s.landmarks}: pose found {f(s.pose_success_rate, '.0%')}"
        f"  face found {f(s.face_success_rate, '.0%')}"
        f"  frames dropped {f(s.frames_dropped_share, '.0%')}"
        f"  smallest frame {s.frame_w_min}x{s.frame_h_min}  ({s.settings})"
    )


def parse_modes(text: str) -> list[str]:
    modes = [m.strip() for m in text.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad or not modes:
        raise argparse.ArgumentTypeError(f"--landmarks takes a comma list of {', '.join(MODES)}")
    return modes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--devices", choices=["fake", "real"], required=True)
    p.add_argument("--transport", choices=[*TRANSPORTS, "both"], default="both")
    p.add_argument("--seconds", type=float, default=60.0, help="capture time per transport")
    p.add_argument("--headed", action="store_true", help="show the browser window")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--landmarks",
        type=parse_modes,
        default=["none"],
        help=f"comma list, run one after another: {','.join(MODES)}",
    )
    p.add_argument("--process-every", type=int, default=None, help="override the spike config")
    p.add_argument("--degradation", choices=["config", "browser-default"], default="config")
    args = p.parse_args()

    missing = sorted({p.name for m in args.landmarks for p in missing_models(m, REPO / "models")})
    if missing:
        print(f"missing model files: {', '.join(missing)}")
        print("run `uv run python scripts/download_models.py --pin` first")
        return 2

    run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M%S}_{args.devices}"
    tmp = Path(tempfile.mkdtemp(prefix="s1_"))
    out: Path = args.out or (DEFAULT_OUT if args.devices == "real" else tmp)
    out.mkdir(parents=True, exist_ok=True)
    external: list[tuple[str, str]] = []
    summaries: list[Summary] = []
    for mode in args.landmarks:
        mode_id = run_id if args.landmarks == ["none"] else f"{run_id}_{mode}"
        summaries += run_mode(args, mode, mode_id, out, tmp, external)

    write_csv(out / f"{run_id}_summary.csv", summaries)
    with (out / f"{run_id}_network.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["resource_type", "url"])
        writer.writerows(sorted(set(external)))
    if external:
        hosts = sorted({u.split("/")[2] for _, u in external if "//" in u})
        print(f"external requests from the page: {', '.join(hosts)} (see {run_id}_network.csv)")
    print(f"results in {out}  (app logs: {tmp})")
    return 0


def run_mode(
    args: argparse.Namespace,
    mode: str,
    run_id: str,
    out: Path,
    tmp: Path,
    external: list[tuple[str, str]],
) -> list[Summary]:
    """Start app.py with one landmark mode, measure each transport, return the summaries."""
    transports = TRANSPORTS if args.transport == "both" else (args.transport,)
    cfg = load_spike_config(SPIKE_YAML)
    min_fps, budget_ms = load_thresholds(LIVE_YAML)
    sampling = choose_sampling(
        mode=mode,
        process_every_override=args.process_every,
        stand_in_every=cfg.process_every,
        target_fps=cfg.landmark_target_fps,
        tolerance=cfg.landmark_sampling_tolerance,
    )
    rtp = "browser-default" if args.degradation == "browser-default" else cfg.rtp_params

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    app_cmd = [sys.executable, str(APP), "--port", str(port), "--out", str(out), "--run-id", run_id]
    app_cmd += ["--transport", args.transport, "--landmarks", mode]
    app_cmd += ["--degradation", args.degradation]
    if args.process_every is not None:
        app_cmd += ["--process-every", str(args.process_every)]
    app_log = (tmp / f"app_{mode}.log").open("w", encoding="utf-8")
    app = subprocess.Popen(app_cmd, cwd=REPO, stdout=app_log, stderr=subprocess.STDOUT)  # noqa: S603
    summaries: list[Summary] = []
    print(f"== landmarks={mode}", flush=True)
    try:
        wait_for(url, proc=app)
        chrome_args: list[str] = []
        if args.devices == "fake":
            tone = write_tone(tmp / "tone.wav")
            chrome_args = [
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={tone}",
            ]
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed, args=chrome_args)
            context = browser.new_context(
                permissions=["camera", "microphone"], viewport={"width": 1280, "height": 1000}
            )
            page = context.new_page()
            page.on(
                "request",
                lambda r: (
                    external.append((r.resource_type, r.url))
                    if not r.url.startswith(LOCAL_PREFIXES)
                    else None
                ),
            )
            page.goto(url)
            page.get_by_role(
                "tab"
            ).first.wait_for()  # Gradio keeps a connection open: no networkidle
            page.wait_for_timeout(1000)
            for transport in transports:
                meter_id = f"s1-{transport}-meter"
                page.evaluate(_POLLER, [meter_id])
                print(f"[{transport}] capturing for {args.seconds:g} s ...", flush=True)
                start_capture(page, transport)
                page.wait_for_timeout(int(args.seconds * 1000))
                raw = page.evaluate(f"window.__s1['{meter_id}']")
                stop_capture(page, transport)
                page.wait_for_timeout(1500)  # let app.py flush the last full second

                samples = latency_samples(transport, raw)
                write_csv(out / f"{run_id}_{transport}_browser.csv", samples)
                server_csv = out / f"{run_id}_{transport}_server.csv"
                rows = read_second_rows(server_csv) if server_csv.exists() else []
                summary = summarise(
                    transport,
                    rows,
                    samples,
                    min_fps=min_fps,
                    budget_ms=budget_ms,
                    warmup_s=cfg.warmup_s,
                    min_share=cfg.min_share_fps_ok,
                    settings=(
                        f"run={run_id} {sampling.describe()} "
                        f"rtp={rtp if transport == 'fastrtc' else 'n/a'}"
                    ),
                )
                summaries.append(summary)
                print_summary(summary)
            browser.close()
    finally:
        app.terminate()
        try:
            app.wait(timeout=10)
        except subprocess.TimeoutExpired:
            app.kill()
        app_log.close()
    return summaries


if __name__ == "__main__":
    sys.exit(main())
