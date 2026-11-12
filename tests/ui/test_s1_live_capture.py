"""End-to-end check of the S1 spike with Chromium's fake camera and microphone.

Runs experiments/s1_live_capture/measure.py (which starts app.py) for both transports and
checks that frames, audio chunks and meter updates flowed and every CSV was written. It does
not assert the S1 pass thresholds: those apply to real devices on the target laptop.

    uv run pytest -m needs_browser
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MEASURE = REPO / "experiments" / "s1_live_capture" / "measure.py"
TRANSPORTS = ("fastrtc", "fallback")

pytestmark = [pytest.mark.slow, pytest.mark.needs_browser]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("s1")
    cmd = [sys.executable, str(MEASURE), "--devices", "fake", "--seconds", "8"]
    cmd += ["--out", str(out), "--run-id", "e2e"]
    result = subprocess.run(  # noqa: S603  (our own script, fixed arguments)
        cmd, cwd=REPO, capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return out


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_server_counters_flowed(run_dir: Path, transport: str):
    rows = _rows(run_dir / f"e2e_{transport}_server.csv")
    assert len(rows) >= 5
    assert sum(int(r["video_frames_processed"]) for r in rows) > 0
    assert sum(int(r["audio_chunks"]) for r in rows) > 0
    assert sum(int(r["meter_updates"]) for r in rows) > 0
    assert {r["warmup"] for r in rows} == {"True", "False"}
    assert all(r["cpu_percent"] != "" for r in rows)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_browser_saw_meter_updates(run_dir: Path, transport: str):
    rows = _rows(run_dir / f"e2e_{transport}_browser.csv")
    assert len(rows) >= 10
    assert all(float(r["meter_ms"]) >= 0 for r in rows)  # same clock: render after publish


def test_summary_has_both_transports(run_dir: Path):
    rows = _rows(run_dir / "e2e_summary.csv")
    assert [r["transport"] for r in rows] == list(TRANSPORTS)
    for r in rows:
        assert r["meter_ms_p95"] != ""
        assert r["passed"] in {"True", "False"}


def test_network_log_written(run_dir: Path):
    rows = _rows(run_dir / "e2e_network.csv")
    hosts = {r["url"].split("/")[2] for r in rows}
    assert "127.0.0.1" not in {h.split(":")[0] for h in hosts}
