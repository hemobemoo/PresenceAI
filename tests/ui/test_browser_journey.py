"""A full S2 session in a real browser, and every request it makes (owner, 2026-11-25).

Page load, consent, record (a tone from a fake microphone the test can mute), explanation, a
silent first answer, the retry, both answers, stop, analyse. Every request the browser makes is
logged; the test fails on any host other than this computer, and on any Content-Security-Policy
violation (something tried to load from outside and was blocked).

Runs the real app in-process on a free loopback port with a fake analyser (no models) and short
answer windows. Marked slow + needs_browser: Playwright Chromium, local only.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from presenceai.pipeline import app_run
from presenceai.schemas.speech import AsrQuality, SpeechRegion, SpeechReport
from presenceai.settings import AppSettings, Settings

pytestmark = [pytest.mark.slow, pytest.mark.needs_browser]

WINDOW_S = 6.0
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
FAKE_MIC = """
(() => {
  navigator.mediaDevices.getUserMedia = async (c) => {
    const ctx = new AudioContext({ sampleRate: 48000 });
    const osc = ctx.createOscillator(); osc.frequency.value = 220;
    const gain = ctx.createGain(); gain.gain.value = 0.3;
    const dest = ctx.createMediaStreamDestination();
    osc.connect(gain); gain.connect(dest); osc.start();
    window.__mic = gain;
    return dest.stream;
  };
  navigator.mediaDevices.enumerateDevices = async () =>
    [{ deviceId: 'fake', kind: 'audioinput', label: 'test tone', groupId: 'g' }];
})();
"""


def energy_vad(samples: np.ndarray, opts: Any = None) -> tuple[SpeechRegion, ...]:
    """Stand-in for Silero: 100 ms blocks above a small energy count as speech."""
    rate, block = 16_000, 1600
    regions: list[SpeechRegion] = []
    start = None
    for i in range(0, len(samples) + block, block):
        loud = i < len(samples) and float(np.abs(samples[i : i + block]).mean()) > 0.01
        if loud and start is None:
            start = i
        elif not loud and start is not None:
            regions.append(SpeechRegion(start_s=start / rate, end_s=min(i, len(samples)) / rate))
            start = None
    return tuple(regions)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def app_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sb: Any, fake_assessment: Any,
            fake_parser: Any) -> Any:  # fmt: skip
    import uvicorn

    from presenceai.speech import vad
    from presenceai.ui.app import create_app
    from presenceai.ui.context import AppContext

    monkeypatch.setattr(vad, "detect_speech", energy_vad)

    def answer(text: str) -> SpeechReport:
        t = sb.spaced(text, language="en", duration_s=WINDOW_S * 3)
        return SpeechReport(
            source_name="a.wav", model="whisper_small", transcript=t,
            asr_quality=AsrQuality(word_count=len(t.words)),
            speech_regions=(SpeechRegion(start_s=0.5, end_s=t.words[-1].end_s),),
        )  # fmt: skip

    def analyze(audio: Any, **_: Any) -> SpeechReport:
        if audio.name == "explanation.wav":
            return fake_assessment.speech
        return answer("we checked it on our own recordings")

    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    settings = settings.model_copy(update={"app": AppSettings(answer_window_s=WINDOW_S)})
    analyzer = app_run.Analyzer(
        settings, load_whisper=lambda name, options: object(), load_parser=lambda: fake_parser,
        analyze=analyze, load_embedder=lambda: None,
    )  # fmt: skip
    ctx = AppContext.open(
        tmp_path / "home", SecretStr(Fernet.generate_key().decode()), raw_media_days=30
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(ctx, analyzer), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(300):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)
    ctx.close()


def test_a_full_s2_session_makes_no_outside_request(app_url: str):
    from playwright.sync_api import sync_playwright

    requests: list[str] = []
    blocked: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.add_init_script(FAKE_MIC)
        page.on("request", lambda r: requests.append(r.url))
        page.on(
            "console",
            lambda m: blocked.append(m.text) if "Content Security Policy" in m.text else None,
        )
        page.goto(app_url + "/")
        page.get_by_role("tab", name="Practice").wait_for(timeout=30_000)
        page.wait_for_timeout(3000)  # Gradio binds its events after the page has rendered
        page.get_by_text("Use the app", exact=True).click()
        page.get_by_role("button", name="Save").click()
        page.get_by_text("Saved. Practice is available.").wait_for(timeout=15_000)
        page.get_by_role("tab", name="Practice").click()
        page.get_by_role("listbox", name="Scenario").click()
        page.get_by_role("option", name="Project defense").click()
        page.get_by_text("Press Record on the microphone box to start").wait_for(timeout=15_000)
        page.get_by_role("button", name="Click to Access Microphone").click()
        page.get_by_role("button", name="start stream").click()
        page.get_by_text("Explanation: 00:03").wait_for(timeout=30_000)
        # silent from the moment question 1 appears (room-noise level, not digital zero)
        page.evaluate("window.__mic.gain.value = 0.001")
        page.get_by_role("button", name="Next: question 1").click()
        page.get_by_text("Question 1 of 2").wait_for(timeout=15_000)
        try:
            page.get_by_text("We didn't hear an answer to question 1").wait_for(timeout=30_000)
        except Exception:
            text = page.locator("body").inner_text()
            i = text.find("Microphone")
            raise AssertionError("no retry offer; page shows: " + text[i : i + 900]) from None
        page.get_by_role("button", name="Try again").click()
        page.evaluate("window.__mic.gain.value = 0.3")
        page.wait_for_timeout(2500)
        page.get_by_role("button", name="Done answering").click()
        page.get_by_text("Question 2 of 2").wait_for(timeout=15_000)
        page.wait_for_timeout(2500)
        page.get_by_role("button", name="Done answering").click()
        page.get_by_text("Both questions are done").wait_for(timeout=15_000)
        page.get_by_role("button", name="start stream").click()  # stop the microphone box
        page.wait_for_timeout(1500)
        page.get_by_role("button", name="Stop and analyse").click()
        page.get_by_text("Analysis finished.").wait_for(timeout=120_000)
        report = page.locator("body").inner_text()
        browser.close()

    assert "Follow" in report and "Questions" in report
    hosts = sorted({urlparse(u).hostname or "" for u in requests if u.startswith("http")})
    outside = [h for h in hosts if h not in LOOPBACK]
    print(f"browser journey: {len(requests)} requests, hosts {hosts}, outside {outside}")
    assert requests and outside == [], outside  # every request stayed on this computer
    assert blocked == [], blocked  # and nothing even tried to load from outside
