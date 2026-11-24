"""Recording and analysis stay on this computer (owner, P3 plan): no STUN/TURN server in Stage A,
no non-loopback connection while recording or analysing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.fernet import Fernet
from pydantic import SecretStr

from presenceai.live.recorder import RecordingBuffer
from presenceai.pipeline import app_run
from presenceai.schemas.storage import Consent
from presenceai.settings import Settings
from presenceai.ui.context import AppContext
from presenceai.ui.pages import practice

T0 = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


def test_webrtc_uses_no_ice_server_on_either_side():
    from aiortc.rtcicetransport import RTCIceGatherer
    from fastrtc import WebRTC

    mic = WebRTC(
        modality="audio", mode="send", rtc_configuration=practice.NO_ICE,
        server_rtc_configuration=practice.NO_ICE,
    )  # fmt: skip
    assert mic.rtc_configuration == {"iceServers": []}  # the browser side
    server = mic.server_rtc_configuration
    assert server is not None and server.iceServers == []
    gatherer = RTCIceGatherer(iceServers=server.iceServers)
    assert gatherer._connection.stun_server is None
    assert gatherer._connection.turn_server is None
    # without the setting aiortc would contact a public STUN server: the guard above matters
    assert RTCIceGatherer()._connection.stun_server is not None


def test_recording_and_analysis_make_no_outside_connection(
    tmp_path: Path, loopback_only: list[str], fake_assessment: Any, fake_parser: Any,
    m4a_file: Path,
):  # fmt: skip
    ctx = AppContext.open(
        tmp_path / "home", SecretStr(Fernet.generate_key().decode()), raw_media_days=30, now=T0
    )
    ctx.db.set_consent(Consent(use_app=True, keep_recordings=True), now=T0)
    analyzer = app_run.Analyzer(
        Settings(_env_file=None),  # pyright: ignore[reportCallIssue]
        load_whisper=lambda name, options: object(),
        load_parser=lambda: fake_parser,
        analyze=lambda audio, **_: fake_assessment.speech,
    )
    buffer = RecordingBuffer(limit_s=180, extra_s=60, min_ratio=0.95)
    handler = practice.make_handler(buffer)
    handler.start_up()
    for _ in range(20):
        handler.receive((16_000, np.zeros((1, 1600), dtype=np.int16)))
    out = practice.analyse_recording(ctx, analyzer, "en", buffer.finish(), language="en",
                                     scenario="s1_presentation", extra_s=60)  # fmt: skip
    assert "### Delivery" in out.report
    out = practice.analyse_upload(ctx, analyzer, "en", str(m4a_file), language="en",
                                  scenario="s1_presentation")  # fmt: skip
    assert "### Delivery" in out.report
    ctx.close()
    assert loopback_only == []
