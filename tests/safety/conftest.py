"""Safety fixtures shared by tests/safety.

`loopback_only`: any connection to a non-loopback address fails the test (owner, P3 plan: "no
non-loopback connections during recording and analysis"; no STUN/TURN in Stage A). Loopback
(127.0.0.0/8, ::1, localhost) stays allowed, because the app talks to its own Gradio server.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from typing import Any

import pytest


def _is_loopback(address: Any) -> bool:
    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX paths and the like: local by definition
    host = str(address[0])
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a host name other than localhost


class NetworkBlocked(AssertionError):
    pass


@pytest.fixture
def loopback_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Yields the list of blocked targets (empty when nothing tried to leave the machine)."""
    blocked: list[str] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create = socket.create_connection

    def check(address: Any) -> None:
        if not _is_loopback(address):
            blocked.append(str(address[0]))
            raise NetworkBlocked(f"non-loopback connection to {address[0]!r}")

    def connect(self: socket.socket, address: Any) -> None:
        check(address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        check(address)
        return real_connect_ex(self, address)

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        check(address)
        return real_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    yield blocked
