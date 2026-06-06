"""Shared test helpers for server tests."""

from __future__ import annotations

import socket


def find_free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def free_port_pair() -> tuple[int, int]:
    """Return two free ports for cmd/pub."""
    return find_free_port(), find_free_port()
