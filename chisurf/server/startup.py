from __future__ import annotations

import selectors
import subprocess
from typing import Any


def _read_available_stderr(proc: subprocess.Popen[Any], limit: int) -> bytes:
    """Read currently available stderr without blocking."""
    if proc.stderr is None:
        return b""
    selector = selectors.DefaultSelector()
    selector.register(proc.stderr, selectors.EVENT_READ)
    try:
        if not selector.select(timeout=0):
            return b""
        return proc.stderr.read1(limit)
    finally:
        selector.unregister(proc.stderr)


def terminate_and_collect_stderr(
    proc: subprocess.Popen[Any],
    limit: int = 2048,
    timeout: float = 1.0,
) -> str:
    """Terminate *proc* and return a bounded stderr sample.

    Reading from ``proc.stderr`` directly can block forever while the child is
    still alive. Use ``communicate()`` after terminating the child so GUI
    startup failure handling remains bounded.
    """
    stderr = _read_available_stderr(proc, limit)
    if proc.poll() is None:
        proc.terminate()
        try:
            _, remaining = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, remaining = proc.communicate(timeout=timeout)
        if remaining:
            stderr += remaining
    else:
        _, remaining = proc.communicate(timeout=timeout)
        if remaining:
            stderr += remaining

    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    else:
        text = str(stderr)
    return text[:limit]
