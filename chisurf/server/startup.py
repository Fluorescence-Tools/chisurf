from __future__ import annotations

import subprocess
from typing import Any


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
    if proc.poll() is None:
        proc.terminate()
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate(timeout=timeout)
    else:
        _, stderr = proc.communicate(timeout=timeout)

    if stderr is None:
        return ""
    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    else:
        text = str(stderr)
    return text[:limit]
