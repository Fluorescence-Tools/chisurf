from __future__ import annotations

import subprocess
import sys
import time

from chisurf.server.startup import terminate_and_collect_stderr


def test_terminate_and_collect_stderr_does_not_block_live_process():
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time; sys.stderr.write('server failed\\n'); sys.stderr.flush(); time.sleep(30)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    time.sleep(0.2)
    stderr = terminate_and_collect_stderr(proc, timeout=1.0)

    assert "server failed" in stderr
    assert proc.poll() is not None


def test_terminate_and_collect_stderr_is_bounded():
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('x' * 100)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    proc.wait(timeout=1.0)
    stderr = terminate_and_collect_stderr(proc, limit=10, timeout=1.0)

    assert stderr == "x" * 10
