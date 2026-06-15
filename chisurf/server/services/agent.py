from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional

from chisurf.server.services import (
    OPERATION_FAILED,
    ServiceResult,
    service_error,
)
from chisurf.server.session import SessionState


def agent_code_run(
    state: SessionState,
    code: str = "",
    timeout_ms: int = 5000,
    cwd: Optional[str] = None,
) -> ServiceResult:
    """Execute a snippet of Python code in a subprocess.

    The code is written to a temporary file and run with ``sys.executable``.
    Output is captured and returned.  If the last line of stdout matches
    ``RESULT_JSON:...`` it is parsed as JSON and included in the result.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    code : str
        Python source code to execute.
    timeout_ms : int, default 5000
        Subprocess timeout in milliseconds.
    cwd : str, optional
        Working directory (defaults to repo root).

    """
    del state
    if not code:
        return service_error("no code provided", error_code=OPERATION_FAILED)

    import chisurf as cs
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(cs.__file__)))
    if cwd:
        work_dir = str(cwd)
    else:
        work_dir = repo_root

    timeout_s = max(1.0, float(timeout_ms) / 1000.0)
    max_output_chars = 20000

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="agent_code_",
        delete=False,
    )
    try:
        tmp.write(code)
        tmp.close()

        env = os.environ.copy()
        env["CHISURF_RPC_HOST"] = "127.0.0.1"
        env["CHISURF_RPC_CMD_PORT"] = "8765"
        env["CHISURF_RPC_PUB_PORT"] = "8766"

        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, tmp.name],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=work_dir,
            env=env,
            shell=False,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        stdout = (proc.stdout or "")[:max_output_chars]
        stderr = (proc.stderr or "")[:max_output_chars]

        result_json: Any = None
        stdout_lines = stdout.splitlines()
        if stdout_lines and stdout_lines[-1].startswith("RESULT_JSON:"):
            import json as _json
            try:
                payload = stdout_lines[-1][len("RESULT_JSON:"):]
                result_json = _json.loads(payload)
            except Exception:
                pass

        return {
            "ok": True,
            "stdout": stdout,
            "stderr": stderr,
            "result_json": result_json,
            "elapsed_ms": elapsed_ms,
        }
    except subprocess.TimeoutExpired:
        return service_error(
            f"code execution timed out after {timeout_s}s",
            error_code=OPERATION_FAILED,
        )
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
