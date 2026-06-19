"""Dedicated ZMQ server adapter for the Time Window Bins API.

New code should use ``backend.services`` or the public ``api`` package
directly.
"""

from __future__ import annotations

from typing import Any

from ..api.contract import METHOD_ANALYZE_FILES, METHOD_DESCRIBE_CONTRACT
from ..backend.services import analyze_files_handler, contract_handler


def _dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a ZMQ method call to the shared backend handler."""
    handlers = {
        METHOD_ANALYZE_FILES: lambda payload: analyze_files_handler(**payload),
        METHOD_DESCRIBE_CONTRACT: lambda _payload: contract_handler(),
    }
    try:
        return handlers[method](params or {})
    except KeyError as exc:
        raise KeyError(f"unknown time-window method: {method}") from exc


def serve(
    host: str = "127.0.0.1",
    cmd_port: int = 9765,
    pub_port: int = 9766,
) -> None:
    """Run a dedicated Time Window Bins ZMQ server."""
    from chisurf.server.transport.zmq import ZmqServer

    server = ZmqServer(
        handler=lambda method, params: _dispatch(method, params or {}),
        cmd_port=cmd_port,
        pub_port=pub_port,
        host=host,
    )
    server.serve_forever()


def analyze_files(
    files: list[str],
    time_window_ms: float = 10.0,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Split TTTR files into time-window BIDs."""
    return analyze_files_handler(
        files=files,
        time_window_ms=time_window_ms,
        output_dir=output_dir,
    )
