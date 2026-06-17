"""Dedicated ZMQ server adapter for MaxEnt MEM services."""

from __future__ import annotations

from typing import Any

from ..api.contract import METHOD_DESCRIBE, METHOD_RUN_FRET, METHOD_RUN_LCURVE, METHOD_RUN_LIFETIME
from ..backend.services import contract_handler, run_fret_handler, run_lcurve_handler, run_lifetime_handler


def _dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a ZMQ method call to the shared backend handler."""
    handlers = {
        METHOD_RUN_LIFETIME: lambda payload: run_lifetime_handler(**(payload or {})),
        METHOD_RUN_FRET: lambda payload: run_fret_handler(**(payload or {})),
        METHOD_RUN_LCURVE: lambda payload: run_lcurve_handler(**(payload or {})),
        METHOD_DESCRIBE: lambda _payload: contract_handler(),
    }
    try:
        return handlers[method](params or {})
    except KeyError as exc:
        raise KeyError(f"unknown maxent_decay method: {method}") from exc


def serve(host: str = "127.0.0.1", cmd_port: int = 8780, pub_port: int = 8781) -> None:
    """Run a dedicated MaxEnt MEM ZMQ server."""
    from chisurf.server.transport.zmq import ZmqServer

    server = ZmqServer(
        handler=lambda method, params: _dispatch(method, params or {}),
        cmd_port=cmd_port,
        pub_port=pub_port,
        host=host,
    )
    server.serve_forever()
