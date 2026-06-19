"""Dedicated ZMQ server adapter for the BVA API."""

from __future__ import annotations

from typing import Any

from ..api.contract import (
    METHOD_COMPUTE_BVA,
    METHOD_DESCRIBE_CONTRACT,
)
from ..backend.services import (
    compute_bva_handler,
    contract_handler,
)


def _dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    handlers = {
        METHOD_COMPUTE_BVA: lambda payload: compute_bva_handler(**payload),
        METHOD_DESCRIBE_CONTRACT: lambda _payload: contract_handler(),
    }
    try:
        return handlers[method](params or {})
    except KeyError as exc:
        raise KeyError(f"unknown burst_bva method: {method}") from exc


def serve(host: str = "127.0.0.1", cmd_port: int = 8775, pub_port: int = 8776) -> None:
    """Run a dedicated BVA ZMQ server."""
    from chisurf.server.transport.zmq import ZmqServer

    server = ZmqServer(
        handler=lambda method, params: _dispatch(method, params or {}),
        cmd_port=cmd_port,
        pub_port=pub_port,
        host=host,
    )
    server.serve_forever()
