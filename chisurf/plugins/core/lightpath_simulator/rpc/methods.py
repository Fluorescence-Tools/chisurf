"""Dedicated ZMQ method adapter for the light-path simulator plugin."""

from __future__ import annotations

from typing import Any

from ..api.contract import (
    METHOD_DESCRIBE_CONTRACT,
    METHOD_GET,
    METHOD_GET_PROBES_INFO,
    METHOD_LIST,
    METHOD_SAVE,
    METHOD_SIMULATE,
)
from .services import (
    contract_handler,
    get_handler,
    get_probes_info_handler,
    list_handler,
    save_handler,
    simulate_handler,
)


def dispatch(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch one dedicated light-path RPC call."""
    payload = params or {}
    handlers = {
        METHOD_SIMULATE: lambda: simulate_handler(**payload),
        METHOD_SAVE: lambda: save_handler(**payload),
        METHOD_LIST: lambda: list_handler(**payload),
        METHOD_GET: lambda: get_handler(**payload),
        METHOD_GET_PROBES_INFO: lambda: get_probes_info_handler(**payload),
        METHOD_DESCRIBE_CONTRACT: contract_handler,
    }
    try:
        return handlers[method]()
    except KeyError as exc:
        raise KeyError(f"unknown light-path simulator method: {method}") from exc


def serve(host: str = "127.0.0.1", cmd_port: int = 8765, pub_port: int = 8766) -> None:
    """Run a dedicated light-path simulator ZMQ server."""
    from chisurf.server.transport.zmq import ZmqServer

    server = ZmqServer(
        handler=lambda method, params: dispatch(method, params or {}),
        cmd_port=cmd_port,
        pub_port=pub_port,
        host=host,
    )
    server.serve_forever()
