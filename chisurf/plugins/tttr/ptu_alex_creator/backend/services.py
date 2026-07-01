"""ServiceDispatcher-compatible RPC handlers for ALEX Creator.

Thin adapters that accept JSON-compatible params, delegate to the ``api`` layer
(which drives the Qt-free ``core``) and return JSON-safe results.
"""

from __future__ import annotations

from typing import Any

from ..api.contract import (
    METHOD_CONVERT,
    METHOD_DESCRIBE_CONTRACT,
    METHOD_HISTOGRAM,
    METHOD_MERGE,
    contract_descriptor,
    request_from_payload,
    result_to_payload,
    run,
)
from ..core import alex_histogram, resolve_filetype


def register_services(dispatcher: Any) -> None:
    """Register ALEX Creator RPC handlers with a ServiceDispatcher."""
    dispatcher.register(METHOD_CONVERT, lambda params: _run_handler(params, "convert"))
    dispatcher.register(METHOD_MERGE, lambda params: _run_handler(params, "merge"))
    dispatcher.register(METHOD_HISTOGRAM, lambda params: _histogram_handler(**(params or {})))
    dispatcher.register(METHOD_DESCRIBE_CONTRACT, lambda params: contract_descriptor())


def list_methods() -> dict[str, str]:
    """Return the ALEX Creator RPC method catalogue."""
    return {
        METHOD_CONVERT: "Convert one ALEX file per input to micro-time.",
        METHOD_MERGE: "Merge several ALEX files into a single micro-time file.",
        METHOD_HISTOGRAM: "Return the ALEX micro-time histogram of a file.",
        METHOD_DESCRIBE_CONTRACT: "Describe the ALEX Creator workflow contract.",
    }


def _run_handler(params: dict[str, Any], mode: str) -> dict[str, Any]:
    request = request_from_payload(params or {})
    request.mode = mode
    return result_to_payload(run(request))


def _histogram_handler(
    path: str,
    alex_period: int = 8000,
    period_shift: int = 0,
    input_format: str = "Auto",
    **_: Any,
) -> dict[str, Any]:
    filetype = resolve_filetype(input_format, path)
    counts = alex_histogram(path, alex_period, period_shift, filetype)
    return {"counts": [int(c) for c in counts]}


__all__ = ["register_services", "list_methods"]
