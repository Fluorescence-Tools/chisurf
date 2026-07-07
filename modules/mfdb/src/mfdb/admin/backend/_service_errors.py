"""Structured RPC error helpers for MFDB admin service handlers.

MFDB is a standalone package and must not import ChiSurf. These mirror the
JSON-RPC error contract used across the service layer so handlers can return
inspectable ``error_code``/``jsonrpc_code`` results instead of ad-hoc dicts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

ServiceResult = Dict[str, Any]

NOT_FOUND = "NOT_FOUND"
INVALID_INPUT = "INVALID_INPUT"
OPERATION_FAILED = "OPERATION_FAILED"

_SERVICE_TO_JSONRPC = {
    NOT_FOUND: -32601,        # METHOD_NOT_FOUND semantics: resource not found
    INVALID_INPUT: -32602,    # INVALID_PARAMS semantics: bad input
    OPERATION_FAILED: -32603,  # INTERNAL_ERROR semantics: operation failed
}


def service_error(
    message: str,
    *,
    error_code: str,
    jsonrpc_code: Optional[int] = None,
    exception: Optional[BaseException] = None,
) -> ServiceResult:
    """Build a structured service error result.

    Parameters
    ----------
    message : str
        Human-readable error message.
    error_code : str
        One of the module error-code constants.
    jsonrpc_code : int, optional
        Explicit JSON-RPC code; derived from ``error_code`` when omitted.
    exception : BaseException, optional
        Originating exception; its type name is attached when given.

    Returns
    -------
    ServiceResult
        ``{"ok": False, "error", "error_code", "jsonrpc_code", ...}``.
    """
    if jsonrpc_code is None:
        jsonrpc_code = _SERVICE_TO_JSONRPC.get(error_code, -32603)
    result: ServiceResult = {
        "ok": False,
        "error": message,
        "error_code": error_code,
        "jsonrpc_code": jsonrpc_code,
    }
    if exception is not None:
        result["exception_type"] = type(exception).__name__
    return result
