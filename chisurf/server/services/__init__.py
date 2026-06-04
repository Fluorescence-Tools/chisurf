from __future__ import annotations

from typing import Any, Dict, Optional

from chisurf.server.session import SessionState


ServiceResult = Dict[str, Any]
"""Standard shape returned by every service function: ``{"ok": bool, ...}``."""


# Service-level error codes (extend as needed)
NOT_FOUND = "NOT_FOUND"
INVALID_INPUT = "INVALID_INPUT"
OPERATION_FAILED = "OPERATION_FAILED"
INVALID_STATE = "INVALID_STATE"

# Maps service error codes to JSON-RPC error codes
_SERVICE_TO_JSONRPC = {
    NOT_FOUND: -32601,       # METHOD_NOT_FOUND semantics: resource not found
    INVALID_INPUT: -32602,   # INVALID_PARAMS semantics: bad input
    OPERATION_FAILED: -32603, # INTERNAL_ERROR semantics: operation failed
    INVALID_STATE: -32603,   # INTERNAL_ERROR semantics: invalid state
}


def service_error(
    message: str,
    *,
    error_code: str,
    jsonrpc_code: Optional[int] = None,
    exception: Optional[BaseException] = None,
) -> ServiceResult:
    """Build a structured service error result.

    All service functions should use this helper (or :func:`service_error`)
    instead of raw ``{"ok": False, "error": ...}`` dicts so that callers
    can inspect ``error_code`` and ``jsonrpc_code`` without guessing.
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


def _resolve_fit(
    state: SessionState,
    fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
) -> tuple[Any, int]:
    """Look up a fit by index or uid. Returns ``(fit, index)`` or ``(None, -1)``."""
    fits = list(state.fits)
    if fit_uid is not None:
        for i, f in enumerate(fits):
            if str(getattr(f, "unique_identifier", "")) == fit_uid:
                return f, i
    if fit_index is not None and 0 <= fit_index < len(fits):
        return fits[fit_index], fit_index
    return None, -1
