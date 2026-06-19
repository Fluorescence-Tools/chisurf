"""RPC method constants and service envelopes for Trace Browser."""

from __future__ import annotations

from typing import Any

METHOD_LIST_FILES = "trace_browser.files.list"
METHOD_GET_METADATA = "trace_browser.metadata.get"
METHOD_SET_METADATA = "trace_browser.metadata.set"
METHOD_LOAD_TRACE = "trace_browser.traces.load"
METHOD_EXPORT_CSV = "trace_browser.export.csv"
METHOD_CONTRACT = "trace_browser.contract.describe"

LEGACY_METHODS = {
    "trace_browser.list_files": METHOD_LIST_FILES,
    "trace_browser.get_metadata": METHOD_GET_METADATA,
    "trace_browser.set_metadata": METHOD_SET_METADATA,
    "trace_browser.load_trace": METHOD_LOAD_TRACE,
    "trace_browser.export_csv": METHOD_EXPORT_CSV,
}


def service_success(result: Any) -> dict[str, Any]:
    """Return a successful RPC envelope."""
    return {"ok": True, "result": result, "error": None}


def service_error(message: str, code: str = "ERROR") -> dict[str, Any]:
    """Return a failed RPC envelope."""
    return {"ok": False, "result": None, "error": {"message": message, "code": code}}


def unwrap(result: dict[str, Any] | None) -> Any:
    """Unwrap a successful RPC envelope."""
    if result and result.get("ok"):
        return result.get("result")
    return None


def contract_descriptor() -> dict[str, Any]:
    """Return the Trace Browser RPC contract."""
    return {
        "methods": [
            METHOD_LIST_FILES,
            METHOD_GET_METADATA,
            METHOD_SET_METADATA,
            METHOD_LOAD_TRACE,
            METHOD_EXPORT_CSV,
            METHOD_CONTRACT,
        ],
        "legacy_aliases": LEGACY_METHODS,
    }
