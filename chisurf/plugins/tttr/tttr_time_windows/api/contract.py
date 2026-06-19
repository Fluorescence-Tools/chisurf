"""Workflow contract for the Time Window Bins plugin.

GUI, CLI and RPC layers normalise to :class:`TimeWindowRequest` and return
:class:`TimeWindowResult` payloads through the helpers defined here.
"""

from __future__ import annotations

from typing import Any

from .models import TimeWindowRequest, TimeWindowResult

PLUGIN_ID = "tttr_time_windows"

METHOD_ANALYZE_FILES = "tttr_time_windows.jobs.analyze_files"
METHOD_DESCRIBE_CONTRACT = "tttr_time_windows.contract.describe"


def request_from_payload(payload: dict[str, Any]) -> TimeWindowRequest:
    """Build a :class:`TimeWindowRequest` from a JSON-compatible payload.

    Parameters
    ----------
    payload : dict
        Request payload with ``files``, ``time_window_ms``, and optional
        ``output_dir``.

    Returns
    -------
    TimeWindowRequest
        Normalised request object.
    """
    return TimeWindowRequest(
        files=[str(f) for f in payload.get("files", [])],
        time_window_ms=float(payload.get("time_window_ms", 10.0)),
        output_dir=payload.get("output_dir"),
    )


def request_to_payload(request: TimeWindowRequest) -> dict[str, Any]:
    """Return a JSON-compatible request payload."""
    return {
        "files": list(request.files),
        "time_window_ms": request.time_window_ms,
        "output_dir": request.output_dir,
    }


def result_to_payload(result: TimeWindowResult) -> dict[str, Any]:
    """Return a JSON-compatible result payload."""
    return result.to_dict()


def service_success(result: TimeWindowResult | dict[str, Any]) -> dict[str, Any]:
    """Wrap a result in the standard JSON-RPC service envelope."""
    payload = (
        result_to_payload(result)
        if isinstance(result, TimeWindowResult)
        else result
    )
    return {"ok": True, "result": payload}


def contract_descriptor() -> dict[str, Any]:
    """Return the JSON-compatible workflow contract descriptor."""
    return {
        "plugin_id": PLUGIN_ID,
        "contract_version": "1.0.0",
        "transport": {
            "rpc": "JSON-RPC over ChiSurf ServiceDispatcher/ZMQ",
            "cli": "chisurf tttr-time-windows",
            "api": "chisurf.plugins.tttr.tttr_time_windows.api",
        },
        "inputs": {
            "AnalyzeFiles": {
                "type": "object",
                "required": ["files"],
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "time_window_ms": {
                        "type": "number",
                        "default": 10.0,
                    },
                    "output_dir": {
                        "type": ["string", "null"],
                    },
                },
            },
        },
        "outputs": {
            "ServiceResult": {
                "type": "object",
                "required": ["ok"],
                "properties": {
                    "ok": {"type": "boolean"},
                    "result": {"type": "object"},
                    "error": {"type": "string"},
                    "error_code": {"type": "string"},
                },
            },
        },
        "rpc_methods": {
            METHOD_ANALYZE_FILES: {
                "input": "AnalyzeFiles",
                "output": "ServiceResult",
                "long_running": True,
            },
            METHOD_DESCRIBE_CONTRACT: {
                "input": "{}",
                "output": "ServiceResult<Contract>",
            },
        },
    }
