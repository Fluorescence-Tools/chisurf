"""Workflow contract for the Micro-time Shifter plugin.

GUI, CLI and RPC layers normalize to :class:`ShiftRequest` and return
:class:`ShiftResult` payloads through the helpers defined here.
"""

from __future__ import annotations

from typing import Any

from .models import MFDBContext, ShiftRequest, ShiftResult

PLUGIN_ID = "microtime_shifter"
CONTRACT_VERSION = "1.0.0"

METHOD_APPLY = "microtime_shift.apply"
METHOD_LOAD_METADATA = "microtime_shift.load_metadata"
METHOD_IDENTIFY = "microtime_shift.identify"
METHOD_HISTOGRAM = "microtime_shift.histogram"
METHOD_DESCRIBE_CONTRACT = "microtime_shift.contract.describe"

CANONICAL_METHODS = (
    METHOD_APPLY,
    METHOD_LOAD_METADATA,
    METHOD_IDENTIFY,
    METHOD_HISTOGRAM,
    METHOD_DESCRIBE_CONTRACT,
)


def _mfdb_context_from_payload(payload: dict[str, Any]) -> MFDBContext:
    """Normalize MFDB context from a JSON payload."""
    raw = payload.get("mfdb") or {}
    if isinstance(raw, MFDBContext):
        return raw
    if isinstance(raw, dict):
        return MFDBContext(
            enabled=bool(raw.get("enabled", True)),
            sample_id=str(raw.get("sample_id") or ""),
            source_artifact_ids={
                str(p): str(aid)
                for p, aid in (raw.get("source_artifact_ids") or {}).items()
                if aid
            },
            register_missing_inputs=bool(raw.get("register_missing_inputs", True)),
            setup_id=str(raw.get("setup_id") or ""),
            setup_version=raw.get("setup_version"),
        )
    return MFDBContext()


def _normalize_channel_shifts(
    raw: dict[str, int] | None,
) -> dict[int, int]:
    """Normalize channel shift keys to int."""
    if not raw:
        return {}
    return {int(k): int(v) for k, v in raw.items()}


def shift_request_from_payload(payload: dict[str, Any]) -> ShiftRequest:
    """Build a :class:`ShiftRequest` from a JSON-compatible payload.

    Parameters
    ----------
    payload : dict
        Workflow/RPC request payload.

    Returns
    -------
    ShiftRequest
        Normalized request object.

    """
    return ShiftRequest(
        files=[str(p) for p in payload.get("files", [])],
        global_shift=int(payload.get("global_shift", 0)),
        channel_shifts=_normalize_channel_shifts(payload.get("channel_shifts")),
        filetype=payload.get("filetype"),
        output_dir=payload.get("output_dir"),
        mfdb=_mfdb_context_from_payload(payload),
    )


def shift_request_to_payload(request: ShiftRequest) -> dict[str, Any]:
    """Return a JSON-compatible shift request payload."""
    return {
        "files": list(request.files),
        "global_shift": request.global_shift,
        "channel_shifts": {str(k): v for k, v in request.channel_shifts.items()},
        "filetype": request.filetype,
        "output_dir": request.output_dir,
        "mfdb": {
            "enabled": request.mfdb.enabled,
            "sample_id": request.mfdb.sample_id,
            "source_artifact_ids": dict(request.mfdb.source_artifact_ids),
            "register_missing_inputs": request.mfdb.register_missing_inputs,
            "setup_id": request.mfdb.setup_id,
            "setup_version": request.mfdb.setup_version,
        },
    }


def shift_result_to_payload(result: ShiftResult) -> dict[str, Any]:
    """Return a JSON-compatible shift result payload."""
    return result.to_dict()


def service_success(result: ShiftResult | dict[str, Any]) -> dict[str, Any]:
    """Wrap an API result in the standard JSON-RPC service envelope."""
    payload = shift_result_to_payload(result) if isinstance(result, ShiftResult) else result
    return {"ok": True, "result": payload}


def contract_descriptor() -> dict[str, Any]:
    """Return the JSON-compatible Micro-time Shifter workflow contract.

    The descriptor is dependency-free so GUI, ZMQ JSON-RPC, and CLI callers
    can validate payloads without importing Qt or tttrlib.
    """
    return {
        "plugin_id": PLUGIN_ID,
        "contract_version": CONTRACT_VERSION,
        "transport": {
            "rpc": "JSON-RPC over ChiSurf ServiceDispatcher/ZMQ",
            "cli": "chisurf microtime-shift",
            "api": "chisurf.plugins.tttr.tttr_microtime_shifter.api",
        },
        "inputs": {
            "Apply": {
                "type": "object",
                "required": ["files"],
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "global_shift": {"type": "integer"},
                    "channel_shifts": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    },
                    "filetype": {"type": ["string", "null"]},
                    "output_dir": {"type": ["string", "null"]},
                    "mfdb": {"$ref": "#/definitions/MFDBContext"},
                },
            },
            "LoadMetadata": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                },
            },
            "Identify": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                },
            },
            "Histogram": {
                "type": "object",
                "required": ["path", "global_shift", "channel_shifts"],
                "properties": {
                    "path": {"type": "string"},
                    "global_shift": {"type": "integer"},
                    "channel_shifts": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
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
            "ShiftResult": {
                "type": "object",
                "required": ["output_paths_by_file"],
                "properties": {
                    "output_paths_by_file": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "applied_shifts_by_file": {
                        "type": "object",
                        "additionalProperties": {"type": "object"},
                    },
                    "mfdb_artifacts": {"type": "object"},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "definitions": {
            "MFDBContext": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "sample_id": {"type": "string"},
                    "source_artifact_ids": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "register_missing_inputs": {"type": "boolean"},
                    "setup_id": {"type": "string"},
                    "setup_version": {"type": ["integer", "null"]},
                },
            },
        },
        "rpc_methods": {
            METHOD_APPLY: {
                "input": "Apply",
                "output": "ServiceResult<ShiftResult>",
                "long_running": False,
            },
            METHOD_LOAD_METADATA: {
                "input": "LoadMetadata",
                "output": "ServiceResult",
            },
            METHOD_IDENTIFY: {
                "input": "Identify",
                "output": "ServiceResult",
            },
            METHOD_HISTOGRAM: {
                "input": "Histogram",
                "output": "ServiceResult",
            },
            METHOD_DESCRIBE_CONTRACT: {
                "input": "{}",
                "output": "ServiceResult<Contract>",
            },
        },
    }
