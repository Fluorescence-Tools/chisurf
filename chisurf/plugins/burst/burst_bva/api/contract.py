"""Workflow contract for the BVA plugin."""

from __future__ import annotations

from typing import Any

from .models import BvaSettings
from .serialization import settings_from_dict, to_jsonable

PLUGIN_ID = "burst_bva"
CONTRACT_VERSION = "1.0.0"

METHOD_COMPUTE_BVA = "burst_bva.jobs.compute"
METHOD_DESCRIBE_CONTRACT = "burst_bva.contract.describe"

CANONICAL_METHODS = (
    METHOD_COMPUTE_BVA,
    METHOD_DESCRIBE_CONTRACT,
)


def bva_request_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON-compatible payload into a BVA request dict."""
    settings_payload = payload.get("settings", {})
    if isinstance(settings_payload, BvaSettings):
        settings = settings_payload
    elif isinstance(settings_payload, dict):
        settings = settings_from_dict(BvaSettings, settings_payload)
    else:
        settings = BvaSettings()
    return {
        "files": [str(p) for p in payload.get("files", [])],
        "settings": settings,
        "analysis_folder": payload.get("analysis_folder"),
        "pattern": payload.get("pattern", "bi4_bur"),
    }


def service_success(result: dict[str, Any] | Any) -> dict[str, Any]:
    """Wrap a result in the standard JSON-RPC service envelope."""
    from .serialization import to_jsonable
    return {"ok": True, "result": to_jsonable(result)}


def contract_descriptor() -> dict[str, Any]:
    """Return the JSON-compatible BVA workflow contract."""
    return {
        "plugin_id": PLUGIN_ID,
        "contract_version": CONTRACT_VERSION,
        "transport": {
            "rpc": "JSON-RPC over ChiSurf ServiceDispatcher/ZMQ",
            "cli": "chisurf bva",
            "api": "chisurf.plugins.burst.burst_bva.api",
        },
        "inputs": {
            "ComputeBVA": {
                "type": "object",
                "required": ["files"],
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}},
                    "analysis_folder": {"type": ["string", "null"]},
                    "pattern": {"type": "string"},
                    "settings": {"$ref": "#/definitions/BvaSettings"},
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
                },
            },
        },
        "definitions": {
            "BvaSettings": {
                "type": "object",
                "properties": {
                    "donor_channels": {"type": "array", "items": {"type": "integer"}},
                    "donor_micro_time_ranges": {"type": "array", "items": {"type": "array", "minItems": 2, "maxItems": 2}},
                    "acceptor_channels": {"type": "array", "items": {"type": "integer"}},
                    "acceptor_micro_time_ranges": {"type": "array", "items": {"type": "array", "minItems": 2, "maxItems": 2}},
                    "minimum_window_length": {"type": "number"},
                    "number_of_photons_per_slice": {"type": "integer"},
                    "file_type": {"type": "string"},
                },
            },
        },
        "rpc_methods": {
            METHOD_COMPUTE_BVA: {
                "input": "ComputeBVA",
                "output": "ServiceResult",
                "long_running": True,
            },
            METHOD_DESCRIBE_CONTRACT: {"input": "{}", "output": "ServiceResult"},
        },
    }
