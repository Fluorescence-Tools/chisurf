"""Stable API/RPC contract for the light-path simulator plugin."""

from __future__ import annotations

from typing import Any

PLUGIN_ID = "lightpath_simulator"
CONTRACT_VERSION = "1.0.0"

METHOD_SIMULATE = "lightpath.simulate"
METHOD_SAVE = "lightpath.save"
METHOD_LIST = "lightpath.list"
METHOD_GET = "lightpath.get"
METHOD_GET_PROBES_INFO = "lightpath.get_probes_info"
METHOD_DESCRIBE_CONTRACT = "lightpath.contract.describe"

CANONICAL_METHODS = (
    METHOD_SIMULATE,
    METHOD_SAVE,
    METHOD_LIST,
    METHOD_GET,
    METHOD_GET_PROBES_INFO,
    METHOD_DESCRIBE_CONTRACT,
)


def service_success(result: dict[str, Any]) -> dict[str, Any]:
    """Wrap a result in the standard ChiSurf service envelope."""
    return {"ok": True, "result": result}


def service_error(error: str, error_code: str = "operation_failed") -> dict[str, Any]:
    """Wrap an error in the standard ChiSurf service envelope."""
    return {"ok": False, "error": error, "error_code": error_code}


def contract_descriptor() -> dict[str, Any]:
    """Return a JSON-compatible contract descriptor."""
    return {
        "plugin_id": PLUGIN_ID,
        "contract_version": CONTRACT_VERSION,
        "transport": {
            "rpc": "JSON-RPC over ChiSurf ServiceDispatcher/ZMQ",
            "cli": "chisurf lightpath-simulator",
            "api": "chisurf.plugins.core.lightpath_simulator.api",
        },
        "inputs": {
            "Simulate": {
                "type": "object",
                "required": ["graph"],
                "properties": {
                    "graph": {"type": "object"},
                    "db_path": {"type": ["string", "null"]},
                },
            },
            "Save": {
                "type": "object",
                "required": ["graph"],
                "properties": {
                    "graph": {"type": "object"},
                    "name": {"type": ["string", "null"]},
                    "db_path": {"type": ["string", "null"]},
                },
            },
            "Get": {
                "type": "object",
                "required": ["operation_id"],
                "properties": {
                    "operation_id": {"type": "string"},
                    "db_path": {"type": ["string", "null"]},
                },
            },
            "List": {
                "type": "object",
                "properties": {"db_path": {"type": ["string", "null"]}},
            },
            "GetProbesInfo": {
                "type": "object",
                "properties": {"db_path": {"type": ["string", "null"]}},
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
            }
        },
        "methods": list(CANONICAL_METHODS),
    }
