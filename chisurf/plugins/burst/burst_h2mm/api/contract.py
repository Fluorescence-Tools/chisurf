"""Workflow contract for the H2MM plugin."""

from __future__ import annotations

from typing import Any

from .serialization import to_jsonable

PLUGIN_ID = "burst_h2mm"
CONTRACT_VERSION = "1.0.0"

METHOD_COMPUTE = "burst_h2mm.jobs.compute"
METHOD_PREPARE_WORKFLOW = "burst_h2mm.workflow.prepare"
METHOD_DESCRIBE_CONTRACT = "burst_h2mm.contract.describe"

CANONICAL_METHODS = (
    METHOD_COMPUTE,
    METHOD_PREPARE_WORKFLOW,
    METHOD_DESCRIBE_CONTRACT,
)


def service_success(result: dict[str, Any] | Any) -> dict[str, Any]:
    """Wrap a result in the standard JSON-RPC service envelope."""
    return {"ok": True, "result": to_jsonable(result)}


def contract_descriptor() -> dict[str, Any]:
    """Return the JSON-compatible H2MM workflow contract."""
    return {
        "plugin_id": PLUGIN_ID,
        "contract_version": CONTRACT_VERSION,
        "transport": {
            "rpc": "JSON-RPC over ChiSurf ServiceDispatcher/ZMQ",
            "cli": "chisurf h2mm",
            "api": "chisurf.plugins.burst.burst_h2mm.api",
        },
        "inputs": {
            "ComputeH2MM": {
                "type": "object",
                "required": ["files"],
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}},
                    "analysis_folder": {"type": ["string", "null"]},
                    "pattern": {"type": "string"},
                    "settings": {"$ref": "#/definitions/H2mmSettings"},
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
            "H2mmSettings": {
                "type": "object",
                "properties": {
                    "streams": {"type": "array", "items": {"type": "object"}},
                    "min_states": {"type": "integer", "minimum": 1},
                    "max_states": {"type": "integer", "minimum": 1},
                    "criterion": {"type": "string", "enum": ["bic", "icl"]},
                    "n_restarts": {"type": "integer", "minimum": 1},
                    "max_iter": {"type": "integer", "minimum": 1},
                    "tol": {"type": "number"},
                    "time_scale": {"type": "integer", "minimum": 1},
                    "min_photons": {"type": "integer", "minimum": 1},
                    "file_type": {"type": "string"},
                    "seed": {"type": "integer"},
                },
            },
        },
        "rpc_methods": {
            METHOD_COMPUTE: {
                "input": "ComputeH2MM",
                "output": "ServiceResult",
                "long_running": True,
            },
            METHOD_PREPARE_WORKFLOW: {
                "input": "WorkflowContext",
                "output": "ServiceResult",
                "long_running": False,
            },
            METHOD_DESCRIBE_CONTRACT: {"input": "{}", "output": "ServiceResult"},
        },
    }
