from __future__ import annotations

from typing import Any

PLUGIN_ID = "irf_estimator"
CONTRACT_VERSION = "1.0.0"

METHOD_ESTIMATE_IRF = "irf_estimator.jobs.estimate"
METHOD_LOAD_DECAY = "irf_estimator.data.load_decay"
METHOD_LOAD_DATASET = "irf_estimator.data.load_dataset"
METHOD_SAVE_IRF = "irf_estimator.data.save_irf"
METHOD_TRANSFER_IRF = "irf_estimator.data.transfer_irf"
METHOD_DESCRIBE_CONTRACT = "irf_estimator.contract.describe"

CANONICAL_METHODS = (
    METHOD_ESTIMATE_IRF,
    METHOD_LOAD_DECAY,
    METHOD_LOAD_DATASET,
    METHOD_SAVE_IRF,
    METHOD_TRANSFER_IRF,
    METHOD_DESCRIBE_CONTRACT,
)


def service_success(result: Any) -> dict[str, Any]:
    """Wrap an API result in the standard JSON-RPC service envelope."""
    return {"ok": True, "result": result}


def contract_descriptor() -> dict[str, Any]:
    """Return the JSON-compatible IRF Estimator workflow contract."""
    return {
        "plugin_id": PLUGIN_ID,
        "contract_version": CONTRACT_VERSION,
        "transport": {
            "rpc": "JSON-RPC over ChiSurf ServiceDispatcher/ZMQ",
            "cli": "chisurf irf-estimator",
            "api": "chisurf.plugins.fluorescence_decay.irf_estimator.api",
        },
        "inputs": {
            "EstimateIRF": {
                "type": "object",
                "properties": {
                    "settings": {"type": "object"},
                    "time_axis": {"type": "array", "items": {"type": "number"}},
                    "intensity": {"type": "array", "items": {"type": "number"}},
                    "dt": {"type": "number"},
                },
            },
            "LoadDecay": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
            "SaveIRF": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "irf_data": {"type": "array", "items": {"type": "number"}},
                    "dt": {"type": "number"},
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
    }
