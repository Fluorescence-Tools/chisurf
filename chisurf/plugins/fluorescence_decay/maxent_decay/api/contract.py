"""Workflow contract for MaxEnt MEM RPC services."""

from __future__ import annotations

from typing import Any

from .models import LCurveResult, MEMResult
from .serialization import request_from_dict, request_to_payload as _request_to_payload, result_to_payload, settings_from_dict, to_jsonable

PLUGIN_ID = "maxent_decay"
CONTRACT_VERSION = "1.0.0"

METHOD_RUN_LIFETIME = "maxent_decay.jobs.run_lifetime_mem"
METHOD_RUN_FRET = "maxent_decay.jobs.run_fret_mem"
METHOD_RUN_LCURVE = "maxent_decay.jobs.run_lcurve"
METHOD_DESCRIBE = "maxent_decay.contract.describe"

CANONICAL_METHODS = (
    METHOD_RUN_LIFETIME,
    METHOD_RUN_FRET,
    METHOD_RUN_LCURVE,
    METHOD_DESCRIBE,
)


def mem_request_from_payload(payload: dict[str, Any]) -> Any:
    """Build a MaxEnt request from a JSON-compatible payload."""
    return request_from_dict(payload)


def service_success(result: Any) -> dict[str, Any]:
    """Wrap an API result in the standard JSON-RPC service envelope."""
    if isinstance(result, MEMResult):
        payload = result_to_payload(result)
    elif isinstance(result, LCurveResult):
        payload = to_jsonable(result)
    else:
        payload = to_jsonable(result)
    return {"ok": True, "result": payload}


def contract_descriptor() -> dict[str, Any]:
    """Return a JSON-compatible MaxEnt workflow contract."""
    return {
        "plugin_id": PLUGIN_ID,
        "contract_version": CONTRACT_VERSION,
        "transport": {
            "rpc": "JSON-RPC over ChiSurf ServiceDispatcher/ZMQ",
            "cli": "chisurf maxent-decay",
            "api": "chisurf.plugins.fluorescence_decay.maxent_decay.api",
        },
        "settings": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["lifetime", "fret"]},
                "nu": {"type": "number"},
                "max_iter": {"type": "integer"},
                "tau_min": {"type": "number"},
                "tau_max": {"type": "number"},
                "tau_bins": {"type": "integer"},
                "tau0": {"type": "number"},
                "R0": {"type": "number"},
                "r_min_frac": {"type": "number"},
                "r_max_frac": {"type": "number"},
                "r_bins": {"type": "integer"},
                "timeshift": {"type": "number"},
                "background": {"type": "number"},
                "lamp_scatter": {"type": "number"},
                "irf_background": {"type": ["number", "null"]},
                "period": {"type": ["number", "null"]},
                "x_donly": {"type": "number"},
            },
        },
        "inputs": {
            "MEMRequest": {
                "type": "object",
                "required": ["decay", "irf", "dt"],
                "properties": {
                    "decay": {"type": "array", "items": {"type": "number"}},
                    "irf": {"type": "array", "items": {"type": "number"}},
                    "dt": {"type": "number"},
                    "fitrange": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "settings": {"$ref": "#/settings"},
                    "prior": {"type": ["array", "null"], "items": {"type": "number"}},
                    "donly": {"type": ["array", "null"], "items": {"type": "number"}},
                },
            },
        },
        "outputs": {
            "MEMResult": {
                "type": "object",
                "required": ["p", "axis", "chisq", "fit_curve", "mode"],
                "properties": {
                    "p": {"type": "array", "items": {"type": "number"}},
                    "axis": {"type": "array", "items": {"type": "number"}},
                    "chisq": {"type": "number"},
                    "fit_curve": {"type": "array", "items": {"type": "number"}},
                    "residuals": {"type": "array", "items": {"type": "number"}},
                    "mode": {"type": "string"},
                },
            },
            "LCurveResult": {
                "type": "object",
                "required": ["log10_nu", "chi2r", "sol_norm"],
                "properties": {
                    "log10_nu": {"type": "array", "items": {"type": "number"}},
                    "chi2r": {"type": "array", "items": {"type": "number"}},
                    "sol_norm": {"type": "array", "items": {"type": "number"}},
                    "corner_index": {"type": ["integer", "null"]},
                },
            },
        },
        "rpc_methods": {
            METHOD_RUN_LIFETIME: {"input": "MEMRequest", "output": "MEMResult", "long_running": True},
            METHOD_RUN_FRET: {"input": "MEMRequest", "output": "MEMResult", "long_running": True},
            METHOD_RUN_LCURVE: {"input": "MEMRequest", "output": "LCurveResult", "long_running": True},
            METHOD_DESCRIBE: {"input": "{}", "output": "Contract", "long_running": False},
        },
    }


def request_to_payload(payload: Any) -> dict[str, Any]:
    """Return a JSON-compatible request payload."""
    return _request_to_payload(payload)


def settings_from_payload(payload: dict[str, Any] | None) -> Any:
    """Build MEM settings from a JSON-compatible dictionary."""
    return settings_from_dict(payload or {})
