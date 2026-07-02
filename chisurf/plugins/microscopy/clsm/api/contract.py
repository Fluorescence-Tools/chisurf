"""RPC contract constants and envelope helpers for the CLSM plugin."""

from __future__ import annotations

from typing import Any

PLUGIN_ID = "clsm"
CONTRACT_VERSION = "2.0.0"

METHOD_SETUPS = "clsm.setups.list"
METHOD_INFO = "clsm.image.info"
METHOD_REPRESENTATION = "clsm.image.representation"
METHOD_DECAY = "clsm.decay.extract"
METHOD_FRC = "clsm.frc.compute"
METHOD_CONTRACT = "clsm.contract.describe"

ALL_METHODS = (
    METHOD_SETUPS,
    METHOD_INFO,
    METHOD_REPRESENTATION,
    METHOD_DECAY,
    METHOD_FRC,
    METHOD_CONTRACT,
)


def contract_descriptor() -> dict[str, Any]:
    """Return a dict describing the plugin RPC contract."""
    return {
        "plugin_id": PLUGIN_ID,
        "version": CONTRACT_VERSION,
        "methods": list(ALL_METHODS),
    }


def service_success(result: Any) -> dict[str, Any]:
    """Wrap a result in a standard success envelope."""
    return {"ok": True, "result": result}


def service_error(msg: Any) -> dict[str, Any]:
    """Wrap an error message in a standard error envelope."""
    return {"ok": False, "error": str(msg)}
