"""RPC contract constants and helpers for psf_determination."""

from __future__ import annotations

from typing import Any

PLUGIN_ID = "psf_determination"
CONTRACT_VERSION = "2.0.0"

METHOD_FIT = "psf_determination.fit.run"
METHOD_CONTRACT = "psf_determination.contract.describe"


def contract_descriptor() -> dict[str, Any]:
    """Return a dict describing the plugin RPC contract."""
    return {
        "plugin_id": PLUGIN_ID,
        "version": CONTRACT_VERSION,
        "methods": [METHOD_FIT, METHOD_CONTRACT],
    }


def service_success(result: Any) -> dict[str, Any]:
    """Wrap a result in a standard success envelope."""
    return {"ok": True, "result": result}


def service_error(msg: Any) -> dict[str, Any]:
    """Wrap an error message in a standard error envelope."""
    return {"ok": False, "error": str(msg)}
