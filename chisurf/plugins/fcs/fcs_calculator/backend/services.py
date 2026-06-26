"""Backend RPC services for the FCS confocal calculator."""

from __future__ import annotations

from typing import Any, Dict

from ..core.algorithms import compute_confocal, water_viscosity_Pa_s, Pa_s_to_mPa_s


def compute_handler(**params) -> Dict[str, Any]:
    try:
        return {"ok": True, "result": compute_confocal(**params)}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


def water_viscosity_handler(temp_C: float) -> Dict[str, Any]:
    try:
        eta = water_viscosity_Pa_s(float(temp_C) + 273.15)
        return {"ok": True, "result": {"eta_mPa_s": Pa_s_to_mPa_s(eta)}}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


def register_services(dispatcher: Any) -> None:
    """Register the fcs_calculator RPC handlers with a ServiceDispatcher."""
    dispatcher.register("fcs_calculator.compute", lambda p: compute_handler(**p))
    dispatcher.register(
        "fcs_calculator.water_viscosity", lambda p: water_viscosity_handler(**p)
    )
