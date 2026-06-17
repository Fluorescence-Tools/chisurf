"""ServiceDispatcher-compatible RPC handlers for MaxEnt MEM."""

from __future__ import annotations

from typing import Any

import numpy as np

import chisurf

from ..api.contract import (
    METHOD_DESCRIBE,
    METHOD_RUN_FRET,
    METHOD_RUN_LCURVE,
    METHOD_RUN_LIFETIME,
    contract_descriptor,
    service_success,
)
from ..api.helpers import (
    build_distance_grid,
    build_tau_grid,
    run_fret_mem_from_arrays,
    run_lifetime_mem_from_arrays,
)
from ..api.models import LCurveResult, MEMRequest, MEMSettings
from ..api.serialization import request_from_dict, settings_from_dict, to_jsonable
from chisurf.server.services import OPERATION_FAILED, service_error


def register_services(dispatcher: Any) -> None:
    """Register MaxEnt MEM RPC handlers with a ServiceDispatcher."""
    dispatcher.register(METHOD_RUN_LIFETIME, lambda params: run_lifetime_handler(**(params or {})))
    dispatcher.register(METHOD_RUN_FRET, lambda params: run_fret_handler(**(params or {})))
    dispatcher.register(METHOD_RUN_LCURVE, lambda params: run_lcurve_handler(**(params or {})))
    dispatcher.register(METHOD_DESCRIBE, lambda params: contract_handler())


def list_methods() -> dict[str, str]:
    """Return the MaxEnt MEM RPC method catalogue."""
    return {
        METHOD_RUN_LIFETIME: "Run MaxEnt lifetime MEM on arrays.",
        METHOD_RUN_FRET: "Run MaxEnt FRET distance MEM on arrays.",
        METHOD_RUN_LCURVE: "Sweep regularization values and return L-curve data.",
        METHOD_DESCRIBE: "Return the MaxEnt MEM workflow contract.",
    }


def _settings(settings: dict[str, Any] | MEMSettings | None) -> MEMSettings:
    return settings_from_dict(settings)


def _request_from_payload(payload: dict[str, Any]) -> MEMRequest:
    return request_from_dict(payload)


def run_lifetime_handler(
    decay: list[float],
    irf: list[float],
    dt: float,
    settings: dict[str, Any] | None = None,
    fitrange: list[int] | tuple[int, int] | None = None,
    prior: list[float] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run lifetime MEM and return a JSON-RPC service result."""
    try:
        payload = {
            "decay": decay,
            "irf": irf,
            "dt": dt,
            "settings": settings or {},
            "fitrange": fitrange,
            "prior": prior,
        }
        payload.update(kwargs)
        request = _request_from_payload(payload)
        result = run_lifetime_mem_from_arrays(
            decay=request.decay,
            irf=request.irf,
            dt=request.dt,
            tau=build_tau_grid(
                tau_min=request.settings.tau_min,
                tau_max=request.settings.tau_max,
                tau_bins=request.settings.tau_bins,
            ),
            timeshift=request.settings.timeshift,
            background=request.settings.background,
            lamp_scatter=request.settings.lamp_scatter,
            fitrange=request.fitrange,
            irf_background=request.settings.irf_background,
            fit_start_fraction=request.settings.fit_start_fraction,
            nu=request.settings.nu,
            max_iter=request.settings.max_iter,
            tol=request.settings.tol,
            optimize_nuisance=request.settings.optimize_nuisance,
            prior=request.prior,
        )
        return service_success(result)
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED)


def run_fret_handler(
    decay: list[float],
    irf: list[float],
    dt: float,
    settings: dict[str, Any] | None = None,
    fitrange: list[int] | tuple[int, int] | None = None,
    prior: list[float] | None = None,
    donly: list[float] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run FRET distance MEM and return a JSON-RPC service result."""
    try:
        payload = {
            "decay": decay,
            "irf": irf,
            "dt": dt,
            "settings": settings or {},
            "fitrange": fitrange,
            "prior": prior,
            "donly": donly,
        }
        payload.update(kwargs)
        request = _request_from_payload(payload)
        settings_obj = request.settings
        result = run_fret_mem_from_arrays(
            decay=request.decay,
            irf=request.irf,
            dt=request.dt,
            R=build_distance_grid(
                R0=settings_obj.R0,
                r_min_frac=settings_obj.r_min_frac,
                r_max_frac=settings_obj.r_max_frac,
                r_bins=settings_obj.r_bins,
            ),
            tau0=settings_obj.tau0,
            R0=settings_obj.R0,
            x_donly=settings_obj.x_donly,
            timeshift=settings_obj.timeshift,
            background=settings_obj.background,
            lamp_scatter=settings_obj.lamp_scatter,
            fitrange=request.fitrange,
            irf_background=settings_obj.irf_background,
            fit_start_fraction=settings_obj.fit_start_fraction,
            nu=settings_obj.nu,
            max_iter=settings_obj.max_iter,
            tol=settings_obj.tol,
            period=settings_obj.period,
            donly=request.donly,
            optimize_nuisance=settings_obj.optimize_nuisance,
            prior=request.prior,
        )
        return service_success(result)
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED)


def run_lcurve_handler(
    decay: list[float],
    irf: list[float],
    dt: float,
    settings: dict[str, Any] | None = None,
    nu_grid: list[float] | None = None,
    fitrange: list[int] | tuple[int, int] | None = None,
    prior: list[float] | None = None,
    donly: list[float] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Sweep nu and return L-curve data."""
    try:
        settings_obj = _settings(settings)
        if nu_grid is None:
            center = float(settings_obj.nu)
            nu_grid = np.geomspace(max(center * 1e-2, np.finfo(float).tiny), center * 1e2, 16).tolist()
        chi2_vals: list[float] = []
        sol_vals: list[float] = []
        log10_vals: list[float] = []
        for nu_val in [float(x) for x in nu_grid]:
            if settings_obj.mode == "fret":
                result = run_fret_mem_from_arrays(
                    decay=decay,
                    irf=irf,
                    dt=dt,
                    R=build_distance_grid(
                        R0=settings_obj.R0,
                        r_min_frac=settings_obj.r_min_frac,
                        r_max_frac=settings_obj.r_max_frac,
                        r_bins=settings_obj.r_bins,
                    ),
                    tau0=settings_obj.tau0,
                    R0=settings_obj.R0,
                    x_donly=settings_obj.x_donly,
                    timeshift=settings_obj.timeshift,
                    background=settings_obj.background,
                    lamp_scatter=settings_obj.lamp_scatter,
                    fitrange=fitrange,
                    irf_background=settings_obj.irf_background,
                    fit_start_fraction=settings_obj.fit_start_fraction,
                    nu=nu_val,
                    max_iter=settings_obj.max_iter,
                    tol=settings_obj.tol,
                    period=settings_obj.period,
                    donly=donly,
                    optimize_nuisance=settings_obj.optimize_nuisance,
                    prior=prior,
                )
            else:
                result = run_lifetime_mem_from_arrays(
                    decay=decay,
                    irf=irf,
                    dt=dt,
                    tau=build_tau_grid(
                        tau_min=settings_obj.tau_min,
                        tau_max=settings_obj.tau_max,
                        tau_bins=settings_obj.tau_bins,
                    ),
                    timeshift=settings_obj.timeshift,
                    background=settings_obj.background,
                    lamp_scatter=settings_obj.lamp_scatter,
                    fitrange=fitrange,
                    irf_background=settings_obj.irf_background,
                    fit_start_fraction=settings_obj.fit_start_fraction,
                    nu=nu_val,
                    max_iter=settings_obj.max_iter,
                    tol=settings_obj.tol,
                    optimize_nuisance=settings_obj.optimize_nuisance,
                    prior=prior,
                )
            chi2_vals.append(float(result.get("chisq", np.nan)))
            p = np.asarray(result.get("p", []), dtype=float).ravel()
            sol_vals.append(float(np.linalg.norm(p)) if p.size else float("nan"))
            log10_vals.append(float(np.log10(nu_val)))

        corner_index = None
        try:
            mask = np.isfinite(chi2_vals) & np.isfinite(sol_vals) & (np.asarray(chi2_vals) > 0.0) & (np.asarray(sol_vals) > 0.0)
            if np.any(mask):
                corner_index = int(np.asarray(chisurf.core.math.regularization.discrete_lcurve_corner(np.asarray(chi2_vals)[mask], np.asarray(sol_vals)[mask]))[0])
        except Exception:
            corner_index = None
        return service_success(
            LCurveResult(
                log10_nu=log10_vals,
                chi2r=chi2_vals,
                sol_norm=sol_vals,
                corner_index=corner_index,
            )
        )
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED)


def contract_handler() -> dict[str, Any]:
    """Return the MaxEnt MEM workflow contract."""
    return service_success(contract_descriptor())


__all__ = [
    "register_services",
    "run_lifetime_handler",
    "run_fret_handler",
    "run_lcurve_handler",
    "contract_handler",
]
