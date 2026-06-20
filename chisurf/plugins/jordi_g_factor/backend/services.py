"""ServiceDispatcher-compatible RPC handlers for Jordi G-Factor calculations."""

from __future__ import annotations

from typing import Any

from ..api.contract import (
    METHOD_CALCULATE,
    METHOD_PERRIN_STEADY_STATE,
    METHOD_SOLVE_LINKED_L,
)
from ..core.calculations import (
    calculate_g_factor_core,
    perrin_steady_state_anisotropy,
    solve_linked_l_from_steady_state,
)


import logging

logger = logging.getLogger(__name__)


def register_services(dispatcher: Any) -> None:
    """Register Jordi G-Factor RPC handlers with a ServiceDispatcher.

    Parameters
    ----------
    dispatcher : ServiceDispatcher
        The server's service dispatcher.
    """
    dispatcher.register(
        METHOD_CALCULATE,
        lambda params: calculate_handler(**params),
    )
    dispatcher.register(
        METHOD_PERRIN_STEADY_STATE,
        lambda params: perrin_steady_state_handler(**params),
    )
    dispatcher.register(
        METHOD_SOLVE_LINKED_L,
        lambda params: solve_linked_l_handler(**params),
    )
    dispatcher.register(
        "jordi_g_factor.archive_g_factor",
        lambda params: archive_g_factor_handler(**params),
    )


def list_methods() -> dict[str, str]:
    """Return the RPC method catalogue."""
    return {
        METHOD_CALCULATE: "Calculate G-factor based on tail matching for Jordi decays.",
        METHOD_PERRIN_STEADY_STATE: "Perrin steady-state anisotropy for a sphere.",
        METHOD_SOLVE_LINKED_L: "Solve for the linked l1=l2 mixing parameter.",
    }


def calculate_handler(
    parallel_data: list[float],
    perpendicular_data: list[float],
    region_bounds: list[float],
    decay_shift: float = 0.0,
    use_bg: bool = False,
    bg_region_bounds: list[float] | None = None,
    flip: bool = False,
) -> dict[str, Any]:
    """ZMQ RPC handler for G-factor calculation."""
    logger.info("ZMQ RPC calculate_handler: starting calculation (len=%d, use_bg=%s)", len(parallel_data), use_bg)
    res = calculate_g_factor_core(
        parallel_data=parallel_data,
        perpendicular_data=perpendicular_data,
        region_bounds=region_bounds,
        decay_shift=decay_shift,
        use_bg=use_bg,
        bg_region_bounds=bg_region_bounds,
        flip=flip,
    )
    logger.info("ZMQ RPC calculate_handler: finished calculation. Result: %s", res)
    return res


def perrin_steady_state_handler(
    tau_ns: float,
    rho_ns: float,
    r0: float = 0.38,
) -> dict[str, Any]:
    """ZMQ RPC handler for Perrin steady-state anisotropy."""
    val = perrin_steady_state_anisotropy(tau_ns=tau_ns, rho_ns=rho_ns, r0=r0)
    return {"r_steady_state": val}


def solve_linked_l_handler(
    sp: float,
    ss: float,
    g_factor: float,
    r_target: float,
) -> dict[str, Any]:
    """ZMQ RPC handler for solving linked l1=l2."""
    val = solve_linked_l_from_steady_state(sp=sp, ss=ss, g_factor=g_factor, r_target=r_target)
    return {"l_estimate": val}


def archive_g_factor_handler(
    file_path: str,
    parameters: dict[str, Any],
    active_user: str | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """ZMQ RPC handler to register reference decay and archive G-factor calibration in MFDB."""
    logger.info("ZMQ RPC archive_g_factor_handler: starting (file=%s, user=%s)", file_path, active_user)
    try:
        from chisurf.core.mfdb.result_registry import register_raw_measurement, register_calibration
        from chisurf.core.mfdb.database_resolver import resolve_database_path
        from chisurf.core.mfdb.repository import MFDatabase
        import os
        import numpy as np

        db_path = resolve_database_path()
        if not db_path or not os.path.exists(os.path.dirname(db_path)):
            logger.warning("archive_g_factor_handler: MFDB is unavailable")
            return {"ok": False, "error": "MFDB is unavailable", "calibration_id": ""}

        # Resolve active user from auth
        user_id = active_user
        if not user_id:
            try:
                with MFDatabase(db_path) as db:
                    if auth:
                        from chisurf.core.mfdb.auth import principal_from_rpc_auth
                        principal = principal_from_rpc_auth(db.conn, auth)
                        if principal and not getattr(principal, "is_anonymous", False):
                            user_id = principal.user_id
            except Exception:
                pass
        if not user_id:
            user_id = "user_default"

        # Calculate r_inf if not already provided or if we can refine it
        r_inf = parameters.get("r_inf")
        g_val = parameters.get("g_factor")
        l1 = parameters.get("l1", 0.0)
        l2 = parameters.get("l2", 0.0)
        decay_shift = parameters.get("decay_shift", 0.0)
        use_bg = parameters.get("use_bg", False)
        flip = parameters.get("flip", False)
        
        region_min = parameters.get("region_min")
        region_max = parameters.get("region_max")

        if g_val is None:
            logger.warning("archive_g_factor_handler: no g_factor provided; cannot archive calibration")
            return {"ok": False, "error": "g_factor is required", "calibration_id": ""}

        # Compute the derived decays (background-corrected VV/VH and the
        # anisotropy r(t)) so they can be archived alongside the raw reference
        # decay, and refine r_inf when a region is given.
        derived_decays = None
        if os.path.exists(file_path):
            try:
                from chisurf.core.fio import read_jordi as _read_jordi
                if _read_jordi is not None:
                    vv, vh = _read_jordi(file_path, split=True)
                else:
                    vec = np.loadtxt(file_path)
                    half = len(vec) // 2
                    vv, vh = vec[:half], vec[half:]
                vv = np.asarray(vv, dtype=float)
                vh = np.asarray(vh, dtype=float)
                if flip:
                    vv, vh = vh, vv
                n = min(len(vv), len(vh))
                vv = vv[:n]
                vh = vh[:n]
                t = np.arange(n, dtype=float)

                bg_vv, bg_vh = 0.0, 0.0
                bg_region = parameters.get("bg_region_bounds") or [t[int(n*0.05)], t[int(n*0.15)]]
                if use_bg:
                    from ..core.calculations import compute_background_levels
                    bg_vv, bg_vh = compute_background_levels(
                        vv, vh, t, t + decay_shift, bg_region
                    )

                vv_corr = np.maximum(vv - bg_vv, 0.0)
                vh_corr = np.maximum(vh - bg_vh, 0.0)

                from ..core.calculations import shift_interp_on_axis, compute_rt
                vh_corr_shifted = shift_interp_on_axis(t, vh_corr, decay_shift)
                r_corr = compute_rt(vv_corr, vh_corr_shifted, g_val, l1=l1, l2=l2)

                derived_decays = {
                    "time": t.tolist(),
                    "vv_corrected": vv_corr.tolist(),
                    "vh_corrected": vh_corr.tolist(),
                    "anisotropy": r_corr.tolist(),
                }

                if r_inf is None and region_min is not None and region_max is not None:
                    rmin = max(float(t[0]), min(float(region_min), float(t[-1])))
                    rmax = max(float(t[0]), min(float(region_max), float(t[-1])))
                    if rmax < rmin:
                        rmin, rmax = rmax, rmin
                    i0 = int(np.argmin(np.abs(t - rmin)))
                    i1 = int(np.argmin(np.abs(t - rmax)))
                    if i1 <= i0:
                        i1 = min(len(t), i0 + 1)
                    r_region = r_corr[i0:i1]
                    r_region = r_region[np.isfinite(r_region)]
                    r_inf = float(np.nanmean(r_region)) if r_region.size > 0 else np.nan
            except Exception as calc_err:
                logger.warning("archive_g_factor_handler: failed to compute derived decays: %s", calc_err)
                derived_decays = None

        # Build reference decay metadata
        meta = {
            "filename": os.path.basename(file_path),
            "channel_roles": {
                "parallel": "VV",
                "perpendicular": "VH",
            },
            "active_user": user_id,
        }
        if "micro_time_resolution" in parameters:
            meta["micro_time_resolution"] = parameters["micro_time_resolution"]

        # Register raw measurement
        ref_decay_id = register_raw_measurement(
            file_path=file_path,
            metadata=meta,
        )
        if not ref_decay_id:
            logger.warning("archive_g_factor_handler: register_raw_measurement returned empty ID")

        # Register the derived decays (corrected VV/VH + anisotropy r(t)) as a
        # processed_data artifact derived from the reference decay.
        derived_decay_id = ""
        if derived_decays is not None:
            from chisurf.core.mfdb.result_registry import register_result
            derived_decay_id = register_result(
                kind="processed_data",
                data=derived_decays,
                parent_artifact_id=ref_decay_id or "",
                operation_type="calibration",
                metadata={
                    "derived_from": "jordi_g_factor",
                    "columns": ["time", "vv_corrected", "vh_corrected", "anisotropy"],
                    "use_bg": 1 if use_bg else 0,
                    "decay_shift": decay_shift,
                    "flip": 1 if flip else 0,
                },
            )
            if not derived_decay_id:
                logger.warning("archive_g_factor_handler: derived-decay registration returned empty ID")

        # Prep calibration parameters
        calib_params = {
            "g_factor": g_val,
            "g_factor_stddev": parameters.get("g_factor_stddev"),
            "g_factor_uncorrected": parameters.get("g_factor_uncorrected"),
            "g_factor_corrected": parameters.get("g_factor_corrected"),
            "r_inf": r_inf,
            "region_min": region_min,
            "region_max": region_max,
            "decay_shift": decay_shift,
            "flip": 1 if flip else 0,
            "use_bg": 1 if use_bg else 0,
            "bg_vv": parameters.get("bg_vv"),
            "bg_vh": parameters.get("bg_vh"),
            "l1": l1,
            "l2": l2,
        }
        # Filter None
        calib_params = {k: v for k, v in calib_params.items() if v is not None}

        # Calibration payload
        payload = dict(calib_params)

        calib_id = register_calibration(
            data=payload,
            calibration_type="g_factor",
            parent_artifact_id=ref_decay_id or "",
            parameters=calib_params,
            method="jordi_g_factor",
            notes=f"Calculated G-factor: {g_val:.4f} using tail matching.",
        )

        if not calib_id:
            logger.warning("archive_g_factor_handler: register_calibration returned empty ID")
            return {"ok": False, "error": "Calibration registration failed", "calibration_id": ""}

        logger.info("archive_g_factor_handler: success. Calibration ID=%s", calib_id)
        return {
            "ok": True,
            "calibration_id": calib_id,
            "reference_decay_id": ref_decay_id or "",
            "derived_decay_id": derived_decay_id,
        }

    except Exception as exc:
        logger.warning("archive_g_factor_handler failed: %s", exc, exc_info=True)
        return {"ok": False, "error": str(exc), "calibration_id": ""}
