"""Backend RPC services for the 2D-FLCS plugin."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import api

METHOD_CORRELATE = "flc2d.correlate"
METHOD_FIT = "flc2d.fit"
METHOD_LIFETIME = "flc2d.lifetime_spectrum"
METHOD_LCURVE = "flc2d.lifetime_lcurve"
METHOD_LOAD_TTTR = "flc2d.load_tttr"
METHOD_DESCRIBE = "flc2d.contract.describe"


def _ok(result: Any) -> dict[str, Any]:
    """Return a successful JSON-RPC service payload."""
    return {"ok": True, "result": result}


def _err(exc: Exception) -> dict[str, Any]:
    """Return a failed JSON-RPC service payload."""
    return {"ok": False, "error": str(exc)}


def _array_or_none(values: Any, *, dtype: Any = float) -> np.ndarray | None:
    """Convert an optional JSON array into a NumPy array."""
    if values is None:
        return None
    return np.asarray(values, dtype=dtype)


def _tttr_payload(data: api.TttrData, *, include_arrays: bool = False) -> dict[str, Any]:
    """Serialize TTTR metadata and optionally photon arrays."""
    payload: dict[str, Any] = {
        "n_photons": int(data.n_photons),
        "macro_time_resolution_s": float(data.macro_time_resolution_s),
        "micro_time_resolution_ns": float(data.micro_time_resolution_ns),
        "n_microtime_channels": int(data.n_microtime_channels),
    }
    if include_arrays:
        payload.update(
            {
                "macro_times": np.asarray(data.macro_times, dtype=np.int64).tolist(),
                "micro_times": np.asarray(data.micro_times, dtype=np.int64).tolist(),
                "routing_channels": np.asarray(data.routing_channels, dtype=np.int64).tolist(),
            }
        )
    return payload


def load_tttr_handler(
    path: str,
    routing_channels: list[int] | None = None,
    include_arrays: bool = True,
) -> dict[str, Any]:
    """Handle ``flc2d.load_tttr``."""
    try:
        data = api.load_tttr(path, routing_channels=routing_channels)
        return _ok(_tttr_payload(data, include_arrays=include_arrays))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def correlate_handler(
    macro_times: list[int],
    micro_times: list[int],
    dT: float,
    ddT: float,
    tMin: float = 1,
    tMax: float = 4096,
    logt_imax: int = 100,
    max_bins: int | None = None,
    build_lin: bool = True,
) -> dict[str, Any]:
    """Handle ``flc2d.correlate`` by building a 2D-FDC matrix."""
    try:
        out = api.two_d_fdc(
            np.asarray(macro_times, dtype=np.int64),
            np.asarray(micro_times, dtype=np.int64),
            dT=dT,
            ddT=ddT,
            tMin=tMin,
            tMax=tMax,
            logt_imax=logt_imax,
            max_bins=max_bins,
            build_lin=build_lin,
        )
        return _ok({key: np.asarray(value).tolist() for key, value in out.items()})
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def fit_handler(
    matrix: list[list[float]],
    time_axis_ns: list[float],
    mode: str = "tikhonov",
    tau_range: list[float] | None = None,
    n_components: int = 24,
    irf: list[float] | None = None,
    irf_time_ns: list[float] | None = None,
    reg: float | None = None,
    max_bins: int | None = None,
) -> dict[str, Any]:
    """Handle ``flc2d.fit`` by inverting a 2D-FDC matrix."""
    try:
        fit_mode = str(mode).lower()
        kwargs = {
            "tau_range": tuple(tau_range or (0.3, 8.0)),
            "n_components": int(n_components),
            "irf": _array_or_none(irf),
            "irf_time_ns": _array_or_none(irf_time_ns),
            "max_bins": max_bins,
        }
        mat = np.asarray(matrix, dtype=float)
        time_axis = np.asarray(time_axis_ns, dtype=float)
        if fit_mode == "mem":
            result = api.fit_mem_2d(mat, time_axis)
        else:
            result = api.two_d_spectrum(mat, time_axis, method=fit_mode, reg=reg, **kwargs)
        return _ok(
            {
                "spectrum": result.spectrum.tolist(),
                "tau_grid": result.tau_grid.tolist(),
                "marginal": result.marginal.tolist(),
                "offset": float(result.offset),
                "chi2": float(result.chi2),
                "reg": None if result.reg is None else float(result.reg),
                "residual": np.asarray(result.residual).tolist(),
                "peak_lifetimes": result.peak_lifetimes(2).tolist(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def lifetime_spectrum_handler(
    micro_times: list[int],
    n_microtime_bins: int,
    micro_time_resolution_ns: float,
    tau_range: list[float] | None = None,
    n_components: int = 40,
    irf: list[float] | None = None,
    irf_time_ns: list[float] | None = None,
    method: str = "nnls",
    reg: float | None = None,
) -> dict[str, Any]:
    """Handle ``flc2d.lifetime_spectrum``."""
    try:
        result = api.lifetime_spectrum(
            np.asarray(micro_times, dtype=np.int64),
            n_microtime_bins=int(n_microtime_bins),
            micro_time_resolution_ns=float(micro_time_resolution_ns),
            tau_range=tuple(tau_range or (0.3, 8.0)),
            n_components=int(n_components),
            irf=_array_or_none(irf),
            irf_time_ns=_array_or_none(irf_time_ns),
            method=method,
            reg=reg,
        )
        return _ok(
            {
                "tau_grid": result.tau_grid.tolist(),
                "amplitudes": result.amplitudes.tolist(),
                "model": np.asarray(result.model).tolist(),
                "residuals": np.asarray(result.residuals).tolist(),
                "chi2": float(result.chi2),
                "reg": None if result.reg is None else float(result.reg),
                "peak_lifetimes": result.peak_lifetimes(2).tolist(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def lifetime_lcurve_handler(
    micro_times: list[int],
    n_microtime_bins: int,
    micro_time_resolution_ns: float,
    tau_range: list[float] | None = None,
    n_components: int = 40,
    irf: list[float] | None = None,
    irf_time_ns: list[float] | None = None,
    method: str = "nnls",
) -> dict[str, Any]:
    """Handle ``flc2d.lifetime_lcurve``."""
    try:
        result = api.lifetime_lcurve(
            np.asarray(micro_times, dtype=np.int64),
            n_microtime_bins=int(n_microtime_bins),
            micro_time_resolution_ns=float(micro_time_resolution_ns),
            tau_range=tuple(tau_range or (0.3, 8.0)),
            n_components=int(n_components),
            irf=_array_or_none(irf),
            irf_time_ns=_array_or_none(irf_time_ns),
            method=method,
        )
        return _ok(
            {
                "reg": np.asarray(result.reg).tolist(),
                "residual_norm": np.asarray(result.residual_norm).tolist(),
                "solution_norm": np.asarray(result.solution_norm).tolist(),
                "corner_index": int(result.corner_index),
                "corner_reg": float(result.corner_reg),
                "corner_point": list(map(float, result.corner_point)),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def contract_handler() -> dict[str, Any]:
    """Return the RPC contract exposed by the plugin."""
    return _ok(
        {
            "namespace": "flc2d",
            "methods": [
                METHOD_LOAD_TTTR,
                METHOD_CORRELATE,
                METHOD_FIT,
                METHOD_LIFETIME,
                METHOD_LCURVE,
                METHOD_DESCRIBE,
            ],
        }
    )


def register_services(dispatcher: Any) -> None:
    """Register ``flc2d.*`` RPC handlers with a service dispatcher."""
    dispatcher.register(METHOD_LOAD_TTTR, lambda params: load_tttr_handler(**(params or {})))
    dispatcher.register(METHOD_CORRELATE, lambda params: correlate_handler(**(params or {})))
    dispatcher.register(METHOD_FIT, lambda params: fit_handler(**(params or {})))
    dispatcher.register(METHOD_LIFETIME, lambda params: lifetime_spectrum_handler(**(params or {})))
    dispatcher.register(METHOD_LCURVE, lambda params: lifetime_lcurve_handler(**(params or {})))
    dispatcher.register(METHOD_DESCRIBE, lambda params: contract_handler())
