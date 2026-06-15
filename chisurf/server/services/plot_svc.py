"""Server-side RPC service for structured plot data payloads.

This module provides the ``plot.fit_data`` endpoint that returns
structured plot payloads for various plot types (fit data, residuals,
distributions, surfaces, scans, tables).  GUI plot widgets consume
these DTOs instead of reading live fit/model/data objects directly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from chisurf.server.services import (
    ServiceResult,
    service_error,
    NOT_FOUND,
    INVALID_INPUT,
    OPERATION_FAILED,
    _resolve_fit,
)
from chisurf.server.session import SessionState


def _sanitize_float_list(values: Any) -> list:
    """Convert numpy array or list to JSON-safe float list.
    
    Replaces NaN/Inf with None.
    """
    import numpy as np
    if values is None:
        return []
    try:
        arr = np.asarray(values, dtype=float)
        sanitized = []
        for v in arr.flat:
            if np.isnan(v) or np.isinf(v):
                sanitized.append(None)
            else:
                sanitized.append(float(v))
        return sanitized
    except Exception:
        return []


def plot_fit_data(
    state: SessionState,
    plot_type: str = "fit_data",
    fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
    **kwargs: Any,
) -> ServiceResult:
    """Return structured plot data for a fit.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    plot_type : str
        One of ``"fit_data"``, ``"residual"``, ``"distribution"``,
        ``"surface"``, ``"scan"``, ``"table"``.
    fit_index : int, optional
        Fit index.
    fit_uid : str, optional
        Fit UID.
    **kwargs
        Additional plot parameters (e.g. ``component_index`` for
        distribution plots, ``scan_job_id`` for scan plots).

    Returns
    -------
    ServiceResult
        With ``plot`` key containing:
        - ``curves``: list of curve dicts with ``x``, ``y``, ``label``, ``style`` keys
        - ``stats``: optional dict with ``chi2r``, ``n_points``, ``n_free``, ``dw``
        - ``type``: the plot type
    """
    fit, idx = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)

    plot_handlers = {
        "fit_data": _build_fit_data_plot,
        "residual": _build_residual_plot,
        "distribution": _build_distribution_plot,
        "table": _build_table_plot,
        "scan": _build_scan_plot,
        "surface": _build_surface_plot,
    }

    handler = plot_handlers.get(plot_type)
    if handler is None:
        return service_error(
            f"unknown plot_type '{plot_type}'",
            error_code=INVALID_INPUT,
        )

    try:
        return handler(fit, **kwargs)
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def _build_fit_data_plot(fit: Any, **kwargs: Any) -> ServiceResult:
    """Build the standard fit data plot payload (data + model + residuals)."""
    curves = []
    data = getattr(fit, "data", None)
    if data is not None:
        curves.append({
            "label": "data",
            "x": _sanitize_float_list(getattr(data, "x", None)),
            "y": _sanitize_float_list(getattr(data, "y", None)),
            "style": {"color": "black", "line_style": "None", "marker": "o", "marker_size": 4},
        })
        # Error bars
        ey = getattr(data, "ey", None)
        if ey is not None:
            curves.append({
                "label": "error",
                "x": _sanitize_float_list(getattr(data, "x", None)),
                "y": _sanitize_float_list(ey),
                "style": {"color": "gray", "line_style": "None", "marker": "None", "plot_type": "error_bar"},
            })

    model = getattr(fit, "model", None)
    if model is not None:
        fx = getattr(model, "x", None)
        fy = getattr(model, "y", None)
        if fx is not None and fy is not None:
            curves.append({
                "label": "fit",
                "x": _sanitize_float_list(fx),
                "y": _sanitize_float_list(fy),
                "style": {"color": "red", "line_width": 2},
            })

    # Fit curve (combined)
    fit_curve = getattr(fit, "fit", None)
    if fit_curve is not None:
        fx = getattr(fit_curve, "x", None)
        fy = getattr(fit_curve, "y", None)
        if fx is not None and fy is not None:
            curves.append({
                "label": "fit_curve",
                "x": _sanitize_float_list(fx),
                "y": _sanitize_float_list(fy),
                "style": {"color": "red", "line_width": 2},
            })

    stats = _build_stats(fit)
    return {"ok": True, "plot": {"type": "fit_data", "curves": curves, "stats": stats}}


def _build_residual_plot(fit: Any, **kwargs: Any) -> ServiceResult:
    """Build the weighted residual plot payload."""
    curves = []
    model = getattr(fit, "model", None)
    if model is not None:
        residuals = getattr(model, "residuals", None)
        x = getattr(model, "x", getattr(getattr(fit, "data", None), "x", None))
        if x is not None and residuals is not None:
            curves.append({
                "label": "weighted residuals",
                "x": _sanitize_float_list(x),
                "y": _sanitize_float_list(residuals),
                "style": {"color": "blue", "line_style": "None", "marker": "o", "marker_size": 4},
            })
            # Zero line
            curves.append({
                "label": "zero",
                "x": _sanitize_float_list(x),
                "y": [0.0] * len(x),
                "style": {"color": "gray", "line_width": 1, "line_style": "--"},
            })
    return {"ok": True, "plot": {"type": "residual", "curves": curves}}


def _build_distribution_plot(fit: Any, **kwargs: Any) -> ServiceResult:
    """Build a distribution plot payload from model attributes.

    The ``component_index`` kwarg can specify which component's
    distribution to plot.
    """
    curves = []
    model = getattr(fit, "model", None)
    if model is not None:
        # Try common distribution attribute names
        dist_name = kwargs.get("distribution_name", "")
        if dist_name:
            dist_data = getattr(model, dist_name, None)
        else:
            for candidate in ("distribution", "distance_distribution",
                              "rate_distribution", "lifetime_distribution"):
                dist_data = getattr(model, candidate, None)
                if dist_data is not None:
                    break
        if dist_data is not None:
            dx = getattr(dist_data, "x", getattr(dist_data, "centers", None))
            dy = getattr(dist_data, "y", getattr(dist_data, "amplitudes", None))
            if dx is not None and dy is not None:
                curves.append({
                    "label": "distribution",
                    "x": _sanitize_float_list(dx),
                    "y": _sanitize_float_list(dy),
                    "style": {"color": "red", "line_width": 2},
                })
    return {"ok": True, "plot": {"type": "distribution", "curves": curves}}


def _build_table_plot(fit: Any, **kwargs: Any) -> ServiceResult:
    """Build a table plot payload with fit parameters."""
    columns = ["name", "value", "fixed", "bounds", "error_estimate"]
    rows = []
    model = getattr(fit, "model", None)
    if model is not None:
        params = getattr(model, "parameters_all_dict", {}) or {}
        for name, p in params.items():
            rows.append({
                "name": name,
                "value": getattr(p, "value", None),
                "fixed": bool(getattr(p, "fixed", False)),
                "bounds": getattr(p, "bounds", None),
                "error_estimate": getattr(p, "error_estimate", None),
            })
    stats = _build_stats(fit)
    return {
        "ok": True,
        "plot": {
            "type": "table",
            "columns": columns,
            "rows": rows,
            "stats": stats,
        },
    }


def _build_scan_plot(fit: Any, **kwargs: Any) -> ServiceResult:
    """Build a parameter scan plot payload from a scan job result."""
    curves = []
    values = kwargs.get("values", [])
    chi2 = kwargs.get("chi2", [])
    if values and chi2:
        curves.append({
            "label": "scan",
            "x": list(values),
            "y": list(chi2),
            "style": {"color": "blue", "line_width": 2, "marker": "o", "marker_size": 4},
        })
    return {"ok": True, "plot": {"type": "scan", "curves": curves}}


def _build_surface_plot(fit: Any, **kwargs: Any) -> ServiceResult:
    """Build a chi2 surface plot payload."""
    data = kwargs.get("surface_data", {})
    x = data.get("x", [])
    y = data.get("y", [])
    z = data.get("z", [])
    return {
        "ok": True,
        "plot": {
            "type": "surface",
            "x": x,
            "y": y,
            "z": z,
        },
    }


def _build_stats(fit: Any) -> Dict[str, Any]:
    """Build a stats dict from a fit."""
    from chisurf.server.services._stats import _safe_chi2, _safe_chi2r, _safe_n_points, _safe_n_free
    dw = None
    try:
        model = getattr(fit, "model", None)
        if model is not None:
            residuals = getattr(model, "residuals", None)
            if residuals is not None and len(residuals) > 1:
                import numpy as np
                r = np.asarray(residuals, dtype=float)
                dw = float(np.sum(np.diff(r) ** 2) / np.sum(r ** 2))
    except Exception:
        pass
    return {
        "chi2": _safe_chi2(fit),
        "chi2r": _safe_chi2r(fit),
        "n_points": _safe_n_points(fit),
        "n_free": _safe_n_free(fit),
        "dw": dw,
    }
