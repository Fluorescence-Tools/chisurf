from __future__ import annotations

from typing import Any, Dict, List, Optional

from chisurf.server.services import (
    ServiceResult,
    service_error,
    NOT_FOUND,
    INVALID_INPUT,
    INVALID_STATE,
    OPERATION_FAILED,
    _resolve_fit,
)
from chisurf.server.services._stats import (
    _collect_fit_params,
    _collect_param_list,
    _safe_chi2,
    _safe_chi2r,
    _safe_n_free,
    _safe_n_points,
)
from chisurf.server.services.datasets import _resolve_dataset
from chisurf.server.session import SessionState


def _fit_data_payload(fit: Any) -> Dict[str, Any]:
    """Extract data-reference metadata from a fit.

    Parameters
    ----------
    fit : object
        Fit instance.

    """
    data = getattr(fit, "data", None)
    if data is None:
        return {}
    return {
        "name": str(getattr(data, "name", "") or ""),
        "uid": str(getattr(data, "unique_identifier", "") or ""),
        "filename": str(getattr(data, "filename", "") or ""),
        "experiment": str(getattr(data, "experiment", "") or getattr(getattr(data, "experiment", None), "name", "") or ""),
    }


def _fit_model_payload(
    fit: Any,
    *,
    fit_uid: str,
    n_points: Optional[int] = None,
    n_free: Optional[int] = None,
    chi2r: Optional[float] = None,
) -> Dict[str, Any]:
    """Extract model metadata from a fit.

    Parameters
    ----------
    fit : object
        Fit instance.
    fit_uid : str
        Fit UID to embed in parameter entries.
    n_points : int, optional
        Pre-computed point count (avoids re-reading).
    n_free : int, optional
        Pre-computed free-parameter count.
    chi2r : float, optional
        Pre-computed reduced chi-squared value.

    """
    model = getattr(fit, "model", None)
    if model is None:
        return {}
    return {
        "name": str(getattr(model, "name", "") or ""),
        "n_points": _safe_n_points(fit) if n_points is None else n_points,
        "n_free": _safe_n_free(fit) if n_free is None else n_free,
        "chi2r": _safe_chi2r(fit) if chi2r is None else chi2r,
        "parameters_all": _collect_param_list(fit, fit_uid=fit_uid),
    }


def _fit_dto(fit: Any, index: int, *, detailed: bool = False) -> Dict[str, Any]:
    """Build a serialisable summary dict for a fit.

    Parameters
    ----------
    fit : object
        Fit instance.
    index : int
        Positional index.
    detailed : bool
        If ``True``, include all parameter details.

    """
    fit_uid = str(getattr(fit, "unique_identifier", "") or "")
    chi2 = _safe_chi2(fit)
    chi2r = _safe_chi2r(fit)
    n_points = _safe_n_points(fit)
    n_free = _safe_n_free(fit)
    parameters = _collect_fit_params(fit)
    parameter_count = len(parameters) if detailed else len(getattr(getattr(fit, "model", None), "parameters_all_dict", {}) or {})
    return {
        "index": index,
        "uid": fit_uid,
        "name": str(getattr(fit, "name", "") or ""),
        "type": type(fit).__name__,
        "chi2": chi2,
        "chi2r": chi2r,
        "n_points": n_points,
        "n_free": n_free,
        "dataset_name": str(getattr(getattr(fit, "data", None), "name", "") or ""),
        "dataset_uid": str(getattr(getattr(fit, "data", None), "unique_identifier", "") or ""),
        "model_name": str(getattr(getattr(fit, "model", None), "name", "") or ""),
        "parameter_count": parameter_count,
        "data": _fit_data_payload(fit),
        "model": _fit_model_payload(fit, fit_uid=fit_uid, n_points=n_points, n_free=n_free, chi2r=chi2r),
        "parameters": parameters,
    }


def list_fits(state: SessionState) -> ServiceResult:
    """Return a summary of all fits in the session.

    Parameters
    ----------
    state : SessionState
        Server-side session state.

    """
    fits = list(state.fits)
    return {
        "ok": True,
        "fits": [_fit_dto(f, idx) for idx, f in enumerate(fits)],
    }


def get_fit_info(
    state: SessionState,
    fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
) -> ServiceResult:
    """Return detailed info for a single fit.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    fit_index : int, optional
        Positional index.
    fit_uid : str, optional
        Unique identifier.

    """
    fit, idx = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    return {
        "ok": True,
        "fit": _fit_dto(fit, idx, detailed=True),
    }


def run_fit(
    state: SessionState,
    fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
    event_bus: Any = None,
) -> ServiceResult:
    """Execute a fit and return results.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    fit_index : int, optional
        Positional index.
    fit_uid : str, optional
        Unique identifier.
    event_bus : object, optional
        Event bus for broadcasting.

    """
    fit, idx = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    try:
        chi2_before = _safe_chi2(fit)
        fit.run()
        chi2_after = _safe_chi2(fit)
        fit_uid_val = str(getattr(fit, "unique_identifier", "") or "")
        if event_bus is not None:
            event_bus.publish("fit.ran", {"fit_index": idx, "fit_uid": fit_uid_val})
        return {
            "ok": True,
            "fit_index": idx,
            "fit_uid": fit_uid_val,
            "chi2_before": chi2_before,
            "chi2_after": chi2_after,
        }
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def fit_set_dataset(
    state: SessionState,
    fit_index: int,
    dataset_index: Optional[int] = None,
    dataset_uid: Optional[str] = None,
    event_bus: Any = None,
) -> ServiceResult:
    """Associate a dataset with a fit.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    fit_index : int
        Fit index.
    dataset_index : int, optional
        Dataset index.
    dataset_uid : str, optional
        Dataset UID.
    event_bus : object, optional
        Event bus for broadcasting.

    """
    fit, _ = _resolve_fit(state, fit_index)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    d, _ = _resolve_dataset(state, dataset_index, dataset_uid)
    if d is None:
        return service_error("dataset not found", error_code=NOT_FOUND)
    try:
        fit.data = d
        if event_bus is not None:
            event_bus.publish("fit.dataset_changed", {"fit_index": fit_index})
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def fit_set_result_idx(
    state: SessionState,
    fit_index: int,
    result_idx: int,
    fit_uid: Optional[str] = None,
    event_bus: Any = None,
) -> ServiceResult:
    """Set the active result index on a fit.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    fit_index : int
        Fit index.
    result_idx : int
        Result index to set.
    fit_uid : str, optional
        Fit UID.
    event_bus : object, optional
        Event bus for broadcasting.

    """
    fit, actual_index = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    try:
        fit.set_result_idx(int(result_idx))
        if event_bus is not None:
            event_bus.publish("fit.result_idx_changed", {"fit_index": actual_index, "fit_uid": str(getattr(fit, "unique_identifier", "") or ""), "result_idx": int(result_idx)})
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def fit_set_fit_range(
    state: SessionState,
    fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
    xmin: int = 0,
    xmax: int = 0,
) -> ServiceResult:
    """Set the fit range on a server-side fit object."""
    fit, idx = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    try:
        fit.fit_range = (int(xmin), int(xmax))
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def ping(state: SessionState) -> ServiceResult:
    """Liveness check returning server version and session counts.

    Parameters
    ----------
    state : SessionState
        Server-side session state.

    """
    import chisurf as cs
    from chisurf.server.protocol import PROTOCOL_VERSION
    return {
        "ok": True,
        "status": "alive",
        "version": getattr(cs, "__version__", "unknown"),
        "protocol_version": PROTOCOL_VERSION,
        "dataset_count": len(state.datasets),
        "fit_count": len(state.fits),
    }


def remove_fits(
    state: SessionState,
    fit_indices: Optional[List[int]] = None,
    fit_uids: Optional[List[str]] = None,
    event_bus: Any = None,
) -> ServiceResult:
    """Remove fits by index or uid.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    fit_indices : list of int, optional
        Indices to remove.
    fit_uids : list of str, optional
        UIDs to remove.
    event_bus : object, optional
        Event bus for broadcasting.

    """
    to_remove: set[int] = set()
    fits = list(state.fits)

    if fit_uids:
        for i, f in enumerate(fits):
            if str(getattr(f, "unique_identifier", "")) in fit_uids:
                to_remove.add(i)

    if fit_indices:
        for i in fit_indices:
            if 0 <= int(i) < len(fits):
                to_remove.add(int(i))

    if not to_remove:
        return service_error("no fits specified for removal", error_code=INVALID_INPUT)

    kept = [f for i, f in enumerate(fits) if i not in to_remove]
    state.fits[:] = kept
    if event_bus is not None:
        event_bus.publish("fit.removed", {"removed_count": len(to_remove), "remaining_count": len(kept)})
    return {"ok": True, "removed_count": len(to_remove), "remaining_count": len(kept)}


def clear_fits(state: SessionState, event_bus: Any = None) -> ServiceResult:
    """Remove all fits from the session.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    event_bus : object, optional
        Event bus for broadcasting.

    """
    count = len(state.fits)
    state.fits.clear()
    if event_bus is not None:
        event_bus.publish("fit.cleared", {"cleared_count": count})
    return {"ok": True, "cleared_count": count}


def fit_create(
    state: SessionState,
    dataset_index: int = 0,
    dataset_indices: Optional[List[int]] = None,
    model_name: Optional[str] = None,
    fit_name: Optional[str] = None,
    model_kw: Optional[Dict[str, Any]] = None,
    event_bus: Any = None,
) -> ServiceResult:
    """Create a new fit on the server and append to SessionState.

    Supports both a single ``dataset_index`` and a list ``dataset_indices``
    (matching the ``core_fit.add_fit`` macro).  ``model_kw`` is forwarded to
    the model constructor.
    """
    datasets = list(state.datasets)
    if not datasets:
        return service_error("no datasets available", error_code=INVALID_STATE)
    indices = dataset_indices if dataset_indices is not None else [dataset_index]
    for i in indices:
        if i < 0 or i >= len(datasets):
            return service_error(f"dataset index {i} out of range", error_code=INVALID_INPUT)
    data_groups = [datasets[i] for i in indices]
    try:
        from chisurf.core.models.model import Model
        from chisurf.core.fitting.fit import FitGroup
    except ImportError as e:
        return service_error(f"fit model/fit classes not importable: {e}", error_code=OPERATION_FAILED, exception=e)
    try:
        model_class = None
        if model_name:
            def _find_model(cls):
                """Recursively search for a model subclass by ``name``."""
                for sc in cls.__subclasses__():
                    if getattr(sc, 'name', None) == model_name:
                        return sc
                    r = _find_model(sc)
                    if r is not None:
                        return r
                return None
            model_class = _find_model(Model)
        if model_class is None:
            return service_error(f"model '{model_name}' not found", error_code=NOT_FOUND)
        kw = dict(model_kw) if model_kw else {}
        fit = FitGroup(
            data=data_groups[0] if len(data_groups) == 1 else data_groups,
            model_class=model_class,
            **kw,
        )
        if fit_name:
            fit.name = fit_name
        state.add_fit(fit)
        if event_bus is not None:
            event_bus.publish("fit.created", {"fit_index": len(state.fits) - 1, "fit_uid": str(getattr(fit, "unique_identifier", "") or "")})
        return {
            "ok": True,
            "uid": str(getattr(fit, "unique_identifier", "") or ""),
            "name": str(getattr(fit, "name", "") or ""),
            "fit_index": len(state.fits) - 1,
        }
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def fit_update(
    state: SessionState,
    fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
    event_bus: Any = None,
) -> ServiceResult:
    """Update a fit (calls fit.update() on server-side object)."""
    fit, _ = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    try:
        if hasattr(fit, "update"):
            fit.update()
            if event_bus is not None:
                event_bus.publish("fit.updated", {"fit_index": fit_index, "fit_uid": fit_uid or str(getattr(fit, "unique_identifier", "") or "")})
            return {"ok": True}
        return service_error("fit has no update method", error_code=OPERATION_FAILED)
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def fit_save(
    state: SessionState,
    fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
    filename: str = "fit_export",
    file_type: str = "csv",
    save_curves: bool = False,
    event_bus: Any = None,
) -> ServiceResult:
    """Save a fit's results to disk on the server."""
    fit, idx = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    try:
        fit.save(filename, file_type, save_curves=save_curves)
        return {"ok": True, "saved_to": filename}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def fit_curve_data(
    state: SessionState,
    fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
) -> ServiceResult:
    """Return the fit's calculated curve data for plotting."""
    fit, idx = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    try:
        from chisurf.server.services.datasets import _sanitize_float_list
        result: Dict[str, Any] = {"ok": True}
        # Experimental data
        data = getattr(fit, "data", None)
        if data is not None:
            result["x"] = _sanitize_float_list(getattr(data, "x", None))
            result["y"] = _sanitize_float_list(getattr(data, "y", None))
        # Calculated curve from model
        model = getattr(fit, "model", None)
        if model is not None:
            result["fx"] = _sanitize_float_list(getattr(model, "x", None))
            result["fy"] = _sanitize_float_list(getattr(model, "y", None))
            result["residuals"] = _sanitize_float_list(getattr(model, "residuals", None))
        # Combined fit curve (for convenience)
        fit_curve = getattr(fit, "fit", None)
        if fit_curve is not None:
            result["fit_x"] = _sanitize_float_list(getattr(fit_curve, "x", None))
            result["fit_y"] = _sanitize_float_list(getattr(fit_curve, "y", None))
        return result
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)
