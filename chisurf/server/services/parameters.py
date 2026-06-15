from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from chisurf.server.services import (
    ServiceResult,
    service_error,
    NOT_FOUND,
    OPERATION_FAILED,
    _resolve_fit,
)
from chisurf.server.session import SessionState


def _resolve_parameter(
    state: SessionState,
    parameter_name: str,
    fit_index: int = 0,
    fit_uid: Optional[str] = None,
    local_idx: Optional[int] = None,
    *,
    fit_error: str = "fit not found",
    access_error: Optional[str] = "cannot access model parameters",
    parameter_error: Optional[str] = None,
) -> tuple[Any, Any, Optional[ServiceResult]]:
    """Look up a parameter by name on a specific fit.

    Returns ``(fit, parameter, None)`` or ``(None, None, error)``.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    parameter_name : str
        Parameter name.
    fit_index : int
        Fit index.
    fit_uid : str, optional
        Fit UID.
    local_idx : int, optional
        Local fit index when *fit* is a fit group.
    fit_error : str
        Error message when fit is not found.
    access_error : str or None
        Error message when parameters cannot be accessed.
    parameter_error : str or None
        Error message when parameter is not found.

    """
    fit, _ = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return None, None, service_error(fit_error, error_code=NOT_FOUND)
    if local_idx is not None:
        grouped_fits = getattr(fit, "grouped_fits", None)
        if not grouped_fits or not 0 <= local_idx < len(grouped_fits):
            return fit, None, service_error("local fit not found", error_code=NOT_FOUND)
        fit = grouped_fits[local_idx]
    try:
        parameters = getattr(fit.model, "parameters_all_dict", {}) or {}
    except Exception:
        if access_error is None:
            parameters = {}
        else:
            return fit, None, service_error(access_error, error_code=OPERATION_FAILED)
    parameter = parameters.get(parameter_name)
    if parameter is None:
        message = parameter_error or f"parameter '{parameter_name}' not found"
        return fit, None, service_error(message, error_code=NOT_FOUND)
    return fit, parameter, None


def _parameter_payload(parameter_name: str, parameter: Any) -> Dict[str, Any]:
    """Build a serialisable dict from a parameter object.

    Parameters
    ----------
    parameter_name : str
        Display name for the parameter.
    parameter : object
        Parameter instance.

    """
    return {
        "name": parameter_name,
        "value": getattr(parameter, "value", None),
        "fixed": bool(getattr(parameter, "fixed", False)),
        "bounds": getattr(parameter, "bounds", None),
        "bounds_on": bool(getattr(parameter, "bounds_on", False)),
        "error_estimate": getattr(parameter, "error_estimate", None),
        "linked_to": str(getattr(getattr(parameter, "link", None), "name", "") or ""),
    }


def get_parameter(
    state: SessionState,
    parameter_name: str,
    fit_index: int = 0,
    fit_uid: Optional[str] = None,
) -> ServiceResult:
    """Return the current state of a single parameter.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    parameter_name : str
        Parameter name.
    fit_index : int
        Fit index.
    fit_uid : str, optional
        Fit UID.

    """
    _, p, error = _resolve_parameter(
        state,
        parameter_name,
        fit_index,
        fit_uid,
        access_error=None,
    )
    if error is not None:
        return error
    return {
        "ok": True,
        "parameter": _parameter_payload(parameter_name, p),
    }


def set_parameter_value(
    state: SessionState,
    parameter_name: str,
    value: float,
    fit_index: int = 0,
    fit_uid: Optional[str] = None,
    local_idx: Optional[int] = None,
) -> ServiceResult:
    """Set the numeric value of a parameter and update the model.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    parameter_name : str
        Parameter name.
    value : float
        New value.
    fit_index : int
        Fit index.
    fit_uid : str, optional
        Fit UID.
    local_idx : int, optional
        Local fit index when the selected fit is a fit group.

    """
    fit, p, error = _resolve_parameter(state, parameter_name, fit_index, fit_uid, local_idx)
    if error is not None:
        return error
    try:
        p.value = float(value)
        if hasattr(fit.model, "update_model"):
            fit.model.update_model()
        if hasattr(fit.model, "finalize"):
            fit.model.finalize()
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def set_parameter_fixed(
    state: SessionState,
    parameter_name: str,
    fixed: bool,
    fit_index: int = 0,
    fit_uid: Optional[str] = None,
    local_idx: Optional[int] = None,
) -> ServiceResult:
    """Fix or free a parameter and finalise the model.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    parameter_name : str
        Parameter name.
    fixed : bool
        ``True`` to fix, ``False`` to free.
    fit_index : int
        Fit index.
    fit_uid : str, optional
        Fit UID.
    local_idx : int, optional
        Local fit index when the selected fit is a fit group.

    """
    fit, p, error = _resolve_parameter(state, parameter_name, fit_index, fit_uid, local_idx)
    if error is not None:
        return error
    try:
        p.fixed = bool(fixed)
        if hasattr(fit.model, "finalize"):
            fit.model.finalize()
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def set_parameter_bounds(
    state: SessionState,
    parameter_name: str,
    bounds: Tuple[float, float],
    fit_index: int = 0,
    fit_uid: Optional[str] = None,
    local_idx: Optional[int] = None,
) -> ServiceResult:
    """Set the (min, max) bounds for a parameter.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    parameter_name : str
        Parameter name.
    bounds : tuple of float
        ``(min, max)`` bound values.
    fit_index : int
        Fit index.
    fit_uid : str, optional
        Fit UID.
    local_idx : int, optional
        Local fit index when the selected fit is a fit group.

    """
    fit, p, error = _resolve_parameter(state, parameter_name, fit_index, fit_uid, local_idx)
    if error is not None:
        return error
    try:
        p.bounds = tuple(float(v) for v in bounds)
        if hasattr(fit.model, "finalize"):
            fit.model.finalize()
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def set_parameter_bounds_on(
    state: SessionState,
    parameter_name: str,
    bounds_on: bool,
    fit_index: int = 0,
    fit_uid: Optional[str] = None,
    local_idx: Optional[int] = None,
) -> ServiceResult:
    """Enable or disable bound constraints for a parameter.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    parameter_name : str
        Parameter name.
    bounds_on : bool
        ``True`` to enable bounds, ``False`` to disable.
    fit_index : int
        Fit index.
    fit_uid : str, optional
        Fit UID.
    local_idx : int, optional
        Local fit index when the selected fit is a fit group.

    """
    _, p, error = _resolve_parameter(state, parameter_name, fit_index, fit_uid, local_idx)
    if error is not None:
        return error
    try:
        p.bounds_on = bool(bounds_on)
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def parameter_link(
    state: SessionState,
    parameter_name: str,
    target_parameter_name: Optional[str] = None,
    fit_index: int = 0,
    target_fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
    target_fit_uid: Optional[str] = None,
    local_idx: Optional[int] = None,
    target_local_idx: Optional[int] = None,
) -> ServiceResult:
    """Link a parameter to another parameter (same or different fit).

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    parameter_name : str
        Source parameter name.
    target_parameter_name : str, optional
        Target parameter name to link to.
    fit_index : int
        Source fit index.
    target_fit_index : int, optional
        Target fit index.
    fit_uid : str, optional
        Source fit UID.
    target_fit_uid : str, optional
        Target fit UID.
    local_idx : int, optional
        Source local fit index when the source fit is a fit group.
    target_local_idx : int, optional
        Target local fit index when the target fit is a fit group.

    """
    fit, p, error = _resolve_parameter(
        state,
        parameter_name,
        fit_index,
        fit_uid,
        local_idx,
        fit_error="source fit not found",
        access_error="cannot access source parameters",
        parameter_error=f"source parameter '{parameter_name}' not found",
    )
    if error is not None:
        return error

    target_fit = fit
    if target_fit_index is not None or target_fit_uid is not None:
        target_fit, _ = _resolve_fit(state, target_fit_index or 0, target_fit_uid)
        if target_fit is None:
            return service_error("target fit not found", error_code=NOT_FOUND)
    elif target_local_idx is not None:
        target_fit, _ = _resolve_fit(state, fit_index, fit_uid)
        if target_fit is None:
            return service_error("target fit not found", error_code=NOT_FOUND)
    if target_local_idx is not None:
        grouped_fits = getattr(target_fit, "grouped_fits", None)
        if not grouped_fits or not 0 <= target_local_idx < len(grouped_fits):
            return service_error("target local fit not found", error_code=NOT_FOUND)
        target_fit = grouped_fits[target_local_idx]

    link_to = None
    if target_parameter_name:
        try:
            tdict = getattr(target_fit.model, "parameters_all_dict", {}) or {}
        except Exception:
            return service_error("cannot access target parameters", error_code=OPERATION_FAILED)
        link_to = tdict.get(target_parameter_name)
        if link_to is None:
            return service_error(f"target parameter '{target_parameter_name}' not found", error_code=NOT_FOUND)

    try:
        p.link = link_to
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def parameter_unlink(
    state: SessionState,
    parameter_name: str,
    fit_index: int = 0,
    fit_uid: Optional[str] = None,
    local_idx: Optional[int] = None,
) -> ServiceResult:
    """Remove a parameter's link (make it independent).

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    parameter_name : str
        Parameter name.
    fit_index : int
        Fit index.
    fit_uid : str, optional
        Fit UID.
    local_idx : int, optional
        Local fit index when the selected fit is a fit group.

    """
    _, p, error = _resolve_parameter(state, parameter_name, fit_index, fit_uid, local_idx)
    if error is not None:
        return error
    try:
        p.link = None
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)
