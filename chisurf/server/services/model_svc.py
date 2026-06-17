"""Server-side RPC services for model operations (components, state).

These services allow GUI widgets to add/remove model components and
get/set model-specific state without accessing live model objects
directly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from chisurf.server.services import (
    ServiceResult,
    service_error,
    NOT_FOUND,
    INVALID_INPUT,
    INVALID_STATE,
    OPERATION_FAILED,
    _resolve_fit,
)
from chisurf.server.session import SessionState


def model_component_add(
    state: SessionState,
    component_type: str,
    fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
    event_bus: Any = None,
    **kwargs: Any,
) -> ServiceResult:
    """Add a component to a fit's model.

    Parameters
    ----------
    state : SessionState
    component_type : str
        The type/name of the component to add (model-family-specific).
    fit_index : int, optional
    fit_uid : str, optional
    event_bus : object, optional
    **kwargs
        Additional arguments forwarded to the component constructor.

    Notes
    -----
    This is a generic endpoint that dispatches to model-family-specific
    add methods.  Model widgets should call this instead of
    ``model.add_lifetime()``, ``model.add_species()``, etc.
    """
    fit, idx = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    model = getattr(fit, "model", None)
    if model is None:
        return service_error("fit has no model", error_code=INVALID_STATE)
    try:
        # Mirror the local action handler (model_macros.add_component):
        # 1. Try model.add_{type}() — e.g. add_rotation on AnisotropyModel
        # 2. Try model.{type}.append() — e.g. model.lifetimes.append()
        # 3. Try model.add_component() fallback
        added = False
        method_name = f"add_{component_type}"
        add_method = getattr(model, method_name, None)
        if callable(add_method):
            add_method(**kwargs)
            added = True
        if not added:
            target = getattr(model, component_type, None)
            if target is not None:
                append = getattr(target, "append", None)
                if callable(append):
                    try:
                        append(**kwargs)
                    except TypeError:
                        append()
                    added = True
        if not added:
            add_generic = getattr(model, "add_component", None)
            if callable(add_generic):
                add_generic(component_type, **kwargs)
                added = True
        if not added:
            return service_error(
                f"model has no method to add component '{component_type}'",
                error_code=OPERATION_FAILED,
            )
        # Re-discover parameters so newly added components appear in
        # parameters_all_dict (needed by parameter.set_value, etc.)
        if callable(getattr(model, "find_parameters", None)):
            model.find_parameters()
        if event_bus is not None:
            event_bus.publish("model.component.added", {
                "fit_uid": str(getattr(fit, "unique_identifier", "") or ""),
                "component_type": component_type,
            })
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def model_component_remove(
    state: SessionState,
    component_index: int,
    fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
    event_bus: Any = None,
) -> ServiceResult:
    """Remove a component from a fit's model by index.

    Parameters
    ----------
    state : SessionState
    component_index : int
        Index of the component to remove.
    fit_index : int, optional
    fit_uid : str, optional
    event_bus : object, optional
    """
    fit, idx = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    model = getattr(fit, "model", None)
    if model is None:
        return service_error("fit has no model", error_code=INVALID_STATE)
    try:
        # Mirror the local action handler (model_macros.remove_component):
        # 1. Try model.remove_component(index)
        # 2. Try model.{type}.pop() — e.g. model.lifetimes.pop()
        removed = False
        remove_method = getattr(model, "remove_component", None)
        if callable(remove_method):
            remove_method(component_index)
            removed = True
        if not removed:
            # Try sub-component pop — the action handler removes the last component
            for candidate in ("lifetimes", "species", "rotations",
                              "distances", "gaussians", component_type):
                target = getattr(model, candidate, None)
                if target is None:
                    continue
                pop = getattr(target, "pop", None)
                if callable(pop):
                    try:
                        pop()
                    except Exception:
                        continue
                    removed = True
                    break
        if not removed:
            return service_error(
                f"model has no method to remove component at index {component_index}",
                error_code=OPERATION_FAILED,
            )
        # Re-discover parameters so removed components disappear from
        # parameters_all_dict
        if callable(getattr(model, "find_parameters", None)):
            model.find_parameters()
        if event_bus is not None:
            event_bus.publish("model.component.removed", {
                "fit_uid": str(getattr(fit, "unique_identifier", "") or ""),
                "component_index": component_index,
            })
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def model_state_get(
    state: SessionState,
    fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
) -> ServiceResult:
    """Get JSON-safe model-specific state.

    Returns model attributes like convolve settings, IRF parameters,
    background parameters, etc.

    Parameters
    ----------
    state : SessionState
    fit_index : int, optional
    fit_uid : str, optional
    """
    fit, idx = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    model = getattr(fit, "model", None)
    if model is None:
        return service_error("fit has no model", error_code=INVALID_STATE)
    try:
        state_dict: Dict[str, Any] = {}
        # Collect common state attributes
        for attr in ("convolve", "irf_range", "background", "scatter",
                     "shift", "pulsed_excitation", "parse_function",
                     "name", "n_components", "n_lifetimes", "n_rotations",
                     "n_species", "n_distances", "n_gaussians",
                     "use_convolve", "use_irf", "use_background",
                     "use_scatter", "use_shift"):
            val = getattr(model, attr, None)
            if val is not None:
                try:
                    # Convert numpy types to JSON-safe
                    if hasattr(val, "item"):
                        val = val.item()
                    state_dict[attr] = val
                except Exception:
                    pass
        return {"ok": True, "state": state_dict}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


# Allowlist of model attributes that can be set via RPC.
# Prevents arbitrary setattr from clients and guards against
# unserializable values.
_MODEL_STATE_SET_ALLOWLIST = frozenset({
    "background",
    "convolve",
    "irf_range",
    "name",
    "parse_function",
    "pulsed_excitation",
    "scatter",
    "shift",
    "use_background",
    "use_convolve",
    "use_irf",
    "use_scatter",
    "use_shift",
})


def model_state_set(
    state: SessionState,
    state_data: Dict[str, Any],
    fit_index: Optional[int] = None,
    fit_uid: Optional[str] = None,
    event_bus: Any = None,
) -> ServiceResult:
    """Set JSON-safe model-specific state.

    Only attributes in ``_MODEL_STATE_SET_ALLOWLIST`` can be set.
    Other keys in *state_data* are silently ignored.

    Parameters
    ----------
    state : SessionState
    state_data : dict
        Dictionary of model attributes to set (must match allowlist).
    fit_index : int, optional
    fit_uid : str, optional
    event_bus : object, optional
    """
    fit, idx = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    model = getattr(fit, "model", None)
    if model is None:
        return service_error("fit has no model", error_code=INVALID_STATE)
    try:
        for key, value in state_data.items():
            if key not in _MODEL_STATE_SET_ALLOWLIST:
                continue
            setattr(model, key, value)
        if event_bus is not None:
            event_bus.publish("model.state.changed", {
                "fit_uid": str(getattr(fit, "unique_identifier", "") or ""),
                "keys": [k for k in state_data if k in _MODEL_STATE_SET_ALLOWLIST],
            })
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)
