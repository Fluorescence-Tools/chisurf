from __future__ import annotations
from chisurf import typing
from chisurf.core.actions._decorator import action
import chisurf as cs


def _resolve_fit(fit_index: typing.Optional[int] = None):
    try:
        cs.logging.info(f"actions._resolve_fit: called with fit_index={fit_index}")
    except Exception:
        pass

    if fit_index is None:
        r = getattr(getattr(cs, "cs", None), "current_fit", None)
        try:
            cs.logging.info(
                f"actions._resolve_fit: returning current_fit type={type(r).__name__ if r is not None else None}"
            )
        except Exception:
            pass
        return r

    idx = int(fit_index)
    fits = list(getattr(cs, "fits", []) or [])
    if 0 <= idx < len(fits):
        r = fits[idx]
        try:
            cs.logging.info(
                f"actions._resolve_fit: resolved index {idx} -> type={type(r).__name__}, name={getattr(r, 'name', None)}"
            )
        except Exception:
            pass
        return r
    return None

@action("model.add_component", schema={"component_name": str})
def add_model_component(component_name: str, fit_index: typing.Optional[int] = None):
    """Add a component to a model."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.add_component(component_name, fit=fit)

@action("model.remove_component", schema={"component_name": str})
def remove_model_component(component_name: str, fit_index: typing.Optional[int] = None):
    """Remove a component from a model."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.remove_component(component_name, fit=fit)

@action("model.set_correction", schema={"correction_type": str, "value": None})
def set_model_correction(correction_type: str, value: typing.Any, fit_index: typing.Optional[int] = None):
    """Set a model correction (e.g. background, pile-up)."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.set_correction(correction_type, value, fit=fit)

@action("model.set_linearization", schema={"idx": int, "lin_name": str})
def set_model_linearization(idx: int, lin_name: str, fit_index: typing.Optional[int] = None):
    """Set model linearization table."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.set_linearization(int(idx), str(lin_name), fit=fit)

@action("model.unload_lintable")
def unload_model_lintable(fit_index: typing.Optional[int] = None):
    """Unload the model linearization table."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.unload_lintable(fit=fit)

@action("model.remove_local_fit", schema={"row": int})
def remove_local_fit(row: int, fit_index: typing.Optional[int] = None):
    """Remove a local fit from a global model."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.remove_local_fit(int(row), fit=fit)

@action("model.clear_local_fits")
def clear_local_fits(fit_index: typing.Optional[int] = None):
    """Clear all local fits from a global model."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.clear_local_fits(fit=fit)

@action("model.append_global_parameter", schema={"parameter_name": str})
def append_global_parameter(parameter_name: str, fit_index: typing.Optional[int] = None):
    """Append a global parameter to a global model."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.append_global_parameter(str(parameter_name), fit=fit)

@action("model.append_fit", schema={"fit_index": int})
def append_fit_to_global(fit_index: int, global_fit_index: typing.Optional[int] = None):
    """Append a fit to a global model."""
    from chisurf.macros import model as model_macros
    cs.logging.info(
        f"actions.model.append_fit_to_global: fit_index={fit_index}, global_fit_index={global_fit_index}"
    )
    fit = _resolve_fit(global_fit_index)
    if fit is None:
        cs.logging.warning("actions.model.append_fit_to_global: global fit resolution returned None; aborting")
        return {}
    try:
        cs.logging.info(
            f"actions.model.append_fit_to_global: resolved global fit type={type(fit).__name__}, name={getattr(fit, 'name', None)}; dispatching to macros"
        )
    except Exception:
        pass
    return model_macros.append_fit(int(fit_index), fit=fit)

@action("model.normalize_amplitudes")
def normalize_model_amplitudes(fit_index: typing.Optional[int] = None):
    """Normalize model amplitudes."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.normalize_amplitudes(fit=fit)

@action("model.absolute_amplitudes")
def absolute_model_amplitudes(fit_index: typing.Optional[int] = None):
    """Set model amplitudes to absolute values."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.absolute_amplitudes(fit=fit)

@action("model.change_irf", schema={"irf_idx": int, "irf_name": str})
def change_model_irf(irf_idx: int, irf_name: str, fit_index: typing.Optional[int] = None):
    """Change the IRF used by the model."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.change_irf(int(irf_idx), str(irf_name), fit=fit)

@action("model.unload_irf")
def unload_model_irf(fit_index: typing.Optional[int] = None):
    """Unload the current IRF."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.unload_irf(fit=fit)

@action("model.update", debounce_ms=200)
def update_model(fit_index: typing.Optional[int] = None):
    """Update the model's state."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.update_model(fit=fit)

@action("model.unload_background_curve")
def unload_model_background_curve(fit_index: typing.Optional[int] = None):
    """Unload the model background curve."""
    from chisurf.macros import model as model_macros
    fit = _resolve_fit(fit_index)
    if fit is None:
        return {}
    return model_macros.unload_background_curve(fit=fit)
