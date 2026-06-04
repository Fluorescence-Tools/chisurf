from __future__ import annotations
from chisurf import typing
from chisurf.runtime.action_decorator import action


@action("parameter.value", schema={"parameter_name": str}, debounce_ms=200, debounce_keys=("parameter_name",))
def set_parameter_value(parameter_name: str, value: float, fit_index: int = 0):
    """Set a parameter value."""
    import chisurf
    fit_obj = chisurf.fits[int(fit_index)]
    fit_obj.set_parameter_value(str(parameter_name), float(value))
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}


@action("parameter.fixed", schema={"parameter_name": str}, debounce_ms=200, debounce_keys=("parameter_name",))
def set_parameter_fixed(parameter_name: str, fixed: bool, fit_index: int = 0):
    """Fix or release a parameter."""
    import chisurf
    fit_obj = chisurf.fits[int(fit_index)]
    fit_obj.set_parameter_fixed(str(parameter_name), bool(fixed))
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}


@action("parameter.link", schema={"source_parameter": str, "target_parameter": str})
def link_parameters(source_parameter: str, target_parameter: str, source_fit_index: int = 0, target_fit_index: int = 0):
    """Link two parameters together."""
    import chisurf
    source_fit = chisurf.fits[int(source_fit_index)]
    target_fit = chisurf.fits[int(target_fit_index)]
    target_fit.link_parameter(str(target_parameter), str(source_parameter), source_fit)
    return {
        "source_uid": str(getattr(source_fit, "unique_identifier", "")),
        "target_uid": str(getattr(target_fit, "unique_identifier", ""))
    }


@action("parameter.scan", schema={"parameter_name": str, "fit_index": int, "scan_range": tuple, "n_steps": int})
def scan_parameter(parameter_name: str, scan_range: typing.Tuple[float, float], n_steps: int = 20, fit_index: int = 0):
    """Perform a parameter scan."""
    import chisurf
    fit_obj = chisurf.fits[int(fit_index)]
    fit_obj.chi2_scan(str(parameter_name), scan_range=scan_range, n_steps=int(n_steps))
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}


@action("parameter.adaptive_scan", schema={"parameter_name": str, "fit_index": int, "scan_range": tuple, "p_value": float, "max_points_per_side": int})
def adaptive_scan_parameter(parameter_name: str, scan_range: typing.Tuple[float, float] = (None, None), p_value: float = 0.99, max_points_per_side: int = 50, fit_index: int = 0):
    """Perform an adaptive F-test-driven parameter scan."""
    import chisurf
    fit_obj = chisurf.fits[int(fit_index)]
    fit_obj.adaptive_chi2_scan(str(parameter_name), scan_range=scan_range, p_value=p_value, max_points_per_side=max_points_per_side)
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}


@action("parameter.bounds.set", schema={"parameter_name": str}, debounce_ms=200, debounce_keys=("parameter_name",))
def set_parameter_bounds(parameter_name: str, bounds: typing.Tuple[float, float], fit_index: int = 0):
    """Set bounds for a parameter."""
    import chisurf
    fit_obj = chisurf.fits[int(fit_index)]
    fit_obj.set_parameter_bounds(str(parameter_name), bounds)
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}


@action("parameter.bounds.on", schema={"parameter_name": str}, debounce_ms=200, debounce_keys=("parameter_name",))
def set_parameter_bounds_on(parameter_name: str, on: bool, fit_index: int = 0):
    """Enable or disable bounds for a parameter."""
    import chisurf
    fit_obj = chisurf.fits[int(fit_index)]
    fit_obj.set_parameter_bounds_on(str(parameter_name), bool(on))
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}


@action("parameter.unlink", schema={"source_parameter": str}, debounce_ms=200, debounce_keys=("source_parameter",))
def unlink_parameter(source_parameter: str, fit_index: int = 0):
    """Unlink a parameter."""
    import chisurf
    fit_obj = chisurf.fits[int(fit_index)]
    fit_obj.unlink_parameter(str(source_parameter))
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}
