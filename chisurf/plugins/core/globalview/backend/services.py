from __future__ import annotations

from typing import Any, Dict, List, Optional

from chisurf.plugins.core.globalview.api.graph import build_graph


def register_services(dispatcher: Any) -> None:
    """Register GlobalView RPC handlers with a ServiceDispatcher.

    Parameters
    ----------
    dispatcher : ServiceDispatcher
        The dispatcher to register handlers with.
    """
    dispatcher.register("globalview.graph.build", lambda params: graph_build_handler(**params))
    dispatcher.register("globalview.parameters.list", lambda params: parameters_list_handler(**params))
    dispatcher.register("globalview.parameters.link", lambda params: parameters_link_handler(**params))
    dispatcher.register("globalview.parameters.unlink", lambda params: parameters_unlink_handler(**params))


def _get_fits() -> List[Any]:
    """Helper: import cs.fits lazily."""
    import chisurf as cs
    return list(getattr(cs, "fits", []))


def _select_fits(
    fit_indices: Optional[List[int]] = None,
    fit_uids: Optional[List[str]] = None,
) -> List[Any]:
    """Select a subset of fits by index or uid, or all if neither given."""
    fits = _get_fits()
    if fit_uids:
        uid_set = set(fit_uids)
        return [f for f in fits if str(getattr(f, "unique_identifier", "")) in uid_set]
    if fit_indices:
        return [fits[i] for i in fit_indices if 0 <= i < len(fits)]
    return fits


def graph_build_handler(
    fit_indices: Optional[List[int]] = None,
    fit_uids: Optional[List[str]] = None,
    include_fixed: bool = True,
    connect_fits: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build a parameter relationship graph."""
    try:
        fits = _select_fits(fit_indices, fit_uids)
        result = build_graph(
            fit_list=fits,
            include_fixed=include_fixed,
            connect_fits=connect_fits,
        )
        return {
            "ok": True,
            "graph": {
                "nodes": [
                    {
                        "node_idx": n.node_idx,
                        "node_type": n.node_type,
                        "name": n.name,
                        "fit_idx": n.fit_idx,
                        "value": n.value,
                        "fixed": n.fixed,
                        "is_linked": n.is_linked,
                        "link_name": n.link_name,
                    }
                    for n in result.nodes
                ],
                "edges": [
                    {"source": e.source, "target": e.target}
                    for e in result.edges
                ],
            },
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def parameters_list_handler(
    fit_indices: Optional[List[int]] = None,
    fit_uids: Optional[List[str]] = None,
    include_fixed: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """List all parameters across fits."""
    try:
        fits = _select_fits(fit_indices, fit_uids)
        params_list = []
        for fi, fit in enumerate(fits):
            try:
                parameters = list(getattr(fit.model, "parameters_all", []) or [])
            except Exception:
                parameters = []
            for p in parameters:
                try:
                    fixed = bool(getattr(p, "fixed", False))
                except Exception:
                    fixed = False
                if fixed and not include_fixed:
                    continue
                params_list.append({
                    "name": str(getattr(p, "name", "")),
                    "value": _safe_float(getattr(p, "value", None)),
                    "fixed": fixed,
                    "fit_idx": fi,
                    "fit_name": str(getattr(fit, "name", "")),
                    "is_linked": bool(getattr(p, "is_linked", False)),
                    "link_name": str(getattr(getattr(p, "link", None), "name", "") or ""),
                    "bounds": _safe_bounds(getattr(p, "bounds", None)),
                    "bounds_on": bool(getattr(p, "bounds_on", False)),
                })
        return {"ok": True, "parameters": params_list}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_bounds(bounds: Any) -> List[Optional[float]]:
    if not bounds:
        return [None, None]
    try:
        return [float(bounds[0]) if bounds[0] is not None else None,
                float(bounds[1]) if len(bounds) > 1 and bounds[1] is not None else None]
    except (TypeError, IndexError, ValueError):
        return [None, None]


def parameters_link_handler(
    source_parameter_name: str,
    target_parameter_name: str,
    source_fit_index: int,
    target_fit_index: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Link two parameters by name across fits."""
    try:
        fits = _get_fits()
        if source_fit_index >= len(fits) or target_fit_index >= len(fits):
            return {"ok": False, "error": "Fit index out of range"}
        source_fit = fits[source_fit_index]
        target_fit = fits[target_fit_index]
        try:
            source_params = getattr(source_fit.model, "parameters_all_dict", {})
            target_params = getattr(target_fit.model, "parameters_all_dict", {})
        except Exception:
            return {"ok": False, "error": "Cannot access parameter dicts"}
        source_param = source_params.get(source_parameter_name)
        target_param = target_params.get(target_parameter_name)
        if source_param is None or target_param is None:
            return {"ok": False, "error": "Parameter not found"}
        source_param.link = target_param
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def parameters_unlink_handler(
    parameter_name: str,
    fit_index: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Unlink a parameter."""
    try:
        fits = _get_fits()
        if fit_index >= len(fits):
            return {"ok": False, "error": "Fit index out of range"}
        fit = fits[fit_index]
        try:
            params_dict = getattr(fit.model, "parameters_all_dict", {})
        except Exception:
            return {"ok": False, "error": "Cannot access parameter dict"}
        param = params_dict.get(parameter_name)
        if param is None:
            return {"ok": False, "error": "Parameter not found"}
        param.link = None
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
