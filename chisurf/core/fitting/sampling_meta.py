from typing import Any, Dict

import numpy as np


def _as_float(value: Any) -> float:
    """Convert scalar-like values to float without relying on deprecated ndarray coercion."""
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array.item())
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return float(value)

def get_sampling_metadata(fit: "chisurf.core.fitting.fit.Fit") -> Dict[str, Any]:
    """
    Extract detailed metadata for the parameters currently being sampled in a fit.
    
    This includes mapping the indices used in the sampling chain back to 
    unique parameter identifiers and providing their full context (fit/model).
    """
    model = fit.model
    # The sampler typically works on free parameters (neither fixed nor linked)
    # The order is determined by FittingParameterGroup.parameters
    free_params = model.parameters
    
    param_meta = []
    for i, p in enumerate(free_params):
        meta = {
            "index": i,
            "name": str(p.name),
            "uid": str(getattr(p, "unique_identifier", p.meta_data.get("unique_identifier", ""))),
            "initial_value": _as_float(p.value),
            "fixed": bool(getattr(p, "fixed", False)),
            "is_linked": bool(getattr(p, "is_linked", False)),
            "bounds_on": bool(getattr(p, "bounds_on", False)),
            "lower_bound": _as_float(p.bounds[0]) if p.bounds[0] is not None else float("-inf"),
            "upper_bound": _as_float(p.bounds[1]) if p.bounds[1] is not None else float("inf"),
        }
        
        # Add link info if applicable
        if meta["is_linked"]:
            link_target = getattr(p, "link", None)
            if link_target:
                meta["link_target_name"] = str(link_target.name)
                meta["link_target_uid"] = str(getattr(link_target, "unique_identifier", link_target.meta_data.get("unique_identifier", "")))
        
        param_meta.append(meta)
        
    return {
        "fit_name": str(fit.name),
        "fit_uid": str(getattr(fit, "unique_identifier", fit.meta_data.get("unique_identifier", ""))),
        "model_name": str(model.__class__.__name__),
        "n_free": len(free_params),
        "parameters": param_meta
    }
