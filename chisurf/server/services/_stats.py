from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_chi2(fit: Any) -> Optional[float]:
    """Return the chi-squared value of *fit*, or ``None`` on failure.

    Parameters
    ----------
    fit : object
        Fit instance.

    """
    try:
        return float(getattr(fit, "chi2", float("nan")))
    except Exception:
        return None


def _safe_chi2r(fit: Any) -> Optional[float]:
    """Return the reduced chi-squared value of *fit*, or ``None`` on failure.

    Parameters
    ----------
    fit : object
        Fit instance.

    """
    try:
        return float(getattr(fit, "chi2r", float("nan")))
    except Exception:
        try:
            model = getattr(fit, "model", None)
            if model is not None:
                return float(getattr(model, "chi2r", float("nan")))
        except Exception:
            pass
        return None


def _safe_n_points(fit: Any) -> Optional[int]:
    """Return the number of fit points, or ``None`` on failure.

    Parameters
    ----------
    fit : object
        Fit instance.

    """
    try:
        model = getattr(fit, "model", None)
        if model is not None:
            return int(getattr(model, "n_points", 0))
    except Exception:
        pass
    return None


def _safe_n_free(fit: Any) -> Optional[int]:
    """Return the number of free (non-fixed) parameters, or ``None``.

    Parameters
    ----------
    fit : object
        Fit instance.

    """
    try:
        model = getattr(fit, "model", None)
        if model is not None:
            return int(getattr(model, "n_free", 0))
    except Exception:
        pass
    return None


def _collect_param_list(fit: Any, fit_uid: str = "") -> List[Dict[str, Any]]:
    """Return parameters as an ordered list (for proxy ``parameters_all``)."""
    result: List[Dict[str, Any]] = []
    try:
        plist = list(getattr(fit.model, "parameters_all", []) or []) if hasattr(fit, "model") else []
        for p in plist:
            result.append({
                "name": str(getattr(p, "name", "")),
                "fit_uid": fit_uid,
                "value": getattr(p, "value", None),
                "fixed": bool(getattr(p, "fixed", False)),
                "bounds": getattr(p, "bounds", None),
                "bounds_on": bool(getattr(p, "bounds_on", False)),
                "is_linked": bool(getattr(p, "is_linked", False)),
                "linked_to": str(getattr(getattr(p, "link", None), "name", "") or ""),
                "error_estimate": getattr(p, "error_estimate", None),
            })
    except Exception:
        pass
    return result


def _collect_fit_params(fit: Any) -> Dict[str, Dict[str, Any]]:
    """Return a dict of parameter-name → parameter properties.

    Parameters
    ----------
    fit : object
        Fit instance.

    """
    params = {}
    try:
        pdict = getattr(fit.model, "parameters_all_dict", {}) if hasattr(fit, "model") else {}
        for name, p in pdict.items():
            params[name] = {
                "value": getattr(p, "value", None),
                "fixed": bool(getattr(p, "fixed", False)),
                "bounds": getattr(p, "bounds", None),
                "bounds_on": bool(getattr(p, "bounds_on", False)),
                "linked_to": str(getattr(getattr(p, "link", None), "name", "") or ""),
                "error_estimate": getattr(p, "error_estimate", None),
            }
    except Exception:
        pass
    return params
