"""Collection of macros / functions that control parsed models."""
from __future__ import annotations

from typing import Any, List

import chisurf as cs
import chisurf.core.experiments

try:
    # Optional imports for type checks; fall back gracefully if types move
    from chisurf.core.fitting.fit import Fit, FitGroup  # type: ignore
except Exception:  # pragma: no cover - robustness for environments without these symbols
    Fit = object  # type: ignore
    FitGroup = object  # type: ignore


def _is_proxy_for(obj: Any, type_name: str) -> bool:
    """Check if *obj* is a proxy object whose DTO declares *type_name*."""
    return hasattr(obj, '_data') and isinstance(getattr(obj, '_data', None), dict) \
        and obj._data.get('type') == type_name


def _as_iterable_fits(target: Any) -> List[Any]:
    """Normalize various target forms to a list of fit objects.

    Accepts a single Fit, a FitGroup (iterable over fits), any iterable of fits,
    or returns an empty list if normalization is not possible.
    """
    if target is None:
        return []

    # Single Fit (real or proxy)
    try:
        if isinstance(target, Fit) or _is_proxy_for(target, 'Fit'):
            return [target]
    except Exception:
        pass

    # FitGroup (real or proxy)
    try:
        if isinstance(target, FitGroup) or _is_proxy_for(target, 'FitGroup'):
            return list(target) if hasattr(target, '__iter__') else [target]
    except Exception:
        pass

    # List/Tuple of fits
    if isinstance(target, (list, tuple)):
        return list(target)

    # Generic iterable fallback
    try:
        return list(iter(target))
    except Exception:
        return []


def _resolve_target_fits(fit_idx: Any = None) -> List[Any]:
    """Resolve the requested targets into a list of fits, robustly.

    Resolution order:
      1. If `fit_idx` is a Fit/FitGroup/iterable → normalize directly.
      2. If `fit_idx` is an int → use cs.fits[fit_idx].
      3. Otherwise (including None):
         - try cs.cs.current_fit
         - then cs.fits.selected
         - then cs.fits[0]
    Returns an empty list if no targets are available.
    """
    # 1) Directly provided objects
    targets = _as_iterable_fits(fit_idx)
    if targets:
        return targets

    # 2) Integer index
    if isinstance(fit_idx, int):
        try:
            return _as_iterable_fits(cs.fits[fit_idx])
        except Exception:
            return []

    # 3) Try common selection sources
    candidate = None
    try:
        candidate = getattr(cs.cs, "current_fit", None)
    except Exception:
        candidate = None

    if candidate is None:
        # Some environments keep selection on cs.fits
        candidate = getattr(cs.fits, "selected", None)

    if candidate is None:
        # Fallback to first group if present
        try:
            candidate = cs.fits[0]
        except Exception:
            candidate = None

    return _as_iterable_fits(candidate)


def change_model(function_str: str, fit_idx: Any = None) -> bool:
    """Update the parsed-model equation string for target fits.

    Parameters:
      function_str: The equation to set on the parse model (e.g., "b+1/abs(N)*...").
      fit_idx: Targets to update. Accepted forms:
        - None: use current/selected fits if available; otherwise do nothing.
        - int: index into cs.fits
        - Fit or FitGroup instance
        - Iterable of Fit instances

    Returns:
      True if at least one fit was updated; False if no suitable targets found.
    """
    targets = _resolve_target_fits(fit_idx)
    if not targets:
        try:
            print("[model_parse.change_model] No fits selected/available; skipping model update.")
        except Exception:
            pass
        return False

    updated_any = False
    for f in targets:
        m = getattr(f, "model", None)
        if m is None:
            continue
        try:
            m.func = f"{function_str}"
            try:
                # Prefer a full model update (parameters, widgets, plots)
                # when available. Fall back to a bare update_model() call
                # for non-widget models.
                try:
                    m.update()
                except Exception:
                    m.update_model()
            except Exception:
                pass
            updated_any = True
        except Exception as e:
            # Log and continue to next target without failing the whole batch
            try:
                print(f"[model_parse.change_model] Failed to set func for {f}: {e}")
            except Exception:
                pass
            continue

    return updated_any
