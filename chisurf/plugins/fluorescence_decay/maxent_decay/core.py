"""MaxEnt TCSPC lifetime / FRET core solvers.

This module provides a stable location inside the plugin package for the
numerical MaxEnt implementation. The current implementation is imported
from ``playground.me_vin4_E_plugin``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

# Re-export the implementation from the development playground module.
from playground.me_vin4_E_plugin import (  # type: ignore
    MIN_PROB,
    load_tcspc_two_column,
    auto_fit_range_tcspc,
    mem_vin4_lifetime as _impl_mem_vin4_lifetime,
    mem_vin4_E as _impl_mem_vin4_E,
)


def _with_mem_progress(
    progress_cb: Optional[Callable[[int, float, float, float, float], None]],
    func: Callable[..., Dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Helper to forward progress callbacks into the core MEM loop.

    The underlying implementation drives iterations via the module-level
    ``_mem_trange`` iterator in :mod:`playground.me_vin4_E_plugin`. To hook
    GUI progress into this loop without altering the core numerics, we
    temporarily wrap ``_mem_trange`` so that it calls ``progress_cb`` once
    per iteration. The callback receives the iteration index and dummy
    values for ``chisq``, ``S``, ``Q`` and ``dgrad``; the GUI currently
    only uses the index.
    """

    if progress_cb is None:
        return func(*args, **kwargs)

    try:
        import playground.me_vin4_E_plugin as _mem_mod  # type: ignore
    except Exception:
        # Fallback: run without GUI progress if the module cannot be loaded.
        return func(*args, **kwargs)

    orig_trange = getattr(_mem_mod, "_mem_trange", None)
    if orig_trange is None:
        return func(*args, **kwargs)

    def _cb_trange(*tr_args: Any, **tr_kwargs: Any):
        for i in orig_trange(*tr_args, **tr_kwargs):
            try:
                progress_cb(int(i), float("nan"), float("nan"), float("nan"), float("nan"))
            except Exception:
                pass
            yield i

    _mem_mod._mem_trange = _cb_trange  # type: ignore[assignment]
    try:
        return func(*args, **kwargs)
    finally:
        _mem_mod._mem_trange = orig_trange  # type: ignore[assignment]


def mem_vin4_lifetime(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Lifetime MaxEnt solver wrapper.

    Accepts optional ``progress_cb`` / ``progress_b`` keywords and forwards
    them into the core MEM iteration loop so the GUI can update a progress
    dialog.
    """

    progress_cb = kwargs.pop("progress_cb", None) or kwargs.pop("progress_b", None)
    return _with_mem_progress(progress_cb, _impl_mem_vin4_lifetime, *args, **kwargs)


def mem_vin4_fret(
    decay: Sequence[float],
    lamp: Sequence[float],
    dt: float,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """FRET-distance MaxEnt solver wrapper.

    Accepts optional ``progress_cb`` / ``progress_b`` keywords and forwards
    them into the core MEM iteration loop so the GUI can update a progress
    dialog.
    """

    progress_cb = kwargs.pop("progress_cb", None) or kwargs.pop("progress_b", None)
    return _with_mem_progress(progress_cb, _impl_mem_vin4_E, decay, lamp, dt, *args, **kwargs)


__all__ = [
    "MIN_PROB",
    "load_tcspc_two_column",
    "auto_fit_range_tcspc",
    "mem_vin4_lifetime",
    "mem_vin4_fret",
]
