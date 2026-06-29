"""ServiceDispatcher-compatible RPC handlers for the k² Distribution Calculator.

Thin adapters: accept JSON-compatible params, delegate to core, return
JSON-safe results.
"""

from __future__ import annotations

from typing import Any

from ..core.algorithms import compute_kappa2_dist


def register_services(dispatcher: Any) -> None:
    """Register RPC handlers with a ServiceDispatcher.

    Parameters
    ----------
    dispatcher : ServiceDispatcher
        The server's service dispatcher.

    """
    dispatcher.register(
        "kappa2_dist.compute",
        lambda params: _kappa2_compute_handler(**params),
    )


def _kappa2_compute_handler(**params: Any) -> dict[str, Any]:
    """Compute k² distribution from the given parameters.

    Returns
    -------
    dict
        ``{"ok": True, "result": {...}}`` or ``{"ok": False, "error": ...}``.
    """
    try:
        result = compute_kappa2_dist(**params)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
