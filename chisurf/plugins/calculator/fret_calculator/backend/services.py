"""ServiceDispatcher-compatible RPC handlers for the FRET Calculator.

Thin adapters: accept JSON-compatible params, delegate to core, return JSON-safe results.
"""

from __future__ import annotations

from typing import Any

from ..core.algorithms import (
    compute_fret_from_distance,
    compute_fret_from_efficiency,
    compute_fret_from_lifetime,
    compute_fret_from_rate,
    compute_homo_fret,
    compute_homo_fret_backmap,
)


def register_services(dispatcher: Any) -> None:
    """Register RPC handlers with a ServiceDispatcher.

    Parameters
    ----------
    dispatcher : ServiceDispatcher
        The server's service dispatcher.

    """
    dispatcher.register(
        "fret_calculator.fret.compute",
        lambda params: fret_compute_handler(**params),
    )
    dispatcher.register(
        "fret_calculator.fret.compute_from_efficiency",
        lambda params: fret_from_efficiency_handler(**params),
    )
    dispatcher.register(
        "fret_calculator.fret.compute_from_lifetime",
        lambda params: fret_from_lifetime_handler(**params),
    )
    dispatcher.register(
        "fret_calculator.fret.compute_from_rate",
        lambda params: fret_from_rate_handler(**params),
    )
    dispatcher.register(
        "fret_calculator.homo.compute",
        lambda params: homo_compute_handler(**params),
    )
    dispatcher.register(
        "fret_calculator.homo.backmap",
        lambda params: homo_backmap_handler(**params),
    )


def list_methods() -> dict[str, str]:
    """Return the RPC method catalogue."""
    return {
        "fret_calculator.fret.compute": "Compute FRET parameters from distance.",
        "fret_calculator.fret.compute_from_efficiency": "Compute FRET parameters from efficiency.",
        "fret_calculator.fret.compute_from_lifetime": "Compute FRET parameters from lifetime.",
        "fret_calculator.fret.compute_from_rate": "Compute FRET parameters from rate constant.",
        "fret_calculator.homo.compute": "Compute homoFRET exchange rate and distance.",
        "fret_calculator.homo.backmap": "Back-map homoFRET distance to anisotropy relaxation time.",
    }


def fret_compute_handler(
    R: float,
    R0: float,
    tau0: float,
    kappa2: float = 0.667,
    sigma: float = 0.0,
) -> dict[str, Any]:
    """Compute FRET parameters from distance.

    Parameters
    ----------
    R : float
        Donor-acceptor distance.
    R0 : float
        Förster radius.
    tau0 : float
        Donor lifetime without FRET.
    kappa2 : float
        Orientation factor.
    sigma : float
        Width of Gaussian distance distribution.

    Returns
    -------
    dict
        JSON-serializable result.
    """
    try:
        result = compute_fret_from_distance(R, R0, tau0, kappa2, sigma)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def fret_from_efficiency_handler(
    E: float,
    R0: float,
    tau0: float,
    kappa2: float = 0.667,
) -> dict[str, Any]:
    """Compute FRET parameters from efficiency."""
    try:
        result = compute_fret_from_efficiency(E, R0, tau0, kappa2)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def fret_from_lifetime_handler(
    tau_DA: float,
    R0: float,
    tau0: float,
    kappa2: float = 0.667,
) -> dict[str, Any]:
    """Compute FRET parameters from lifetime."""
    try:
        result = compute_fret_from_lifetime(tau_DA, R0, tau0, kappa2)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def fret_from_rate_handler(
    kFRET: float,
    R0: float,
    tau0: float,
    kappa2: float = 0.667,
) -> dict[str, Any]:
    """Compute FRET parameters from rate constant."""
    try:
        result = compute_fret_from_rate(kFRET, R0, tau0, kappa2)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def homo_compute_handler(
    t_RM: float,
    rho: float,
    tau0: float,
    R0: float,
) -> dict[str, Any]:
    """Compute homoFRET exchange rate and effective distance."""
    try:
        result = compute_homo_fret(t_RM, rho, tau0, R0)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def homo_backmap_handler(
    R_DA: float,
    R0: float,
    tau0: float,
    rho: float,
) -> dict[str, Any]:
    """Back-map homoFRET distance to anisotropy relaxation time."""
    try:
        result = compute_homo_fret_backmap(R_DA, R0, tau0, rho)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
