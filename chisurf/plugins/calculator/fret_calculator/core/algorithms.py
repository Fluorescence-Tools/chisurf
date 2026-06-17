"""Pure FRET and HomoFRET computation functions.

No Qt, no ZMQ, no GUI dependencies — pure math only.
"""

from __future__ import annotations

import numpy as np

from chisurf.core.fluorescence.general import (
    distance_to_fret_efficiency,
    distance_to_fret_rate_constant,
    fret_efficiency_to_distance,
    fret_efficiency_to_lifetime,
    fretrate_to_distance,
    gaussian2rates,
    lifetime_to_fret_efficiency,
)

# ── heteroFRET ───────────────────────────────────────────────────


def compute_fret_from_distance(
    R: float,
    R0: float,
    tau0: float,
    kappa2: float = 0.667,
    sigma: float = 0.0,
) -> dict[str, float]:
    """Compute FRET parameters from donor-acceptor distance.

    Parameters
    ----------
    R : float
        Donor-acceptor distance (same units as R0).
    R0 : float
        Förster radius.
    tau0 : float
        Donor fluorescence lifetime without FRET.
    kappa2 : float
        Orientation factor (default 2/3 for free rotation).
    sigma : float
        Width of Gaussian distance distribution.  When *sigma* > 0 the
        returned efficiency and rate are distribution-averaged values.

    Returns
    -------
    dict
        ``{"R", "R0", "tau0", "kappa2", "sigma", "E", "tau_DA", "kFRET"}``
    """
    if sigma > 0:
        rates = gaussian2rates(
            means=[R],
            sigmas=[sigma],
            amplitudes=[1.0],
            tau0=tau0,
            kappa2=kappa2,
            R0=R0,
            n_points=64,
            interleaved=False,
        )
        weights = rates[:, 0]
        rate_values = rates[:, 1]
        kFRET = float(np.sum(weights * rate_values))
        avg_E = 0.0
        total_w = 0.0
        for w, r in zip(weights, rate_values):
            eff = r * tau0 / (1.0 + r * tau0)
            avg_E += w * eff
            total_w += w
        if total_w > 0:
            avg_E /= total_w
        tau_DA = fret_efficiency_to_lifetime(avg_E, tau0)
    else:
        kFRET = float(distance_to_fret_rate_constant(R, R0, tau0, kappa2))
        avg_E = float(distance_to_fret_efficiency(R, R0))
        tau_DA = fret_efficiency_to_lifetime(avg_E, tau0)

    return {
        "R": R,
        "R0": R0,
        "tau0": tau0,
        "kappa2": kappa2,
        "sigma": sigma,
        "E": avg_E,
        "tau_DA": tau_DA,
        "kFRET": kFRET,
    }


def compute_fret_from_efficiency(
    E: float,
    R0: float,
    tau0: float,
    kappa2: float = 0.667,
) -> dict[str, float]:
    """Compute FRET parameters from transfer efficiency.

    Parameters
    ----------
    E : float
        FRET efficiency (0–1).
    R0 : float
        Förster radius.
    tau0 : float
        Donor lifetime without FRET.
    kappa2 : float
        Orientation factor.

    Returns
    -------
    dict
        ``{"R", "R0", "tau0", "kappa2", "sigma", "E", "tau_DA", "kFRET"}``
    """
    R = float(fret_efficiency_to_distance(E, R0))
    tau_DA = fret_efficiency_to_lifetime(E, tau0)
    kFRET = float(distance_to_fret_rate_constant(R, R0, tau0, kappa2))
    return {
        "R": R,
        "R0": R0,
        "tau0": tau0,
        "kappa2": kappa2,
        "sigma": 0.0,
        "E": E,
        "tau_DA": tau_DA,
        "kFRET": kFRET,
    }


def compute_fret_from_lifetime(
    tau_DA: float,
    R0: float,
    tau0: float,
    kappa2: float = 0.667,
) -> dict[str, float]:
    """Compute FRET parameters from donor lifetime in presence of acceptor.

    Parameters
    ----------
    tau_DA : float
        Donor lifetime with acceptor present.
    R0 : float
        Förster radius.
    tau0 : float
        Donor lifetime without FRET.
    kappa2 : float
        Orientation factor.

    Returns
    -------
    dict
        ``{"R", "R0", "tau0", "kappa2", "sigma", "E", "tau_DA", "kFRET"}``
    """
    E = float(lifetime_to_fret_efficiency(tau_DA, tau0))
    R = float(fret_efficiency_to_distance(E, R0))
    kFRET = float(distance_to_fret_rate_constant(R, R0, tau0, kappa2))
    return {
        "R": R,
        "R0": R0,
        "tau0": tau0,
        "kappa2": kappa2,
        "sigma": 0.0,
        "E": E,
        "tau_DA": tau_DA,
        "kFRET": kFRET,
    }


def compute_fret_from_rate(
    kFRET: float,
    R0: float,
    tau0: float,
    kappa2: float = 0.667,
) -> dict[str, float]:
    """Compute FRET parameters from FRET rate constant.

    Parameters
    ----------
    kFRET : float
        FRET rate constant (1/time, same units as 1/tau0).
    R0 : float
        Förster radius.
    tau0 : float
        Donor lifetime without FRET.
    kappa2 : float
        Orientation factor.

    Returns
    -------
    dict
        ``{"R", "R0", "tau0", "kappa2", "sigma", "E", "tau_DA", "kFRET"}``
    """
    R = float(fretrate_to_distance(kFRET, R0, tau0, kappa2))
    E = float(distance_to_fret_efficiency(R, R0))
    tau_DA = fret_efficiency_to_lifetime(E, tau0)
    return {
        "R": R,
        "R0": R0,
        "tau0": tau0,
        "kappa2": kappa2,
        "sigma": 0.0,
        "E": E,
        "tau_DA": tau_DA,
        "kFRET": kFRET,
    }


# ── homoFRET ─────────────────────────────────────────────────────


def compute_homo_fret(
    t_RM: float,
    rho: float,
    tau0: float,
    R0: float,
) -> dict[str, float]:
    """Compute homoFRET exchange rate and effective distance.

    Parameters
    ----------
    t_RM : float
        Anisotropy relaxation time (same time units as *rho* and *tau0*).
    rho : float
        Rotational correlation time without homoFRET.
    tau0 : float
        Donor fluorescence lifetime.
    R0 : float
        Förster radius.

    Returns
    -------
    dict
        ``{"k_homo", "R_DA", "t_RM", "rho", "tau0", "R0"}``
    """
    if t_RM <= 0 or rho <= 0 or tau0 <= 0 or R0 <= 0:
        return {"k_homo": 0.0, "R_DA": np.nan, "t_RM": t_RM, "rho": rho, "tau0": tau0, "R0": R0}

    diff = (1.0 / t_RM) - (1.0 / rho)
    k_homo = max(0.0, 0.5 * diff)

    prod = k_homo * tau0
    if prod > 0.0:
        R_DA = float(R0 * prod ** (-1.0 / 6.0))
    else:
        R_DA = np.nan

    return {
        "k_homo": k_homo,
        "R_DA": R_DA,
        "t_RM": t_RM,
        "rho": rho,
        "tau0": tau0,
        "R0": R0,
    }


def compute_homo_fret_backmap(
    R_DA: float,
    R0: float,
    tau0: float,
    rho: float,
) -> dict[str, float]:
    """Back-map from effective homoFRET distance to anisotropy relaxation time.

    Parameters
    ----------
    R_DA : float
        Effective donor-acceptor distance from homoFRET.
    R0 : float
        Förster radius.
    tau0 : float
        Donor fluorescence lifetime.
    rho : float
        Rotational correlation time without homoFRET.

    Returns
    -------
    dict
        ``{"k_homo", "R_DA", "t_RM", "rho", "tau0", "R0"}``
    """
    if R_DA <= 0 or R0 <= 0 or tau0 <= 0 or rho <= 0:
        return {"k_homo": np.nan, "R_DA": R_DA, "t_RM": np.nan, "rho": rho, "tau0": tau0, "R0": R0}

    ratio = R0 / R_DA
    k_homo = (ratio ** 6) / tau0

    denom = (2.0 * k_homo) + (1.0 / rho)
    t_RM = 1.0 / denom if denom > 0 else np.nan

    return {
        "k_homo": k_homo,
        "R_DA": R_DA,
        "t_RM": t_RM,
        "rho": rho,
        "tau0": tau0,
        "R0": R0,
    }
