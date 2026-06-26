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
    lifetime_to_fret_efficiency,
)

# ── heteroFRET ───────────────────────────────────────────────────


def distance_distribution(
    mean: float,
    sigma: float,
    distribution: str = "gaussian",
    n_points: int = 64,
    m_sigma: float | None = None,
    bins: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(distances, weights)`` for a donor-acceptor distance distribution.

    Two physical models are supported:

    ``"gaussian"``
        A normal distribution ``N(mean, sigma)`` over the distance. This is the
        classical model but is unphysical near contact: it is symmetric about
        ``mean`` and assigns weight to (clamped) near-zero / negative distances.

    ``"chi"``
        The radial distribution of an isotropic 3D Gaussian whose centre is at
        ``mean`` (a non-central chi distribution with 3 degrees of freedom)::

            p(r) ∝ r * [exp(-(r-mean)²/2σ²) - exp(-(r+mean)²/2σ²)],   r ≥ 0

        It is strictly non-negative and vanishes as ``r²`` at contact, so it is
        the appropriate choice for distances — especially around zero distance.

    The returned weights are normalised to sum to one.
    """
    s = max(float(sigma), 1e-9)
    if m_sigma is None:
        # The Gaussian keeps its historical ±1.5σ window (and stays away from the
        # near-zero region where FRET rates diverge); the chi distribution uses a
        # wider window so its skewed tail is captured.
        m_sigma = 3.5 if distribution == "chi" else 1.5
    if bins is None:
        # Sample where the distribution carries weight. For plotting a fixed,
        # wider grid can be supplied via ``bins`` so the curve is shown in context.
        if distribution == "chi":
            r_max = float(mean) + m_sigma * s
            bins = np.linspace(1e-9, max(r_max, 1e-6), n_points)
        else:
            g_min = max(1e-9, float(mean) - m_sigma * s)
            g_max = float(mean) + m_sigma * s
            bins = np.linspace(g_min, g_max, n_points)
    else:
        bins = np.asarray(bins, dtype=float)

    if distribution == "chi":
        p = bins * (
            np.exp(-((bins - mean) ** 2) / (2.0 * s * s))
            - np.exp(-((bins + mean) ** 2) / (2.0 * s * s))
        )
        p = np.clip(p, 0.0, None)
    else:
        p = np.exp(-((bins - mean) ** 2) / (2.0 * s * s))
    total = float(np.sum(p))
    if total > 0:
        p = p / total
    return bins, p


def compute_fret_from_distance(
    R: float,
    R0: float,
    tau0: float,
    kappa2: float = 0.667,
    sigma: float = 0.0,
    distribution: str = "gaussian",
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
        Width of the distance distribution.  When *sigma* > 0 the returned
        efficiency and rate are distribution-averaged values.
    distribution : str
        ``"gaussian"`` (default) or ``"chi"`` (3D non-central chi, non-negative
        and physical near zero distance). See :func:`distance_distribution`.

    Returns
    -------
    dict
        ``{"R", "R0", "tau0", "kappa2", "sigma", "distribution",
        "E", "tau_DA", "kFRET"}``
    """
    if sigma > 0:
        bins, weights = distance_distribution(R, sigma, distribution)
        rate_values = distance_to_fret_rate_constant(bins, R0, tau0, kappa2)
        eff = rate_values * tau0 / (1.0 + rate_values * tau0)
        avg_E = float(np.sum(weights * eff))
        tau_DA = fret_efficiency_to_lifetime(avg_E, tau0)
        # Effective FRET rate consistent with the distribution-averaged
        # efficiency/lifetime. (A naive mean of per-distance rates diverges,
        # because the rate ~1/r^6 blows up as the distribution approaches
        # contact — both for Gaussian-with-clamp and for the chi distribution.)
        kFRET = avg_E / ((1.0 - avg_E) * tau0) if avg_E < 1.0 else float("inf")
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
        "distribution": distribution,
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
