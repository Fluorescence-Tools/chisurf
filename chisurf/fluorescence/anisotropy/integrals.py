from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AnisotropyResult:
    r_e: float
    r_s: Optional[float]
    G: float
    chi: Optional[float]
    S_ges: Optional[float]


def perrin_steady_state_anisotropy(
        tau: float,
        rho: float,
        r0: float = 0.38,
) -> float:
    """Compute the Perrin steady-state anisotropy.

    Parameters
    ----------
    tau : float
        Fluorescence lifetime.
    rho : float
        Rotational correlation time.
    r0 : float, optional
        Fundamental anisotropy at time zero (default 0.38).

    Returns
    -------
    float
        The steady-state anisotropy.

    Raises
    ------
    ValueError
        If rho is non-positive.
    """
    tau = float(tau)
    rho = float(rho)
    if rho <= 0.0:
        raise ValueError("rho must be > 0")
    return float(r0 / (1.0 + tau / rho))


def compute_g_factor_isotropic(
        s_p,
        s_s,
        axis: Optional[int] = None,
) -> float:
    """Compute the G-factor assuming isotropic rotational diffusion.

    Parameters
    ----------
    s_p : array-like
        Parallel signal intensities.
    s_s : array-like
        Perpendicular signal intensities.
    axis : int, optional
        Axis along which to sum. If None, sums over all elements.

    Returns
    -------
    float
        The computed G-factor.

    Raises
    ------
    ZeroDivisionError
        If the sum of the parallel signal is zero.
    """
    sp = np.asarray(s_p, dtype=float)
    ss = np.asarray(s_s, dtype=float)
    num = float(np.sum(ss, axis=axis))
    den = float(np.sum(sp, axis=axis))
    if den == 0.0:
        raise ZeroDivisionError("Cannot compute G: sum(S_p) is zero")
    return num / den


def compute_g_factor_perrin(
        s_p,
        s_s,
        tau: float,
        rho: float,
        r0: float = 0.38,
        l1: float = 0.0,
        l2: float = 0.0,
        axis: Optional[int] = None,
) -> float:
    """Compute G from integrated channels using Perrin-corrected anisotropy.

    Uses Eq. (2.4-22) solved for G with r = r0 / (1 + tau/rho).
    """
    sp = float(np.sum(np.asarray(s_p, dtype=float), axis=axis))
    ss = float(np.sum(np.asarray(s_s, dtype=float), axis=axis))
    if sp == 0.0:
        raise ZeroDivisionError("Cannot compute G: sum(S_p) is zero")
    r = perrin_steady_state_anisotropy(tau=tau, rho=rho, r0=r0)
    den = sp * (1.0 - r * (1.0 - 3.0 * float(l2)))
    if den == 0.0:
        raise ZeroDivisionError("Cannot compute G: denominator is zero")
    num = ss * (1.0 + r * (2.0 - 3.0 * float(l1)))
    return float(num / den)


def anisotropy_from_integrals(
        s_p,
        s_s,
        G: float,
        l1: float = 0.0,
        l2: float = 0.0,
        gamma: float = 0.0,
        B_p: float = 0.0,
        B_s: float = 0.0,
        axis: Optional[int] = None,
        scatter_corrected: bool = False,
) -> AnisotropyResult:
    """Compute anisotropy from integrated signal intensities.

    Parameters
    ----------
    s_p : array-like
        Parallel signal intensities.
    s_s : array-like
        Perpendicular signal intensities.
    G : float
        G-factor correction for detection sensitivity.
    l1 : float, optional
        Mixing factor for parallel channel (default 0.0).
    l2 : float, optional
        Mixing factor for perpendicular channel (default 0.0).
    gamma : float, optional
        Scatter correction factor (default 0.0).
    B_p : float, optional
        Background for parallel channel (default 0.0).
    B_s : float, optional
        Background for perpendicular channel (default 0.0).
    axis : int, optional
        Axis along which to sum signal arrays.
    scatter_corrected : bool, optional
        If True, apply scatter correction (default False).

    Returns
    -------
    AnisotropyResult
        Dataclass containing computed anisotropy values.

    Raises
    ------
    ValueError
        If G is non-positive.
    ZeroDivisionError
        If the anisotropy denominator is zero.
    """
    g = float(G)
    if g <= 0.0:
        raise ValueError("G must be > 0")

    sp = float(np.sum(np.asarray(s_p, dtype=float), axis=axis))
    ss = float(np.sum(np.asarray(s_s, dtype=float), axis=axis))

    num_e = g * sp - ss
    den_e = (1.0 - 3.0 * float(l2)) * g * sp + (2.0 - 3.0 * float(l1)) * ss
    if den_e == 0.0:
        raise ZeroDivisionError("Denominator in r_E is zero")
    r_e = num_e / den_e

    if not scatter_corrected:
        return AnisotropyResult(r_e=float(r_e), r_s=None, G=g, chi=None, S_ges=None)

    den_chi = g * float(B_p) + 2.0 * float(B_s)
    if den_chi == 0.0:
        raise ZeroDivisionError("Denominator in chi is zero")
    chi = (2.0 * float(B_s)) / den_chi
    s_ges = g * sp + 2.0 * ss

    num_s = (g * sp - ss) - float(gamma) * (1.0 - 1.5 * chi) * s_ges
    den_s = (
            (1.0 - 3.0 * float(l2)) * g * sp
            + (2.0 - 3.0 * float(l1)) * ss
            - float(gamma) * (1.0 - 3.0 * float(l2) - 1.5 * chi * (1.0 - 2.0 * float(l2))) * s_ges
    )
    if den_s == 0.0:
        raise ZeroDivisionError("Denominator in r_S is zero")
    r_s = num_s / den_s
    return AnisotropyResult(r_e=float(r_e), r_s=float(r_s), G=g, chi=float(chi), S_ges=float(s_ges))
