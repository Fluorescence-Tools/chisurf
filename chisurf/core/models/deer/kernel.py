from __future__ import annotations

"""Native DEER/PELDOR dipolar physics (Qt-free, numpy/scipy only).

This is a self-contained reimplementation of the small subset of the dipolar
forward model needed to fit 4-pulse DEER time traces inside ChiSurf. Nothing
here depends on any external EPR-analysis package.

Conventions
-----------
* Time ``t`` is in microseconds (µs).
* Distance ``r`` is in **Ångström (Å)** — matching ChiSurf's FRET convention.
* The dipolar angular frequency is ``omega_dd(r) = D / r**3`` with
  ``D = 2*pi * 52.04e3`` rad·µs⁻¹·Å³ (nitroxide/electron gyromagnetic constant;
  the familiar ``2*pi * 52.04`` nm³ value scaled by ``1 nm³ = 1000 Å³``).

The intramolecular kernel for a powder-averaged spin pair is

    K(t, r) = integral_0^1 cos[(3*xi**2 - 1) * omega_dd(r) * t] dxi

which has the closed Fresnel form implemented in :func:`dipolar_kernel`.
"""

import numpy as np
from scipy import special
from scipy.integrate import trapezoid

#: Dipolar constant ``D`` in rad·µs⁻¹·Å³ (``2*pi * 52.04`` nm³ · 1000 Å³/nm³).
DIPOLAR_CONSTANT = 2.0 * np.pi * 52.04e3


def dipolar_kernel(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Return the powder-averaged dipolar kernel matrix ``K[i, j] = K(t_i, r_j)``.

    Parameters
    ----------
    t : numpy.ndarray
        Time axis in microseconds, shape ``(nt,)``.
    r : numpy.ndarray
        Distance axis in Ångström, shape ``(nr,)``.

    Returns
    -------
    numpy.ndarray
        Kernel of shape ``(nt, nr)``. ``K(0, r) = 1`` for every ``r``.

    Notes
    -----
    Uses the Fresnel closed form of the powder average::

        theta = omega_dd(r) * |t|
        z     = sqrt(6 * theta / pi)
        K     = sqrt(pi / (6*theta)) * (cos(theta)*C(z) + sin(theta)*S(z))

    where ``C`` and ``S`` are the Fresnel integrals
    (``scipy.special.fresnel`` with the ``cos(pi/2 u**2)`` normalisation).
    """
    t = np.asarray(t, dtype=float).reshape(-1, 1)
    r = np.asarray(r, dtype=float).reshape(1, -1)

    omega = DIPOLAR_CONSTANT / np.clip(np.abs(r), 1e-6, None) ** 3
    theta = omega * np.abs(t)  # (nt, nr), >= 0

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.sqrt(6.0 * theta / np.pi)
        ssa, csa = special.fresnel(z)  # scipy returns (S, C)
        pre = np.sqrt(np.pi / (6.0 * theta))
        k = pre * (np.cos(theta) * csa + np.sin(theta) * ssa)

    # Limit theta -> 0 gives K -> 1 (numerically 0/0 above).
    k = np.where(theta < 1e-12, 1.0, k)
    return k


def _normalize(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Normalise a distance distribution to unit area on grid ``r``."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    area = trapezoid(p, r)
    if area > 0:
        p = p / area
    return p


def dd_gauss(r: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Single normalised Gaussian distance distribution ``P(r)``.

    Parameters
    ----------
    r : numpy.ndarray
        Distance axis (Å).
    mean : float
        Centre distance ``r0`` (Å).
    sigma : float
        Standard deviation (Å).
    """
    r = np.asarray(r, dtype=float)
    sigma = max(float(sigma), 1e-6)
    p = np.exp(-0.5 * ((r - float(mean)) / sigma) ** 2)
    return _normalize(p, r)


def dd_gauss_multi(r: np.ndarray, means, sigmas, amplitudes) -> np.ndarray:
    """Sum of Gaussians ``P(r) = sum_i a_i * N(r; mean_i, sigma_i)``.

    The result is renormalised to unit area; the relative ``amplitudes`` set the
    weight of each component.
    """
    r = np.asarray(r, dtype=float)
    p = np.zeros_like(r)
    for m, s, a in zip(means, sigmas, amplitudes):
        p = p + max(float(a), 0.0) * dd_gauss(r, m, s)
    return _normalize(p, r)


def dd_rice(r: np.ndarray, nu: float, sigma: float) -> np.ndarray:
    """3D Rice/Rician distance distribution ``P(r)``.

    Parameters
    ----------
    r : numpy.ndarray
        Distance axis (Å).
    nu : float
        Location parameter ``nu`` (Å).
    sigma : float
        Spread parameter ``sigma`` (Å).
    """
    r = np.asarray(r, dtype=float)
    sigma = max(float(sigma), 1e-6)
    nu = max(float(nu), 0.0)
    # i0e(x) = exp(-|x|) * I0(x): keeps the product finite for large argument.
    arg = r * nu / sigma ** 2
    p = (r / sigma ** 2) * np.exp(-(r ** 2 + nu ** 2) / (2.0 * sigma ** 2) + np.abs(arg)) * special.i0e(arg)
    p = np.where(r > 0, p, 0.0)
    return _normalize(p, r)


def background(t: np.ndarray, model: str = "hom3d", k: float = 0.05, d: float = 3.0) -> np.ndarray:
    """Intermolecular background decay ``B(t)``.

    Parameters
    ----------
    t : numpy.ndarray
        Time axis (µs).
    model : str
        One of ``'hom3d'``/``'exp'`` (``exp(-k|t|)``) or ``'strexp'``
        (``exp(-(k|t|)**(d/3))``). Homogeneous-3D is exactly a mono-exponential
        in ``|t|``.
    k : float
        Decay rate (µs⁻¹).
    d : float
        Fractal dimension for the stretched-exponential model (``d=3`` ->
        mono-exponential).
    """
    at = np.abs(np.asarray(t, dtype=float))
    k = max(float(k), 0.0)
    if model == "strexp":
        return np.exp(-((k * at) ** (max(float(d), 1e-3) / 3.0)))
    # 'hom3d', 'exp', or unknown -> homogeneous-3D mono-exponential
    return np.exp(-k * at)


def deer_signal(
    t: np.ndarray,
    r: np.ndarray,
    p_r: np.ndarray,
    mod_depth: float,
    bg_model: str = "hom3d",
    bg_k: float = 0.05,
    bg_d: float = 3.0,
    scale: float = 1.0,
    kernel: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the full 4pDEER time-domain signal ``V(t)``.

    ``V(t) = scale * [(1 - lam) + lam * (K @ P)] * B(t)`` for a single dominant
    dipolar pathway with modulation depth ``lam = mod_depth``.

    Parameters
    ----------
    t : numpy.ndarray
        Time axis (µs), already shifted so ``t = 0`` is the dipolar zero time.
    r, p_r : numpy.ndarray
        Distance axis (Å) and (unnormalised) distribution; ``p_r`` is
        normalised to unit area internally.
    mod_depth : float
        Modulation depth ``lambda`` in ``[0, 1]``.
    bg_model, bg_k, bg_d : str, float, float
        Background specification, see :func:`background`.
    scale : float
        Overall amplitude scale (absorbs data normalisation).
    kernel : numpy.ndarray, optional
        Precomputed ``K(t, r)`` matrix; recomputed when ``None``.
    """
    r = np.asarray(r, dtype=float)
    p = _normalize(p_r, r)
    k_mat = dipolar_kernel(t, r) if kernel is None else kernel
    form_factor = trapezoid(k_mat * p.reshape(1, -1), r, axis=1)
    lam = float(np.clip(mod_depth, 0.0, 1.0))
    intra = (1.0 - lam) + lam * form_factor
    b = background(t, bg_model, bg_k, bg_d)
    return float(scale) * intra * b
