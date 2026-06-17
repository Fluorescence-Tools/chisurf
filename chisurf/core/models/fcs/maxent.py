from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


@dataclass
class FCSMaxEntLCurveResult:
    """Result of an FCS MaxEnt L-curve sweep.

    Parameters
    ----------
    log10_reg : numpy.ndarray
        Log10 regularization values used for the sweep.
    reg : numpy.ndarray
        Linear regularization values used for the sweep.
    chi2r : numpy.ndarray
        Reduced chi-squared value at each regularization point.
    solution_norm : numpy.ndarray
        Euclidean norm of the MaxEnt distribution at each point.
    corner_index : int or None
        Automatically detected L-curve corner index into the arrays.
    results : tuple[dict, ...]
        Per-point MaxEnt result dictionaries when ``return_results=True`` was
        requested, otherwise an empty tuple.
    """

    log10_reg: np.ndarray
    reg: np.ndarray
    chi2r: np.ndarray
    solution_norm: np.ndarray
    corner_index: int | None
    results: tuple[dict, ...] = ()


def build_diffusion_kernel(
        tau: np.ndarray,
        td_grid: np.ndarray,
        s: float = 3.5,
) -> np.ndarray:
    """Build a simple 3D Gaussian FCS diffusion kernel.

    Each column ``k[:, i]`` contains the normalized correlation shape for a
    diffusion time ``td_grid[i]``. The overall amplitude (``1/N``) and any
    constant offset are *not* included and should be handled separately.

    Parameters
    ----------
    tau : array_like
        Correlation lag times.
    td_grid : array_like
        Diffusion times that define the MaxEnt grid.
    s : float, optional
        Axial-to-radial waist ratio ``z0 / w0`` in the 3D Gaussian model.

    Returns
    -------
    numpy.ndarray
        Kernel matrix of shape ``(tau.size, td_grid.size)``.

    Examples
    --------
    Build a small kernel and check its shape:

    >>> tau = np.logspace(-3, 1, 4)
    >>> td = np.logspace(-2, 0, 3)
    >>> k = build_diffusion_kernel(tau, td, s=3.5)
    >>> k.shape
    (4, 3)
    """
    tau = np.asarray(tau, dtype=float)
    td_grid = np.asarray(td_grid, dtype=float)
    k = np.empty((tau.size, td_grid.size), dtype=float)
    for j, td in enumerate(td_grid):
        # Standard 3D Gaussian volume FCS model without offset and 1/N factor
        k[:, j] = (1.0 / (1.0 + tau / td)) / np.sqrt(1.0 + (tau / td) / (s ** 2))
    return k


@njit(cache=True)
def _quickfit_mem_iteration_numba(
        Vred: np.ndarray,
        svals: np.ndarray,
        Ured: np.ndarray,
        M: np.ndarray,
        stdev: np.ndarray,
        ydata: np.ndarray,
        m_prior: np.ndarray,
        alpha: float,
        num_iter: int,
):
    """Numba-accelerated MaxEnt iteration loop (QuickFit-style MEM).

    Parameters
    ----------
    Vred : np.ndarray
        Truncated left singular vectors (Nd x s).
    svals : np.ndarray
        Truncated singular values (s,).
    Ured : np.ndarray
        Truncated right singular vectors (N x s).
    M : np.ndarray
        Curvature matrix (s x s) built from singular values and weights.
    stdev : np.ndarray
        Data standard deviations (Nd,).
    ydata : np.ndarray
        Measured data vector (Nd,) after baseline subtraction.
    m_prior : np.ndarray
        Prior distribution on the grid (N,).
    alpha : float
        Entropy regularization strength.
    num_iter : int
        Number of iterations.

    Returns
    -------
    f : np.ndarray
        Final MaxEnt distribution (N,).
    F : np.ndarray
        Reconstructed data vector (Nd,).
    """
    N = Ured.shape[0]
    s = svals.shape[0]

    # Ensure a positive, normalized prior m on the distribution grid
    m = m_prior.copy()
    for i in range(N):
        if m[i] <= 0.0:
            m[i] = 1.0
    total = 0.0
    for i in range(N):
        total += m[i]
    if total <= 0.0:
        total = float(N)
    inv_total = 1.0 / total
    for i in range(N):
        m[i] *= inv_total

    stdev2 = stdev * stdev

    u = np.zeros(s, dtype=np.float64)
    f = np.zeros(N, dtype=np.float64)
    eye = np.eye(s, dtype=np.float64)
    max_exponent = 100.0

    for _ in range(num_iter):
        # Current distribution: f = m * exp(Ured @ u)
        work = Ured @ u
        for i in range(N):
            if work[i] > max_exponent:
                work[i] = max_exponent
            elif work[i] < -max_exponent:
                work[i] = -max_exponent
        f = m * np.exp(work)
        for i in range(N):
            if not np.isfinite(f[i]) or f[i] < 0.0:
                f[i] = 0.0

        # K = Ured^T diag(f) Ured
        K = Ured.T @ (f.reshape(N, 1) * Ured)

        # Model data: F = Vred * Sred * Ured^T * f
        tmp_s = Ured.T @ f
        tmp_s = svals * tmp_s
        F = Vred @ tmp_s

        res = F - ydata
        work5 = res / stdev2
        tmp2 = Vred.T @ work5
        gvec = svals * tmp2

        A = alpha * eye + M @ K
        b = -alpha * u - gvec
        du = np.linalg.solve(A, b)
        u = u + du

    # Final distribution and model curve
    work = Ured @ u
    for i in range(N):
        if work[i] > max_exponent:
            work[i] = max_exponent
        elif work[i] < -max_exponent:
            work[i] = -max_exponent
    f = m * np.exp(work)
    tmp_s = Ured.T @ f
    tmp_s = svals * tmp_s
    F = Vred @ tmp_s

    # Enforce finite, non-negative distribution
    for i in range(N):
        if not np.isfinite(f[i]) or f[i] < 0.0:
            f[i] = 0.0

    return f, F


def fcs_maxent(
        tau: np.ndarray,
        g: np.ndarray,
        td_min: float | None = None,
        td_max: float | None = None,
        n_td: int = 80,
        s: float = 3.5,
        reg: float = 0.1,
        weights: np.ndarray | None = None,
        prior: np.ndarray | None = None,
        td_grid: np.ndarray | None = None,
        **kwargs,
) -> dict:
    r"""Run a simple MaxEnt inversion on an FCS correlation curve.

    Parameters
    ----------
    tau : array_like
        Correlation lag times (same units as ``td_min`` / ``td_max``).
    g : array_like
        Measured correlation amplitudes :math:`G(\tau)`.
    td_min, td_max : float, optional
        Minimum and maximum diffusion times for the grid. If omitted,
        they are estimated from the ``tau`` range.
    n_td : int, optional
        Number of diffusion-time grid points (log-spaced) when
        ``td_grid`` is not supplied.
    s : float, optional
        Axial-to-radial waist ratio ``z0 / w0`` in the 3D Gaussian volume
        model used to build the kernel.
    reg : float, optional
        Entropy regularization weight (``alpha`` in the underlying solver).
    weights : array_like, optional
        Data weights (same length as ``tau``). If *None*, all ones are used.
        Internally these are converted into standard deviations.
    prior : array_like, optional
        Prior distribution on the diffusion-time grid. If *None*, a uniform
        prior is used.
    td_grid : array_like, optional
        Explicit diffusion-time grid. If given, ``td_min``, ``td_max`` and
        ``n_td`` are ignored.
    **kwargs
        Additional options forwarded to the internal solver. For historical
        reasons ``regularization_factor`` is accepted as an alias for
        ``reg``. The maximum number of iterations can be controlled via
        ``num_iter`` or ``max_iter``.

    Returns
    -------
    dict
        Dictionary with keys ``"tau"``, ``"g"``, ``"g_fit"``,
        ``"td_grid"`` and ``"p"`` (the MaxEnt distribution).

    Examples
    --------
    Perform a tiny MaxEnt inversion on a synthetic single-component curve:

    >>> tau = np.logspace(-3, 1, 16)
    >>> g = np.exp(-tau / 0.05)
    >>> result = fcs_maxent(tau, g, n_td=8, reg=0.1, num_iter=5)
    >>> sorted(result.keys())
    ['g', 'g_fit', 'p', 'tau', 'td_grid']
    >>> result['g_fit'].shape == g.shape
    True
    >>> result['td_grid'].shape == result['p'].shape
    True
    """
    tau = np.asarray(tau, dtype=float).ravel()
    g = np.asarray(g, dtype=float).ravel()
    if tau.size != g.size:
        raise ValueError("tau and g must have the same length")

    # Backwards compatibility: allow callers to still pass
    # ``regularization_factor=...`` and map it onto ``reg``.
    if "regularization_factor" in kwargs:
        reg = float(kwargs.pop("regularization_factor"))

    # Mode switch: in QuickFit, the correlation passed to MEM typically decays
    # to 0 (offset-subtracted), while in ChiSurf FCS curves are usually stored
    # with a baseline of ~1. The 'decay_to_one' flag controls whether we
    # internally subtract/add a baseline around the MaxEnt inversion.
    decay_to_one = bool(kwargs.pop("decay_to_one", True))

    baseline_kw = kwargs.pop("baseline", None)
    if baseline_kw is not None:
        baseline = float(baseline_kw)
    else:
        baseline = 1.0 if decay_to_one else 0.0

    g_data = g - baseline

    if td_grid is not None:
        td_grid = np.asarray(td_grid, dtype=float).ravel()
        if td_grid.size == 0:
            raise ValueError("td_grid must not be empty")
    else:
        if td_min is None:
            td_min = max(np.min(tau) * 1e-2, tau[0] * 1e-1 if tau[0] > 0 else 1e-6)
        if td_max is None:
            td_max = np.max(tau) * 1e2

        td_grid = np.logspace(np.log10(td_min), np.log10(td_max), n_td)

    # Build kernel (forward operator)
    A = build_diffusion_kernel(tau, td_grid, s=s)

    # Map weights -> standard deviations similar to QuickFit's implementation.
    # In the GUI, weights are typically provided as ~1/sigma, so invert here.
    if weights is None:
        stdev = np.ones_like(g_data, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != g.size:
            raise ValueError("weights must have same length as g")
        stdev = np.empty_like(w)
        tiny = 1e-12
        for i in range(w.size):
            val = w[i]
            if np.isfinite(val) and abs(val) > tiny:
                stdev[i] = 1.0 / val
            else:
                stdev[i] = 1.0

    # Clamp and renormalize stdev to have an average of ~1, as in QuickFit.
    stdev = np.clip(np.abs(stdev), 1e-3, 1e3)
    scale = stdev.size / np.sum(stdev)
    stdev = stdev * scale

    # Prior on the diffusion-time grid: use provided prior or uniform.
    if prior is None:
        m_prior = np.ones(td_grid.size, dtype=float)
    else:
        m_prior = np.asarray(prior, dtype=float).ravel()
        if m_prior.size != td_grid.size:
            raise ValueError("prior must have length n_td")

    # SVD of kernel A (Nd x N) -> T = V * S * U^T (QuickFit notation).
    # NumPy returns A = U_np * S * VT, where U_np corresponds to V and
    # VT.T corresponds to U in the QuickFit code.
    U_np, svals, VT = np.linalg.svd(A, full_matrices=False)
    if svals.size == 0 or svals[0] <= 0.0:
        raise RuntimeError("SVD of kernel failed or produced no singular values")

    # Truncate singular space as in MaxEntB040: keep svals >= svals[0]/1e5.
    thresh = svals[0] / 100000.0
    mask = svals >= thresh
    if not np.any(mask):
        mask[0] = True
    s_count = int(mask.sum())

    svals_red = svals[:s_count].astype(np.float64)
    Vred = U_np[:, :s_count].astype(np.float64)      # Nd x s
    Ured = VT[:s_count, :].T.astype(np.float64)      # N x s

    # Build M = Sred^T * Vred^T * Sigma^{-1} * Vred * Sred, where
    # Sigma^{-1} = diag(1 / stdev^2).
    inv_sigma2 = 1.0 / (stdev.astype(np.float64) ** 2)
    VW = Vred * inv_sigma2[:, None]
    M_pre = Vred.T @ VW
    M = (svals_red[:, None] * M_pre) * svals_red[None, :]

    # Number of MaxEnt iterations (QuickFit-style).
    num_iter = int(kwargs.pop("num_iter", kwargs.pop("max_iter", 200)))
    alpha = float(reg)

    p, g_fit_data = _quickfit_mem_iteration_numba(
        Vred,
        svals_red,
        Ured,
        M,
        stdev.astype(np.float64),
        g_data.astype(np.float64),
        m_prior.astype(np.float64),
        alpha,
        num_iter,
    )

    g_fit = g_fit_data + baseline

    return {
        "tau": tau,
        "g": g,
        "g_fit": g_fit,
        "td_grid": td_grid,
        "p": p,
    }


def _rh_grid_to_td_grid(
        rh_grid: np.ndarray,
        w0_um: float,
        temperature: float = 298.15,
        viscosity: float = 1.0e-3,
) -> np.ndarray:
    """Convert a hydrodynamic-radius grid to a diffusion-time grid.

    Parameters
    ----------
    rh_grid : array_like
        Hydrodynamic radii in nanometers.
    w0_um : float
        Lateral waist radius of the observation volume in micrometers.
    temperature : float, optional
        Temperature in Kelvin.
    viscosity : float, optional
        Dynamic viscosity of the medium in Pa·s.

    Returns
    -------
    numpy.ndarray
        Diffusion times in milliseconds corresponding to ``rh_grid``.

    Examples
    --------
    >>> td = _rh_grid_to_td_grid(np.array([1.0, 10.0]), w0_um=0.3)
    >>> td.shape
    (2,)
    """
    rh = np.asarray(rh_grid, dtype=float).ravel()
    w0_m = float(w0_um) * 1.0e-6
    k_B = 1.380649e-23
    eta = float(viscosity)
    T = float(temperature)

    td = np.empty_like(rh, dtype=float)
    for i in range(rh.size):
        r_m = rh[i] * 1.0e-9
        if r_m <= 0.0 or not np.isfinite(r_m):
            td[i] = np.nan
            continue
        # Diffusion coefficient D in m^2/s from Einstein–Stokes
        D = k_B * T / (6.0 * np.pi * eta * r_m)
        # Characteristic diffusion time in seconds, then convert to ms to
        # match the FCS convention for t_c.
        td_s = w0_m * w0_m / (4.0 * D)
        td[i] = td_s * 1.0e3
    return td


def _water_viscosity_Pa_s(temperature: float) -> float:
    """Return dynamic viscosity of water (Pa·s) as a function of T [K].

    Uses the same empirical relation as ``water_viscosity_Pa_s`` in the FCS
    calculator plugin (Kapusta 2010 PicoQuant app note):

        eta(T) = A * 10**(B / (T - C)),

    with A = 2.414e-5 Pa·s, B = 247.8 K, C = 140 K.

    Examples
    --------
    The value at room temperature is on the order of 1 mPa·s:

    >>> round(_water_viscosity_Pa_s(298.15), 4)
    0.0009
    """
    A, B, C = 2.414e-5, 247.8, 140.0
    T = float(temperature)
    if T <= C:
        T = C + 1.0
    return A * 10.0 ** (B / (T - C))


def fcs_maxent_rh(
        tau: np.ndarray,
        g: np.ndarray,
        rh_min: float = 0.5,
        rh_max: float = 50.0,
        n_rh: int = 64,
        w0: float = 0.3,
        s: float = 3.5,
        reg: float = 0.1,
        temperature: float = 298.15,
        viscosity: float | None = None,
        weights: np.ndarray | None = None,
        prior: np.ndarray | None = None,
        **kwargs,
) -> dict:
    r"""Run a MaxEnt inversion parameterized in hydrodynamic radius.

    This convenience wrapper constructs a diffusion-time grid from a grid of
    hydrodynamic radii using the Einstein–Stokes relation and then calls
    :func:`fcs_maxent`.

    Parameters
    ----------
    tau : array_like
        Correlation lag times.
    g : array_like
        Measured correlation amplitudes :math:`G(\tau)`.
    rh_min, rh_max : float, optional
        Minimum and maximum hydrodynamic radius in nanometers.
    n_rh : int, optional
        Number of grid points between ``rh_min`` and ``rh_max``.
    w0 : float, optional
        Lateral beam waist in micrometers used for the diffusion-time
        conversion.
    s : float, optional
        Axial-to-radial waist ratio passed to :func:`build_diffusion_kernel`.
    reg : float, optional
        Entropy regularization weight for :func:`fcs_maxent`.
    temperature : float, optional
        Temperature in Kelvin.
    viscosity : float, optional
        Dynamic viscosity in Pa·s. If *None*, the viscosity of water at the
        given temperature is used.
    weights, prior : array_like, optional
        Forwarded to :func:`fcs_maxent`.
    **kwargs
        Additional keyword arguments forwarded to :func:`fcs_maxent`.

    Returns
    -------
    dict
        Dictionary returned by :func:`fcs_maxent` with an extra key
        ``"rh_grid"`` containing the hydrodynamic-radius grid.

    Examples
    --------
    >>> tau = np.logspace(-3, 1, 16)
    >>> g = np.exp(-tau / 0.1)
    >>> result = fcs_maxent_rh(tau, g, n_rh=8, w0=0.3, reg=0.1, num_iter=5)
    >>> sorted(result.keys())
    ['g', 'g_fit', 'p', 'rh_grid', 'tau', 'td_grid']
    >>> result['rh_grid'].shape == result['p'].shape
    True
    """
    tau = np.asarray(tau, dtype=float).ravel()
    g = np.asarray(g, dtype=float).ravel()
    if tau.size != g.size:
        raise ValueError("tau and g must have the same length")

    rh_min_val = max(float(rh_min), 1.0e-3)
    rh_max_val = max(float(rh_max), rh_min_val * 1.001)
    n_rh_val = int(n_rh) if int(n_rh) > 2 else 3
    rh_grid = np.logspace(np.log10(rh_min_val), np.log10(rh_max_val), n_rh_val)

    T = float(temperature)
    if viscosity is None:
        eta = _water_viscosity_Pa_s(T)
    else:
        eta = float(viscosity)

    td_grid = _rh_grid_to_td_grid(
        rh_grid,
        w0_um=float(w0),
        temperature=T,
        viscosity=eta,
    )

    result = fcs_maxent(
        tau=tau,
        g=g,
        td_min=None,
        td_max=None,
        n_td=n_rh_val,
        s=s,
        reg=reg,
        weights=weights,
        prior=prior,
        td_grid=td_grid,
        **kwargs,
    )

    result["rh_grid"] = rh_grid
    return result


def _lcurve_grid(
        n_points: int,
        log10_min: float,
        log10_max: float,
) -> np.ndarray:
    """Build the log10 regularization grid for an FCS MaxEnt L-curve.

    Parameters
    ----------
    n_points : int
        Number of grid points. Values below two are clamped to two.
    log10_min, log10_max : float
        Inclusive log10 regularization range.

    Returns
    -------
    numpy.ndarray
        One-dimensional log10 regularization grid.
    """
    n = max(int(n_points), 2)
    lo = float(log10_min)
    hi = float(log10_max)
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        return np.array([lo, hi], dtype=float)
    return np.linspace(lo, hi, n)


def _valid_y_error(y_error: Any, size: int) -> np.ndarray:
    """Return positive finite y errors for chi-squared evaluation.

    Parameters
    ----------
    y_error : array_like or None
        Experimental y standard deviations. ``None`` means unit weights.
    size : int
        Expected number of points.

    Returns
    -------
    numpy.ndarray
        Positive finite error array with length ``size``.
    """
    if y_error is None:
        return np.ones(size, dtype=float)
    errors = np.asarray(y_error, dtype=float).ravel()
    if errors.size != size:
        raise ValueError("y_error must have same length as g")
    errors = np.where(np.isfinite(errors) & (errors > 0.0), errors, 1.0)
    return errors


def _chi2r_from_arrays(
        g: np.ndarray,
        g_fit: np.ndarray,
        y_error: Any,
        xmin: int = 0,
        xmax: int | None = None,
        mask: np.ndarray | None = None,
        n_free: int = 0,
) -> float:
    """Compute reduced chi-squared from raw arrays.

    Parameters
    ----------
    g, g_fit : array_like
        Experimental and reconstructed FCS correlation curves.
    y_error : array_like or None
        Experimental y standard deviations.
    xmin, xmax : int, optional
        Fit-window index range.
    mask : array_like or None
        Boolean mask applied inside the fit window.
    n_free : int, optional
        Number of free parameters used to reduce chi-squared.

    Returns
    -------
    float
        Reduced chi-squared, or ``nan`` if the window is too small.
    """
    g = np.asarray(g, dtype=float).ravel()
    g_fit = np.asarray(g_fit, dtype=float).ravel()
    if g.size != g_fit.size:
        raise ValueError("g and g_fit must have same length")
    n = g.size
    if n == 0:
        return float("nan")
    x0 = max(int(xmin or 0), 0)
    if xmax is None:
        x1 = n
    else:
        x1 = min(max(int(xmax), x0), n)
    if x1 <= x0:
        return float("nan")
    idx = np.arange(x0, x1, dtype=int)
    if mask is not None:
        m = np.asarray(mask)
        if m.size == n:
            idx = idx[m[idx].astype(bool)]
        elif m.size == x1 - x0:
            idx = idx[m.astype(bool)]
    if idx.size <= max(int(n_free) + 1, 1):
        return float("nan")
    errors = _valid_y_error(y_error, n)
    wres = (g[idx] - g_fit[idx]) / errors[idx]
    dof = float(idx.size - int(n_free) - 1.0)
    if dof <= 0.0:
        return float("nan")
    return float(np.sum(wres * wres) / dof)


def _solution_norm(result: dict) -> float:
    """Return the Euclidean norm of an FCS MaxEnt distribution.

    Parameters
    ----------
    result : dict
        MaxEnt result dictionary containing a ``"p"`` distribution.

    Returns
    -------
    float
        Euclidean norm of ``result["p"]``.
    """
    p = np.asarray(result.get("p", []), dtype=float).ravel()
    if p.size == 0:
        return float("nan")
    return float(np.linalg.norm(p))


def _maxent_stdev_from_weights(
        weights: np.ndarray | None,
        size: int,
) -> np.ndarray:
    """Convert MaxEnt weights to normalized standard deviations.

    Parameters
    ----------
    weights : array_like or None
        Inverse standard deviations. ``None`` means unit weights.
    size : int
        Expected number of data points.

    Returns
    -------
    numpy.ndarray
        Normalized standard deviations.
    """
    if weights is None:
        stdev = np.ones(size, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != size:
            raise ValueError("weights must have same length as g")
        stdev = np.empty_like(w)
        tiny = 1e-12
        for i in range(w.size):
            val = w[i]
            if np.isfinite(val) and abs(val) > tiny:
                stdev[i] = 1.0 / val
            else:
                stdev[i] = 1.0
    stdev = np.clip(np.abs(stdev), 1e-3, 1e3)
    scale = stdev.size / np.sum(stdev)
    return stdev * scale


def _maxent_prior(prior: np.ndarray | None, size: int) -> np.ndarray:
    """Return a positive MaxEnt prior distribution.

    Parameters
    ----------
    prior : array_like or None
        Prior distribution. ``None`` means a uniform prior.
    size : int
        Expected number of grid points.

    Returns
    -------
    numpy.ndarray
        Prior distribution with length ``size``.
    """
    if prior is None:
        return np.ones(size, dtype=float)
    m_prior = np.asarray(prior, dtype=float).ravel()
    if m_prior.size != size:
        raise ValueError("prior must have length n_td")
    return m_prior


def _maxent_svd_components(
        A: np.ndarray,
        stdev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build truncated SVD components used by the MaxEnt solver.

    Parameters
    ----------
    A : numpy.ndarray
        Forward operator matrix.
    stdev : numpy.ndarray
        Normalized data standard deviations.

    Returns
    -------
    tuple of numpy.ndarray
        ``Vred``, ``svals_red``, ``Ured``, and ``M``.
    """
    U_np, svals, VT = np.linalg.svd(A, full_matrices=False)
    if svals.size == 0 or svals[0] <= 0.0:
        raise RuntimeError("SVD of kernel failed or produced no singular values")
    thresh = svals[0] / 100000.0
    mask = svals >= thresh
    if not np.any(mask):
        mask[0] = True
    s_count = int(mask.sum())
    svals_red = svals[:s_count].astype(np.float64)
    Vred = U_np[:, :s_count].astype(np.float64)
    Ured = VT[:s_count, :].T.astype(np.float64)
    inv_sigma2 = 1.0 / (stdev.astype(np.float64) ** 2)
    VW = Vred * inv_sigma2[:, None]
    M_pre = Vred.T @ VW
    M = (svals_red[:, None] * M_pre) * svals_red[None, :]
    return Vred, svals_red, Ured, M


def _maxent_td_grid(
        tau: np.ndarray,
        td_min: float | None,
        td_max: float | None,
        n_td: int,
) -> np.ndarray:
    """Build the diffusion-time grid used by FCS MaxEnt.

    Parameters
    ----------
    tau : numpy.ndarray
        Lag-time array used for automatic grid estimation.
    td_min, td_max : float or None
        Diffusion-time bounds.
    n_td : int
        Number of grid points.

    Returns
    -------
    numpy.ndarray
        Log-spaced diffusion-time grid.
    """
    if td_min is None:
        td_min = max(np.min(tau) * 1e-2, tau[0] * 1e-1 if tau[0] > 0 else 1e-6)
    if td_max is None:
        td_max = np.max(tau) * 1e2
    return np.logspace(np.log10(td_min), np.log10(td_max), n_td)


def _compute_fcs_maxent_l_curve(
        tau: np.ndarray,
        g: np.ndarray,
        *,
        y_error: Any = None,
        log10_min: float = -3.0,
        log10_max: float = 3.0,
        n_points: int = 32,
        xmin: int = 0,
        xmax: int | None = None,
        mask: np.ndarray | None = None,
        n_free: int = 0,
        solver: str = "td",
        solver_kwargs: dict[str, Any] | None = None,
        return_results: bool = False,
        on_error: str = "nan",
) -> FCSMaxEntLCurveResult:
    """Run an FCS MaxEnt L-curve sweep in core code.

    Parameters
    ----------
    tau, g : array_like
        Experimental lag times and correlation amplitudes.
    y_error : array_like or None, optional
        Experimental y standard deviations. ``None`` means unit weights.
    log10_min, log10_max : float, optional
        Inclusive log10 regularization range.
    n_points : int, optional
        Number of regularization points.
    xmin, xmax : int, optional
        Fit-window index range used for chi-squared reduction.
    mask : array_like or None, optional
        Boolean mask applied inside the fit window.
    n_free : int, optional
        Number of free parameters used to reduce chi-squared.
    solver : {"td", "rh"}, optional
        Select diffusion-time or hydrodynamic-radius MaxEnt solver.
    solver_kwargs : dict, optional
        Keyword arguments forwarded to the selected MaxEnt solver.
    return_results : bool, optional
        If True, store each per-point MaxEnt result dictionary.
    on_error : {"nan", "raise"}, optional
        How to handle a failed MaxEnt point.

    Returns
    -------
    FCSMaxEntLCurveResult
        L-curve arrays and optional per-point results.
    """
    tau_arr = np.asarray(tau, dtype=float).ravel()
    g_arr = np.asarray(g, dtype=float).ravel()
    if tau_arr.size != g_arr.size:
        raise ValueError("tau and g must have same length")
    if tau_arr.size == 0:
        return FCSMaxEntLCurveResult(
            log10_reg=np.array([], dtype=float),
            reg=np.array([], dtype=float),
            chi2r=np.array([], dtype=float),
            solution_norm=np.array([], dtype=float),
            corner_index=None,
            results=(),
        )

    grid = _lcurve_grid(n_points, log10_min, log10_max)
    chi2_vals = np.empty_like(grid, dtype=float)
    sol_vals = np.empty_like(grid, dtype=float)
    results: list[dict] = [] if return_results else []
    kwargs = dict(solver_kwargs or {})
    if y_error is not None and "weights" not in kwargs:
        kwargs["weights"] = 1.0 / _valid_y_error(y_error, tau_arr.size)

    try:
        if solver == "td":
            td_min = kwargs.pop("td_min", None)
            td_max = kwargs.pop("td_max", None)
            n_td = int(kwargs.pop("n_td", 80))
            s = float(kwargs.pop("s", 3.5))
            baseline = float(kwargs.pop("baseline", 1.0))
            prior = kwargs.pop("prior", None)
            td_grid = kwargs.pop("td_grid", None)
            if td_grid is None:
                td_grid = _maxent_td_grid(tau_arr, td_min, td_max, n_td)
            else:
                td_grid = np.asarray(td_grid, dtype=float).ravel()
            A = build_diffusion_kernel(tau_arr, td_grid, s=s)
            stdev = _maxent_stdev_from_weights(kwargs.get("weights"), g_arr.size)
            Vred, svals_red, Ured, M = _maxent_svd_components(A, stdev)
            m_prior = _maxent_prior(prior, td_grid.size)
            g_data = g_arr - baseline
            for i, log10_reg in enumerate(grid):
                alpha = 10.0 ** float(log10_reg)
                num_iter = int(kwargs.get("num_iter", kwargs.get("max_iter", 200)))
                p, g_fit_data = _quickfit_mem_iteration_numba(
                    Vred,
                    svals_red,
                    Ured,
                    M,
                    stdev.astype(np.float64),
                    g_data.astype(np.float64),
                    m_prior.astype(np.float64),
                    alpha,
                    num_iter,
                )
                result = {
                    "tau": tau_arr,
                    "g": g_arr,
                    "g_fit": g_fit_data + baseline,
                    "td_grid": td_grid,
                    "p": p,
                }
                chi2_vals[i] = _chi2r_from_arrays(
                    g_arr,
                    result["g_fit"],
                    y_error=y_error,
                    xmin=xmin,
                    xmax=xmax,
                    mask=mask,
                    n_free=n_free,
                )
                sol_vals[i] = _solution_norm(result)
                if return_results:
                    results.append(result)
        elif solver == "rh":
            rh_min = float(kwargs.pop("rh_min", 0.5))
            rh_max = float(kwargs.pop("rh_max", 50.0))
            n_rh = int(kwargs.pop("n_rh", 64))
            w0 = float(kwargs.pop("w0", 0.3))
            s = float(kwargs.pop("s", 3.5))
            baseline = float(kwargs.pop("baseline", 1.0))
            temperature = float(kwargs.pop("temperature", 298.15))
            viscosity = kwargs.pop("viscosity", None)
            prior = kwargs.pop("prior", None)
            rh_min_val = max(rh_min, 1.0e-3)
            rh_max_val = max(rh_max, rh_min_val * 1.001)
            n_rh_val = n_rh if n_rh > 2 else 3
            rh_grid = np.logspace(np.log10(rh_min_val), np.log10(rh_max_val), n_rh_val)
            eta = _water_viscosity_Pa_s(temperature) if viscosity is None else float(viscosity)
            td_grid = _rh_grid_to_td_grid(
                rh_grid,
                w0_um=w0,
                temperature=temperature,
                viscosity=eta,
            )
            A = build_diffusion_kernel(tau_arr, td_grid, s=s)
            stdev = _maxent_stdev_from_weights(kwargs.get("weights"), g_arr.size)
            Vred, svals_red, Ured, M = _maxent_svd_components(A, stdev)
            m_prior = _maxent_prior(prior, td_grid.size)
            g_data = g_arr - baseline
            for i, log10_reg in enumerate(grid):
                alpha = 10.0 ** float(log10_reg)
                num_iter = int(kwargs.get("num_iter", kwargs.get("max_iter", 200)))
                p, g_fit_data = _quickfit_mem_iteration_numba(
                    Vred,
                    svals_red,
                    Ured,
                    M,
                    stdev.astype(np.float64),
                    g_data.astype(np.float64),
                    m_prior.astype(np.float64),
                    alpha,
                    num_iter,
                )
                result = {
                    "tau": tau_arr,
                    "g": g_arr,
                    "g_fit": g_fit_data + baseline,
                    "td_grid": td_grid,
                    "p": p,
                    "rh_grid": rh_grid,
                }
                chi2_vals[i] = _chi2r_from_arrays(
                    g_arr,
                    result["g_fit"],
                    y_error=y_error,
                    xmin=xmin,
                    xmax=xmax,
                    mask=mask,
                    n_free=n_free,
                )
                sol_vals[i] = _solution_norm(result)
                if return_results:
                    results.append(result)
        else:
            raise ValueError("solver must be 'td' or 'rh'")
    except Exception:
        if on_error == "raise":
            raise
        chi2_vals[:] = float("nan")
        sol_vals[:] = float("nan")
        if return_results:
            results = [{} for _ in grid]

    try:
        from chisurf.core.math import regularization
        corner = regularization.discrete_lcurve_corner(chi2_vals, sol_vals)
    except Exception:
        corner = None

    return FCSMaxEntLCurveResult(
        log10_reg=grid,
        reg=10.0 ** grid,
        chi2r=chi2_vals,
        solution_norm=sol_vals,
        corner_index=corner,
        results=tuple(results),
    )


def compute_fcs_maxent_l_curve(
        tau: np.ndarray,
        g: np.ndarray,
        *,
        y_error: Any = None,
        log10_min: float = -3.0,
        log10_max: float = 3.0,
        n_points: int = 32,
        xmin: int = 0,
        xmax: int | None = None,
        mask: np.ndarray | None = None,
        n_free: int = 0,
        td_min: float | None = None,
        td_max: float | None = None,
        n_td: int = 80,
        s: float = 3.5,
        baseline: float = 1.0,
        prior: np.ndarray | None = None,
        td_grid: np.ndarray | None = None,
        return_results: bool = False,
        **solver_kwargs: Any,
) -> FCSMaxEntLCurveResult:
    """Compute an L-curve for diffusion-time MaxEnt FCS inversion.

    Parameters
    ----------
    tau, g : array_like
        Experimental lag times and correlation amplitudes.
    y_error : array_like or None, optional
        Experimental y standard deviations. ``None`` means unit weights.
    log10_min, log10_max : float, optional
        Inclusive log10 regularization range.
    n_points : int, optional
        Number of regularization points.
    xmin, xmax : int, optional
        Fit-window index range used for chi-squared reduction.
    mask : array_like or None, optional
        Boolean mask applied inside the fit window.
    n_free : int, optional
        Number of free parameters used to reduce chi-squared.
    td_min, td_max : float, optional
        Diffusion-time grid bounds.
    n_td : int, optional
        Number of diffusion-time grid points.
    s : float, optional
        Axial-to-radial waist ratio.
    baseline : float, optional
        Constant baseline subtracted before inversion and added to ``g_fit``.
    prior, td_grid : array_like, optional
        Forwarded to :func:`fcs_maxent`.
    return_results : bool, optional
        If True, store each per-point MaxEnt result dictionary.
    **solver_kwargs
        Additional keyword arguments forwarded to :func:`fcs_maxent`.

    Returns
    -------
    FCSMaxEntLCurveResult
        L-curve arrays and optional per-point results.
    """
    kwargs = dict(solver_kwargs)
    kwargs.update(
        td_min=td_min,
        td_max=td_max,
        n_td=n_td,
        s=s,
        baseline=baseline,
        prior=prior,
        td_grid=td_grid,
    )
    return _compute_fcs_maxent_l_curve(
        tau,
        g,
        y_error=y_error,
        log10_min=log10_min,
        log10_max=log10_max,
        n_points=n_points,
        xmin=xmin,
        xmax=xmax,
        mask=mask,
        n_free=n_free,
        solver="td",
        solver_kwargs=kwargs,
        return_results=return_results,
    )


def compute_fcs_maxent_rh_l_curve(
        tau: np.ndarray,
        g: np.ndarray,
        *,
        y_error: Any = None,
        log10_min: float = -3.0,
        log10_max: float = 3.0,
        n_points: int = 32,
        xmin: int = 0,
        xmax: int | None = None,
        mask: np.ndarray | None = None,
        n_free: int = 0,
        rh_min: float = 0.5,
        rh_max: float = 50.0,
        n_rh: int = 64,
        w0: float = 0.3,
        s: float = 3.5,
        baseline: float = 1.0,
        temperature: float = 298.15,
        viscosity: float | None = None,
        prior: np.ndarray | None = None,
        return_results: bool = False,
        **solver_kwargs: Any,
) -> FCSMaxEntLCurveResult:
    """Compute an L-curve for hydrodynamic-radius MaxEnt FCS inversion.

    Parameters
    ----------
    tau, g : array_like
        Experimental lag times and correlation amplitudes.
    y_error : array_like or None, optional
        Experimental y standard deviations. ``None`` means unit weights.
    log10_min, log10_max : float, optional
        Inclusive log10 regularization range.
    n_points : int, optional
        Number of regularization points.
    xmin, xmax : int, optional
        Fit-window index range used for chi-squared reduction.
    mask : array_like or None, optional
        Boolean mask applied inside the fit window.
    n_free : int, optional
        Number of free parameters used to reduce chi-squared.
    rh_min, rh_max : float, optional
        Hydrodynamic-radius grid bounds in nanometers.
    n_rh : int, optional
        Number of hydrodynamic-radius grid points.
    w0 : float, optional
        Lateral beam waist in micrometers.
    s : float, optional
        Axial-to-radial waist ratio.
    baseline : float, optional
        Constant baseline subtracted before inversion and added to ``g_fit``.
    temperature : float, optional
        Temperature in Kelvin.
    viscosity : float, optional
        Dynamic viscosity in Pa·s.
    prior : array_like, optional
        Prior distribution on the hydrodynamic-radius grid.
    return_results : bool, optional
        If True, store each per-point MaxEnt result dictionary.
    **solver_kwargs
        Additional keyword arguments forwarded to :func:`fcs_maxent_rh`.

    Returns
    -------
    FCSMaxEntLCurveResult
        L-curve arrays and optional per-point results.
    """
    kwargs = dict(solver_kwargs)
    kwargs.update(
        rh_min=rh_min,
        rh_max=rh_max,
        n_rh=n_rh,
        w0=w0,
        s=s,
        baseline=baseline,
        temperature=temperature,
        viscosity=viscosity,
        prior=prior,
    )
    return _compute_fcs_maxent_l_curve(
        tau,
        g,
        y_error=y_error,
        log10_min=log10_min,
        log10_max=log10_max,
        n_points=n_points,
        xmin=xmin,
        xmax=xmax,
        mask=mask,
        n_free=n_free,
        solver="rh",
        solver_kwargs=kwargs,
        return_results=return_results,
    )


def plot_fcs_maxent_result(
        result: dict,
        ax_corr=None,
        ax_dist=None,
        show: bool = True,
        log_tau: bool = True,
        log_td: bool = True,
):
    """Plot measured vs MaxEnt-reconstructed FCS curve and the distribution.

    Parameters
    ----------
    result : dict
        Output of :func:`fcs_maxent`.
    ax_corr, ax_dist : matplotlib Axes, optional
        Axes to plot the correlation curve and the distribution. If None,
        new figures/axes are created.
    show : bool, optional
        If True, call ``plt.show()`` at the end (when axes are created here).
    log_tau, log_td : bool, optional
        If True, use log10 x-axis for tau and td_grid, respectively.
    """
    tau = np.asarray(result["tau"])
    g = np.asarray(result["g"])
    g_fit = np.asarray(result["g_fit"])
    td_grid = np.asarray(result["td_grid"])
    p = np.asarray(result["p"])

    created_fig = False
    if ax_corr is None or ax_dist is None:
        fig, (ax_corr, ax_dist) = plt.subplots(1, 2, figsize=(10, 4))
        created_fig = True

    # Correlation curve: data vs MaxEnt reconstruction
    ax_corr.plot(tau, g, "o", ms=4, label="data")
    ax_corr.plot(tau, g_fit, "-", lw=1.5, label="MaxEnt fit")
    if log_tau:
        ax_corr.set_xscale("log")
    # Use valid mathtext labels for tau
    ax_corr.set_xlabel(r"$\\tau$")
    ax_corr.set_ylabel(r"$G(\\tau)$")
    ax_corr.legend(loc="best")

    # Distribution in diffusion times
    ax_dist.plot(td_grid, p, "-", lw=1.5)
    if log_td:
        ax_dist.set_xscale("log")
    ax_dist.set_xlabel(r"$\\tau_D$")
    ax_dist.set_ylabel(r"$P(\\tau_D)$")

    if created_fig and show:
        plt.tight_layout()
        plt.show()
