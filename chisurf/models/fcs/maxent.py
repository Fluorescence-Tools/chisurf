from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numba import njit


def build_diffusion_kernel(
        tau: np.ndarray,
        td_grid: np.ndarray,
        s: float = 3.5,
) -> np.ndarray:
    """Build a simple 3D Gaussian FCS diffusion kernel.

    Each column k[:, i] is the normalized correlation shape for a given
    diffusion time td_grid[i]. The overall amplitude (1/N) and offset (b)
    are *not* included and should be handled separately if needed.
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
    Nd = ydata.shape[0]
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
    """Run a simple MaxEnt inversion on an FCS correlation curve.

    Parameters
    ----------
    tau : array_like
        Correlation lag times (same units as td_min/td_max).
    g : array_like
        Measured correlation amplitudes G(tau).
    td_min, td_max : float, optional
        Minimum and maximum diffusion times for the grid. If omitted,
        they are estimated from the tau range.
    n_td : int, optional
        Number of diffusion-time grid points (log-spaced).
    s : float, optional
        Axial-to-radial waist ratio z0/w0 in the 3D Gaussian volume model.
    regularization_factor : float, optional
        Entropy weight parameter passed as ``nu`` to the MaxEnt solver.
    weights : array_like, optional
        Data weights (same length as tau). If None, all ones are used.
    prior : array_like, optional
        Prior distribution on the diffusion-time grid. If None, uniform.

    Returns
    -------
    dict
        Dictionary with keys ``tau``, ``g``, ``g_fit``, ``td_grid``, ``p``.
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
    tau = np.asarray(tau, dtype=float).ravel()
    g = np.asarray(g, dtype=float).ravel()
    if tau.size != g.size:
        raise ValueError("tau and g must have the same length")

    rh_min_val = max(float(rh_min), 1.0e-3)
    rh_max_val = max(float(rh_max), rh_min_val * 1.001)
    n_rh_val = int(n_rh) if int(n_rh) > 2 else 3
    rh_grid = np.linspace(rh_min_val, rh_max_val, n_rh_val)

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
