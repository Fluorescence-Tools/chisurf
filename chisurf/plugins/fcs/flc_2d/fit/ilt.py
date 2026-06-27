"""Fast inverse-Laplace transform (ILT) for 1D and 2D fluorescence-decay data.

This is the modern, fast replacement for the dense Nelder-Mead MEM fit. It solves
the linear inverse problem that underlies 2D-FLC:

    M = E @ P @ E.T + offset            (2D fluorescence-decay correlation)
    d = E @ a + offset                  (1D fluorescence decay)

where ``E`` (``n_data x n_comp``) is the IRF-convolved exponential basis built from a
grid of trial lifetimes, ``a`` (``n_comp``) is the 1D lifetime distribution and ``P``
(``n_comp x n_comp``) is the 2D lifetime distribution (its diagonal is the marginal
lifetime spectrum, its off-diagonal encodes lifetime interconversion between the two
correlated time points).

Two regularization strategies are provided:

* **Tikhonov** (``method="tikhonov"``) -- closed-form, exploits the Kronecker/Sylvester
  structure so a full 2D spectrum is solved in milliseconds via one eigendecomposition
  of ``E.T @ E``. Optionally clipped to be non-negative.
* **NNLS** (``method="nnls"``) -- strict non-negativity through an augmented
  least-squares system solved with :func:`scipy.optimize.nnls`. Slower but rigorous;
  used as the default for the small/log-binned matrices that 2D-FLC produces.

The regularization weight ``reg`` can be given explicitly or selected automatically with
an L-curve (``reg=None``).

References
----------
Kondo, Gordon, Schlau-Cohen et al., 2D fluorescence lifetime correlation spectroscopy
(MATLAB ``TK_FitF_2DMEM_07.m``); Lin et al., fast 2D inverse-Laplace by Tikhonov
regularization with non-negativity (``tikregnc``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chisurf.core.math.regularization import LCurveData

__all__ = [
    "build_exp_basis",
    "ilt_1d",
    "ilt_2d",
    "lcurve_1d",
    "ILTResult1D",
    "ILTResult2D",
    "LCurveData",
    "lifetime_grid",
]


def lifetime_grid(tau_min: float, tau_max: float, n_comp: int, *, log: bool = True) -> np.ndarray:
    """Return a grid of trial lifetimes.

    Parameters
    ----------
    tau_min, tau_max
        Range of lifetimes (same units as the time axis, typically ns).
    n_comp
        Number of grid points.
    log
        If ``True`` (default) the grid is logarithmically spaced, which matches the
        multiplicative resolution of fluorescence-lifetime data.
    """
    if tau_min <= 0 or tau_max <= tau_min:
        raise ValueError("require 0 < tau_min < tau_max")
    if n_comp < 1:
        raise ValueError("n_comp must be >= 1")
    if log:
        return np.geomspace(tau_min, tau_max, n_comp)
    return np.linspace(tau_min, tau_max, n_comp)


def build_exp_basis(
    time_ns: np.ndarray,
    tau_grid_ns: np.ndarray,
    *,
    irf: np.ndarray | None = None,
    irf_time_ns: np.ndarray | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Build the IRF-convolved exponential basis ``E`` (``n_data x n_comp``).

    Column ``m`` is ``exp(-t / tau_grid_ns[m])`` optionally convolved with the
    instrument response function ``irf`` and normalized to unit maximum (matching the
    MATLAB ``TK_ExpMultiDeco_For2DFLC`` convention).

    Parameters
    ----------
    time_ns
        Decay time axis (ns), must be sorted ascending and (for IRF convolution)
        uniformly spaced.
    tau_grid_ns
        Trial lifetimes (ns).
    irf, irf_time_ns
        Optional instrument response function and its time axis (ns). The IRF is
        resampled onto ``time_ns`` and its baseline removed before convolution.
    normalize
        Normalize each basis column to a maximum of 1.0 (default).
    """
    time_ns = np.asarray(time_ns, dtype=float)
    tau_grid_ns = np.asarray(tau_grid_ns, dtype=float)
    n_data = time_ns.size
    n_comp = tau_grid_ns.size

    # Time relative to the start of the gate; the basis decays are referenced to t=0.
    t0 = time_ns - time_ns[0]
    decays = np.exp(-t0[:, None] / tau_grid_ns[None, :])  # (n_data, n_comp)

    if irf is not None:
        irf = np.asarray(irf, dtype=float)
        if irf_time_ns is not None:
            irf = np.interp(time_ns, np.asarray(irf_time_ns, dtype=float), irf, left=0.0, right=0.0)
        elif irf.size != n_data:
            raise ValueError("irf length must match time_ns when irf_time_ns is not given")
        irf = irf - np.median(irf[: max(1, n_data // 20)])  # remove dark/offset baseline
        irf = np.clip(irf, 0.0, None)
        s = irf.sum()
        if s > 0:
            irf = irf / s
        # Causal discrete convolution, truncated to the data window.
        conv = np.empty_like(decays)
        for m in range(n_comp):
            conv[:, m] = np.convolve(decays[:, m], irf)[:n_data]
        decays = conv

    if normalize:
        peak = decays.max(axis=0)
        peak[peak == 0] = 1.0
        decays = decays / peak[None, :]
    return decays


@dataclass
class ILTResult1D:
    """Result of a 1D inverse-Laplace fit."""

    amplitudes: np.ndarray  # (n_comp,) lifetime distribution
    tau_grid: np.ndarray  # (n_comp,) lifetimes (ns)
    offset: float  # fitted constant baseline
    model: np.ndarray  # (n_data,) reconstructed decay
    residuals: np.ndarray  # (n_data,) weighted residuals
    chi2: float
    reg: float

    def peak_lifetimes(self, n_peaks: int = 2, rel_height: float = 0.1) -> np.ndarray:
        """Return the most prominent lifetimes (local maxima of the distribution)."""
        return _distribution_peaks(self.tau_grid, self.amplitudes, n_peaks, rel_height)


@dataclass
class ILTResult2D:
    """Result of a 2D inverse-Laplace fit."""

    spectrum: np.ndarray  # (n_comp, n_comp) 2D lifetime distribution P
    tau_grid: np.ndarray  # (n_comp,) lifetimes (ns)
    offset: float
    model: np.ndarray  # (n_data, n_data) reconstructed 2D-FDC
    chi2: float
    reg: float
    marginal: np.ndarray = field(default_factory=lambda: np.empty(0))  # row-sum of P
    residual: np.ndarray = field(default_factory=lambda: np.empty(0))  # data - model

    def __post_init__(self) -> None:
        """Fill the marginal lifetime distribution (row-sum of P) if not provided."""
        if self.marginal.size == 0:
            self.marginal = self.spectrum.sum(axis=1)

    def peak_lifetimes(self, n_peaks: int = 2, rel_height: float = 0.1) -> np.ndarray:
        """Return the most prominent lifetimes of the marginal distribution."""
        return _distribution_peaks(self.tau_grid, self.marginal, n_peaks, rel_height)


def ilt_1d(
    decay: np.ndarray,
    basis: np.ndarray,
    tau_grid: np.ndarray,
    *,
    reg: float | None = None,
    method: str = "nnls",
    fit_offset: bool = True,
    weights: np.ndarray | None = None,
    reg_order: int = 0,
) -> ILTResult1D:
    """Solve the regularized 1D inverse-Laplace problem ``decay ~= basis @ a``.

    By default the regularization penalizes the **second derivative** of the lifetime
    distribution (``reg_order=2``), which yields the smooth distributions expected from a
    lifetime ILT/MEM instead of the spiky spectra a plain identity penalty produces.

    Parameters
    ----------
    decay
        Measured fluorescence decay histogram (``n_data``).
    basis
        Exponential basis ``E`` from :func:`build_exp_basis` (``n_data x n_comp``).
    tau_grid
        Lifetimes corresponding to the basis columns (ns).
    reg
        Regularization weight. ``None`` selects it by an L-curve over a log-spaced range.
    method
        ``"nnls"`` (non-negative, default) or ``"tikhonov"`` (closed form, clipped).
    fit_offset
        Append a constant column to absorb a flat background.
    weights
        Optional per-bin weights (default Poisson ``1/max(decay, 1)``).
    reg_order
        Order of the smoothness penalty: ``0`` = identity/amplitude norm (default, best
        peak resolution), ``1`` = first-difference, ``2`` = second-difference (smoothest;
        use when a clean measured/synthetic IRF is available and a smooth distribution is
        preferred over resolving closely-spaced lifetimes).
    """
    decay = np.asarray(decay, dtype=float)
    E = np.asarray(basis, dtype=float)
    n_data, n_comp = E.shape
    if decay.shape != (n_data,):
        raise ValueError("decay length must match basis rows")

    if weights is None:
        weights = 1.0 / np.maximum(decay, 1.0)
    w = np.sqrt(np.asarray(weights, dtype=float))

    design = E
    if fit_offset:
        design = np.hstack([E, np.ones((n_data, 1))])
    Wd = w[:, None] * design
    wy = w * decay

    # Penalty operator L (rows x design-columns); the offset column is never penalized.
    L = _penalty_matrix(n_comp, reg_order)
    if fit_offset:
        L = np.hstack([L, np.zeros((L.shape[0], 1))])

    if reg is None:
        # The identity penalty (order 0) has a fast closed-form L-curve; the smoothness
        # operators use the general (per-candidate solve) L-curve.
        reg = _lcurve_reg(Wd, wy) if reg_order <= 0 else _lcurve_reg_general(Wd, wy, L, method)

    coef = _solve_reg_L(Wd, wy, L, reg, method)
    amps = coef[:n_comp]
    offset = float(coef[n_comp]) if fit_offset else 0.0

    model = E @ amps + offset
    residuals = (model - decay) * w
    chi2 = float(np.sum(residuals**2) / max(n_data - 1, 1))
    return ILTResult1D(
        amps, np.asarray(tau_grid, float), offset, model, residuals, chi2, float(reg)
    )


def ilt_2d(
    matrix: np.ndarray,
    basis: np.ndarray,
    tau_grid: np.ndarray,
    *,
    reg: float | None = None,
    method: str = "tikhonov",
    fit_offset: bool = True,
    weights: np.ndarray | None = None,
    nonneg: bool = True,
) -> ILTResult2D:
    """Solve the regularized 2D inverse-Laplace problem ``M ~= E @ P @ E.T``.

    The ``"tikhonov"`` method uses the Sylvester/eigen structure and is extremely fast
    (one eigendecomposition of ``E.T @ E``); with ``nonneg=True`` the result is clipped
    to be non-negative. The ``"nnls"`` method enforces strict non-negativity on the full
    Kronecker system and is slower but exact.

    Parameters
    ----------
    matrix
        Measured 2D-FDC matrix (``n_data x n_data``).
    basis
        Exponential basis ``E`` (``n_data x n_comp``).
    tau_grid
        Lifetimes for the basis columns (ns).
    reg
        Tikhonov weight; ``None`` selects via L-curve.
    method
        ``"tikhonov"`` (default, fast) or ``"nnls"`` (strict, slower).
    fit_offset
        Subtract a fitted constant offset (the uncorrelated baseline) before inversion.
    weights
        Optional per-element weights (default Poisson ``1/(M + mean(M))``).
    nonneg
        Clip the Tikhonov spectrum to be non-negative (ignored for NNLS).
    """
    M = np.asarray(matrix, dtype=float)
    E = np.asarray(basis, dtype=float)
    n_data, n_comp = E.shape
    if M.shape != (n_data, n_data):
        raise ValueError("matrix must be square with size matching basis rows")

    if weights is None:
        weights = 1.0 / (M + M.mean() + 1.0)
    W = np.asarray(weights, dtype=float)

    offset = 0.0
    Mc = M
    if fit_offset:
        offset = float(np.median(np.concatenate([M[0, :], M[:, 0], M[-1, :], M[:, -1]])))
        Mc = M - offset

    if method == "nnls":
        P = _ilt_2d_nnls(Mc, E, W, reg)
        used_reg = reg if reg is not None else 0.0
    else:
        P, used_reg = _ilt_2d_tikhonov(Mc, E, reg, nonneg=nonneg)

    model = E @ P @ E.T + offset
    resid = (model - M) * np.sqrt(W)
    chi2 = float(np.sum(resid**2) / max(n_data * n_data - n_comp, 1))
    return ILTResult2D(
        P, np.asarray(tau_grid, float), offset, model, chi2, float(used_reg), residual=(M - model)
    )


# --------------------------------------------------------------------------- helpers


def lcurve_1d(
    decay: np.ndarray,
    basis: np.ndarray,
    *,
    method: str = "nnls",
    fit_offset: bool = True,
    weights: np.ndarray | None = None,
    reg_order: int = 0,
    n_points: int = 24,
) -> LCurveData:
    """Sample the L-curve (residual vs solution norm) of the 1D inversion.

    Solves the regularized problem at ``n_points`` log-spaced weights and marks the
    maximum-curvature corner (the same weight :func:`ilt_1d` selects when ``reg=None``).
    """
    decay = np.asarray(decay, dtype=float)
    E = np.asarray(basis, dtype=float)
    n_data, n_comp = E.shape
    if weights is None:
        weights = 1.0 / np.maximum(decay, 1.0)
    w = np.sqrt(np.asarray(weights, dtype=float))
    design = np.hstack([E, np.ones((n_data, 1))]) if fit_offset else E
    Wd = w[:, None] * design
    wy = w * decay
    L = _penalty_matrix(n_comp, reg_order)
    if fit_offset:
        L = np.hstack([L, np.zeros((L.shape[0], 1))])

    s = np.linalg.svd(Wd, compute_uv=False)
    s2 = s[s > 0] ** 2
    if s2.size == 0:
        regs = np.geomspace(1e-6, 1e2, n_points)
    else:
        regs = np.geomspace(s2.max() * 1e-6, s2.max() * 1e2, n_points)
    res, sol = [], []
    for r in regs:
        c = _solve_reg_L(Wd, wy, L, float(r), method)
        res.append(float(np.linalg.norm(Wd @ c - wy)))
        sol.append(float(np.linalg.norm(L @ c)))
    res = np.asarray(res)
    sol = np.asarray(sol)
    chosen = _lcurve_reg(Wd, wy) if reg_order <= 0 else _lcurve_reg_general(Wd, wy, L, method)
    j = int(np.argmin(np.abs(np.log(regs) - np.log(chosen))))
    return LCurveData(reg=regs, residual_norm=res, solution_norm=sol, corner_index=j)


def _penalty_matrix(n: int, order: int) -> np.ndarray:
    """Return the smoothness penalty operator of the given difference order (``rows x n``)."""
    if order <= 0 or n <= order:
        return np.eye(n)
    D = np.eye(n)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D


def _solve_reg_L(Wd: np.ndarray, wy: np.ndarray, L: np.ndarray, reg: float, method: str):
    """Solve ``min ||Wd c - wy||^2 + reg ||L c||^2`` (optionally with ``c >= 0``)."""
    if method == "nnls":
        from scipy.optimize import nnls

        A = np.vstack([Wd, np.sqrt(reg) * L])
        b = np.concatenate([wy, np.zeros(L.shape[0])])
        coef, _ = nnls(A, b, maxiter=40 * Wd.shape[1])
        return coef
    # closed-form generalized Tikhonov
    G = Wd.T @ Wd + reg * (L.T @ L)
    return np.linalg.solve(G, Wd.T @ wy)


def _lcurve_reg(Wd: np.ndarray, wy: np.ndarray) -> float:
    """Pick an identity-Tikhonov weight by maximum curvature of the L-curve (closed form).

    Uses a single thin SVD of the design and evaluates the residual and solution norms for
    every candidate ``reg`` in closed form. With ``Wd = U diag(s) V^T`` and
    ``beta = U^T wy`` each candidate costs O(rank) rather than O(n^3).
    """
    U, s, _Vt = np.linalg.svd(Wd, full_matrices=False)
    s2 = s**2
    pos = s2[s2 > 0]
    if pos.size == 0:
        return 1.0
    beta = U.T @ wy
    perp2 = max(float(wy @ wy - beta @ beta), 0.0)  # residual outside the SVD range
    regs = np.geomspace(pos.min() * 1e-3, pos.max() * 1e1, 24)
    f = regs[:, None] / (s2[None, :] + regs[:, None])  # (n_reg, rank)
    res_norm = np.sqrt((f**2 @ (beta**2)) + perp2)
    sol_norm = np.sqrt(((s / (s2 + regs[:, None])) ** 2) @ (beta**2))
    rho = np.log(res_norm + 1e-300)
    eta = np.log(sol_norm + 1e-300)
    curv = _curvature(np.log(regs), rho, eta)
    return float(regs[int(np.argmax(curv))])


def _lcurve_reg_general(Wd: np.ndarray, wy: np.ndarray, L: np.ndarray, method: str) -> float:
    """Pick the regularization weight by L-curve curvature for a general penalty ``L``.

    Solves the (small) regularized system at each candidate ``reg`` and selects the
    maximum-curvature corner of the ``log||residual|| vs log||L c||`` curve. With at most a
    few dozen lifetime grid points this is fast even when ``method='nnls'``.
    """
    # Scale candidate range to the problem's singular values for stability.
    s = np.linalg.svd(Wd, compute_uv=False)
    s2 = s[s > 0] ** 2
    if s2.size == 0:
        return 1.0
    regs = np.geomspace(s2.max() * 1e-6, s2.max() * 1e2, 16)
    rho, eta = [], []
    for r in regs:
        c = _solve_reg_L(Wd, wy, L, float(r), method)
        rho.append(np.log(np.linalg.norm(Wd @ c - wy) + 1e-300))
        eta.append(np.log(np.linalg.norm(L @ c) + 1e-300))
    curv = _curvature(np.log(regs), np.asarray(rho), np.asarray(eta))
    return float(regs[int(np.argmax(curv))])


def _curvature(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx, dy = np.gradient(x, t), np.gradient(y, t)
    ddx, ddy = np.gradient(dx, t), np.gradient(dy, t)
    denom = (dx**2 + dy**2) ** 1.5 + 1e-300
    return np.abs(dx * ddy - dy * ddx) / denom


def _ilt_2d_tikhonov(
    M: np.ndarray, E: np.ndarray, reg: float | None, *, nonneg: bool
) -> tuple[np.ndarray, float]:
    """Closed-form 2D Tikhonov via the Sylvester/eigen trick.

    Minimizes ``||E P E.T - M||^2 + reg ||P||^2``. With ``G = E.T E = U diag(g) U.T`` and
    ``B = E.T M E`` the rotated solution is ``Pt_ij = (U.T B U)_ij / (g_i g_j + reg)``.
    """
    G = E.T @ E
    g, U = np.linalg.eigh(G)
    B = E.T @ M @ E
    Bt = U.T @ B @ U
    if reg is None:
        # heuristic: scale to a small fraction of the largest squared singular value
        reg = float((g.max() ** 2) * 1e-3)
    denom = np.outer(g, g) + reg
    Pt = Bt / denom
    P = U @ Pt @ U.T
    if nonneg:
        P = np.clip(P, 0.0, None)
    return P, float(reg)


def _ilt_2d_nnls(M: np.ndarray, E: np.ndarray, W: np.ndarray, reg: float | None) -> np.ndarray:
    """Strict non-negative 2D ILT on the Kronecker system (slower, exact)."""
    from scipy.optimize import nnls

    n_data, n_comp = E.shape
    sw = np.sqrt(W).ravel()
    D = np.kron(E, E) * sw[:, None]  # (n_data^2, n_comp^2)
    b = (M.ravel()) * sw
    if reg is None:
        s = np.linalg.svd(E, compute_uv=False)
        reg = float((s.max() ** 2) ** 2 * 1e-3)
    A = np.vstack([D, np.sqrt(reg) * np.eye(n_comp * n_comp)])
    bb = np.concatenate([b, np.zeros(n_comp * n_comp)])
    p, _ = nnls(A, bb, maxiter=10 * n_comp * n_comp)
    return p.reshape(n_comp, n_comp)


def _distribution_peaks(
    tau: np.ndarray, amp: np.ndarray, n_peaks: int, rel_height: float
) -> np.ndarray:
    """Return up to ``n_peaks`` lifetimes at local maxima, strongest first."""
    amp = np.asarray(amp, dtype=float)
    if amp.size == 0 or amp.max() <= 0:
        return np.empty(0)
    thr = rel_height * amp.max()
    peaks = []
    for i in range(amp.size):
        left = amp[i - 1] if i > 0 else -np.inf
        right = amp[i + 1] if i < amp.size - 1 else -np.inf
        if amp[i] >= left and amp[i] >= right and amp[i] >= thr:
            peaks.append(i)
    peaks.sort(key=lambda i: amp[i], reverse=True)
    return tau[peaks[:n_peaks]]
