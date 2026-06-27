"""Explicit 1D Maximum-Entropy (MEM) inverse-Laplace fit with the MATLAB ``mi`` priors.

Port of ``TK_FitF_1DMEM_*`` / ``TK_FitF_1DMEM_MinimizeQ_*`` and ``TK_mi_ModelFunction``
from the Kondo/Schlau-Cohen 2D-FLC code. The 1D-MEM recovers a lifetime distribution
``A(tau)`` from a measured (1D) fluorescence-decay correlation by minimizing the MEM
estimator::

    Q = chi2 - 2 * S / lambda

where ``chi2`` is the weighted misfit of the model ``E @ A + y0`` to the data and ``S`` is
the Skilling-Gull entropy relative to a prior ``mi``::

    S = sum_i ( A_i - mi_i - A_i * log(A_i / mi_i) )

The MATLAB reference minimizes ``Q`` with Nelder-Mead (``fminsearch``) over the full
``A`` vector, which is slow and only marginally convergent. This port keeps the **exact
objective and the iterative outer schedule** (recompute the prior ``mi`` each outer step,
ramp the regularizer ``lambda`` by ``RegulatorFactor``) but minimizes with a bounded
gradient solver (L-BFGS-B over ``a = log A``, analytic gradient) — orders of magnitude
faster and properly convergent.

The four ``mi`` prior types match ``TK_mi_ModelFunction``:

* ``0`` — flat constant prior weighted by the lifetime resolution ``(1 - exp(-(tMax-tMin)/tau))^2``.
* ``1`` — the current distribution normalized to unit mean (self-prior).
* ``2`` — the ``(mi0 + 3*mi1)/4`` blend.
* ``3`` — a Gaussian (in grid-index space) matched to the mean/std of the current ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["mi_prior", "solve_mem_1d", "OneDMEMResult"]


def mi_prior(
    amplitudes: np.ndarray,
    tau_grid: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    mi_type: int = 0,
) -> np.ndarray:
    """Return the entropy prior ``mi`` for a 1D lifetime distribution.

    Port of ``TK_mi_ModelFunction`` (single-state case). ``amplitudes`` is the current
    distribution ``A(tau)``; the returned prior has the same shape.

    Parameters
    ----------
    amplitudes
        Current lifetime distribution ``A`` (``n_comp``).
    tau_grid
        Lifetimes of the grid points (ns).
    t_min, t_max
        Fit-window bounds (ns), used by the resolution weighting of types 0 and 2.
    mi_type
        Prior selector (0-3) matching ``TK_mi_ModelFunction``.
    """
    A = np.asarray(amplitudes, dtype=float)
    tau = np.asarray(tau_grid, dtype=float)
    res_w = (1.0 - np.exp(-(t_max - t_min) / tau)) ** 2  # lifetime-resolution weight

    def _type0() -> np.ndarray:
        mean_all = float(A.mean())
        col_mean = float(A.mean()) or 1.0
        return (mean_all / col_mean) * res_w

    def _type1() -> np.ndarray:
        m = float(A.mean()) or 1.0
        return A / m

    if mi_type == 0:
        mi = _type0()
    elif mi_type == 1:
        mi = _type1()
    elif mi_type == 2:
        mi = (_type0() + 3.0 * _type1()) / 4.0
    elif mi_type == 3:
        ivec = np.arange(1, A.size + 1, dtype=float)
        total = float(A.sum()) or 1.0
        ave_i = float(ivec @ A) / total
        std_i = np.sqrt(float(((ivec - ave_i) ** 2) @ A) / total) or 1.0
        g = np.exp(-((ivec - ave_i) ** 2) / (2.0 * std_i * std_i)) / (std_i * np.sqrt(2 * np.pi))
        mi = g * total
    else:
        raise ValueError(f"unknown mi_type {mi_type}")
    # guard against zeros/negatives (the entropy needs a strictly positive prior)
    mi = np.clip(mi, 0.0, None)
    floor = mi.max() * 1e-10 if mi.max() > 0 else 1e-30
    return mi + floor


@dataclass
class OneDMEMResult:
    """Result of a 1D maximum-entropy inverse-Laplace fit."""

    amplitudes: np.ndarray  # (n_comp,) recovered lifetime distribution A(tau)
    tau_grid: np.ndarray  # (n_comp,) lifetimes (ns)
    offset: float  # fitted constant baseline y0
    model: np.ndarray  # (n_data,) reconstructed decay
    chi2: float  # weighted chi-square
    entropy: float  # final entropy S
    estimator_q: float  # final Q = chi2 - 2 S / lambda
    reg: float  # final regularizer lambda

    def peak_lifetimes(self, n_peaks: int = 2, rel_height: float = 0.1) -> np.ndarray:
        """Return the most prominent lifetimes (local maxima of the distribution)."""
        from .ilt import _distribution_peaks

        return _distribution_peaks(self.tau_grid, self.amplitudes, n_peaks, rel_height)


def _q_and_grad(a, E, y, w, mi, reg, fit_offset):
    """Return ``(Q, grad)`` w.r.t. log-amplitudes ``a`` (last entry = offset if fitted)."""
    n_comp = mi.size
    A = np.exp(a[:n_comp])
    y0 = a[n_comp] if fit_offset else 0.0
    model = E @ A + y0
    diff = model - y
    n = y.size
    chi2 = float(np.sum(w * diff * diff)) / n
    # entropy S = sum(A - mi - A log(A/mi)); dS/dA = -log(A/mi)
    ratio = A / mi
    S = float(np.sum(A - mi - A * np.log(ratio)))
    Q = chi2 - 2.0 * S / reg

    dchi2_dA = (2.0 / n) * (E.T @ (w * diff))
    dQ_dA = dchi2_dA + (2.0 / reg) * np.log(ratio)
    grad = np.empty_like(a)
    grad[:n_comp] = dQ_dA * A  # chain rule A = exp(a)
    if fit_offset:
        grad[n_comp] = (2.0 / n) * float(np.sum(w * diff))
    return Q, grad


def solve_mem_1d(
    decay: np.ndarray,
    basis: np.ndarray,
    tau_grid: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    reg: float = 50.0,
    reg_factor: float = 1.2,
    n_outer: int = 12,
    n_inner: int = 200,
    mi_type: int = 0,
    t_min: float = 0.0,
    t_max: float | None = None,
    fit_offset: bool = True,
    initial: np.ndarray | None = None,
) -> OneDMEMResult:
    """Fit a 1D lifetime distribution by maximum entropy (faithful MATLAB objective).

    Parameters
    ----------
    decay
        Measured fluorescence-decay correlation (``n_data``).
    basis
        IRF-convolved exponential basis ``E`` (``n_data x n_comp``); see
        :func:`chisurf.plugins.fcs.flc_2d.fit.ilt.build_exp_basis`.
    tau_grid
        Lifetimes of the basis columns (ns).
    weights
        Per-bin weights (default Poisson-like ``1 / (decay + mean(decay))``).
    reg
        Initial regularizer ``lambda`` (``RegulatorConst``). Larger = smoother.
    reg_factor
        Per-outer-step multiplier (``RegulatorFactor``); ``>1`` relaxes regularization.
    n_outer
        Number of outer iterations (prior re-estimation + ``lambda`` ramp).
    n_inner
        Max inner L-BFGS-B iterations per outer step.
    mi_type
        Entropy-prior type (0-3), see :func:`mi_prior`.
    t_min, t_max
        Fit-window bounds (ns) for the prior weighting; ``t_max`` defaults to ``tau`` max.
    fit_offset
        Fit a constant baseline ``y0``.
    initial
        Optional initial distribution (default flat).
    """
    from scipy.optimize import minimize

    y = np.asarray(decay, dtype=float)
    E = np.asarray(basis, dtype=float)
    tau = np.asarray(tau_grid, dtype=float)
    n_data, n_comp = E.shape
    if y.shape != (n_data,):
        raise ValueError("decay length must match basis rows")
    if t_max is None:
        t_max = float(tau.max())

    # Work in scaled units so log-amplitudes stay O(1) (avoids exp overflow).
    y_scale = max(float(y.max()), 1.0)
    ys = y / y_scale

    if weights is None:
        weights = 1.0 / (ys + ys.mean() + 1e-6)
    w = np.asarray(weights, dtype=float)

    if initial is None:
        A = np.full(n_comp, max(float(ys.max()), 1e-6) / n_comp)
    else:
        A = np.clip(np.asarray(initial, dtype=float) / y_scale, 1e-12, None).copy()
    y0 = float(np.median(ys)) if fit_offset else 0.0

    # Bounds keep log-amplitudes finite; offset (last entry) is unbounded.
    bounds = [(-60.0, 30.0)] * n_comp + ([(None, None)] if fit_offset else [])

    lam = float(reg)
    Q = entropy = chi2 = 0.0
    for _ in range(n_outer):
        mi = mi_prior(A, tau, t_min=t_min, t_max=t_max, mi_type=mi_type)
        a0 = np.clip(np.log(np.clip(A, 1e-26, None)), -60.0, 30.0)
        x0 = np.concatenate([a0, [y0]]) if fit_offset else a0
        res = minimize(
            _q_and_grad,
            x0,
            args=(E, ys, w, mi, lam, fit_offset),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": n_inner},
        )
        A = np.exp(np.clip(res.x[:n_comp], -60.0, 30.0))
        y0 = float(res.x[n_comp]) if fit_offset else 0.0
        Q = float(res.fun)
        ratio = A / mi
        entropy = float(np.sum(A - mi - A * np.log(ratio)))
        model = E @ A + y0
        chi2 = float(np.sum(w * (model - ys) ** 2)) / n_data
        lam *= reg_factor

    # Rescale back to data units.
    A *= y_scale
    y0 *= y_scale
    model = E @ A + y0
    return OneDMEMResult(A, tau, y0, model, chi2, entropy, Q, lam / reg_factor)
