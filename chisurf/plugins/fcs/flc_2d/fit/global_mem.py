"""Global (multi-lag) 2D maximum-entropy method for 2D-FLC.

Port of the idea behind ``TK_GFitF_2DMEM``: invert several 2D-FDC matrices, measured at
different macro-time lags ``dT``, **jointly** with a single shared lifetime distribution.

Each lag matrix is modelled as ``M_k = E A G_k A^T E^T`` with

* ``A`` (``n_comp x n_states``, >= 0) — the lifetime amplitudes of each kinetic state,
  **shared across all lags**;
* ``G_k`` (``n_states x n_states``) — the inter-state correlation at lag ``k`` (its
  evolution with lag is the kinetics; off-diagonal terms grow as the states interconvert).

``A`` is parameterized as ``exp(a)`` (non-negative, entropy well-defined); the ``G_k`` are
free (correlations may be negative). The objective is the summed Poisson chi-square plus a
Skilling-Gull entropy on the shared ``A``, minimized by L-BFGS-B with an analytic gradient.
By default the fit targets the *correlation residual* of each matrix (independence baseline
removed) so the small lifetime cross-peaks drive the solution.

.. note::
   Cleanly *separating* the per-state lifetime distributions from weakly-correlated or
   equal-brightness data is an ill-conditioned inverse problem; the shared marginal and the
   per-lag correlation magnitudes are robust, but for quantitative per-state rate constants
   prefer the rate-matrix fit on the lifetime-filtered correlation
   (:func:`chisurf.plugins.fcs.flc_2d.fit.kinetics.fit_rate_matrix`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .ilt import ilt_2d

logger = logging.getLogger(__name__)

__all__ = ["solve_global_mem_2d", "GlobalMEMResult"]


@dataclass
class GlobalMEMResult:
    """Result of a global multi-lag 2D-MEM fit."""

    amplitudes: np.ndarray  # (n_comp, n_states) shared lifetime amplitudes A
    correlations: np.ndarray  # (n_lags, n_states, n_states) per-lag G_k
    tau_grid: np.ndarray  # (n_comp,) lifetimes (ns)
    marginal: np.ndarray  # (n_comp,) shared lifetime distribution = A.sum(axis=1)
    chi2: float

    def state_lifetimes(self) -> np.ndarray:
        """Amplitude-weighted mean lifetime of each kinetic state (ns)."""
        w = self.amplitudes / np.clip(self.amplitudes.sum(axis=0, keepdims=True), 1e-300, None)
        return self.tau_grid @ w


def _correlation_residual(M: np.ndarray) -> np.ndarray:
    """Return the 2D-FLC correlation matrix: M minus its independence (outer-product) baseline.

    ``M_cor = M - r c^T / N`` (``r``/``c`` row/column sums, ``N`` total) removes the large
    *uncorrelated* part of the 2D-FDC so the small lifetime cross-peaks (the kinetics) drive
    the fit, mirroring the MATLAB 2D-FLC normalization.
    """
    r = M.sum(axis=1)
    c = M.sum(axis=0)
    N = M.sum()
    if N <= 0:
        return M.copy()
    return M - np.outer(r, c) / N


def solve_global_mem_2d(
    matrices,
    basis: np.ndarray,
    tau_grid: np.ndarray,
    *,
    n_states: int = 2,
    regulator: float = 1.0,
    weights=None,
    subtract_baseline: bool = True,
    max_iter: int = 800,
) -> GlobalMEMResult:
    """Jointly invert several lag matrices with a shared lifetime distribution.

    Parameters
    ----------
    matrices
        Sequence of square 2D-FDC matrices (one per lag), all ``n_data x n_data``.
    basis
        IRF-convolved exponential basis ``E`` (``n_data x n_comp``).
    tau_grid
        Lifetimes for the basis columns (ns).
    n_states
        Number of kinetic states (columns of the shared ``A``).
    regulator
        Entropy weight ``lambda`` on the shared amplitudes.
    weights
        Optional per-lag weight matrices (default Poisson per matrix).
    subtract_baseline
        Fit the correlation residual (independence baseline removed) so the lifetime
        cross-peaks drive the solution (recommended).
    max_iter
        Maximum L-BFGS-B iterations.
    """
    from scipy.optimize import minimize

    E = np.asarray(basis, dtype=float)
    n_data, n_comp = E.shape
    raw = [np.asarray(M, dtype=float) for M in matrices]
    n_lags = len(raw)
    for M in raw:
        if M.shape != (n_data, n_data):
            raise ValueError("all matrices must be square and match the basis rows")

    # Poisson weights come from the raw counts; fit the correlation residual.
    if weights is None:
        W = [1.0 / (M + M.mean() + 1.0) for M in raw]
    else:
        W = [np.asarray(w, dtype=float) for w in weights]
    mats = [_correlation_residual(M) for M in raw] if subtract_baseline else raw

    # Warm start: the shared lifetime amplitudes A come from the RAW data (the correlation
    # residual has zero marginals), G_k starts near identity and the fit shapes it.
    P_avg = np.zeros((n_comp, n_comp))
    for M in raw:
        P_avg += np.clip(ilt_2d(M, E, tau_grid, method="tikhonov").spectrum, 0, None)
    P_avg /= n_lags
    marg = P_avg.sum(axis=1)
    marg = np.maximum(marg, marg.max() * 1e-6 + 1e-12)
    A0 = np.tile((marg / n_states)[:, None], (1, n_states))
    # break state symmetry by tilting columns towards short/long lifetimes
    tilt = np.linspace(0.5, 1.5, n_states)
    A0 = A0 * tilt[None, :]
    za0 = np.log(A0).ravel()
    G0 = np.tile(np.eye(n_states)[None, :, :], (n_lags, 1, 1)) * 0.5
    g0 = G0.ravel()
    x0 = np.concatenate([za0, g0])

    n_a = n_comp * n_states
    m_prior = A0.copy()
    log_m = np.log(m_prior).ravel()
    lam = float(regulator)

    def unpack(x):
        A = np.exp(x[:n_a]).reshape(n_comp, n_states)
        G = x[n_a:].reshape(n_lags, n_states, n_states)
        return A, G

    def objective(x):
        A, G = unpack(x)
        B = E @ A  # (n_data, n_states)
        grad_B = np.zeros_like(B)
        grad_g = np.empty((n_lags, n_states, n_states))
        chi2 = 0.0
        for k in range(n_lags):
            model = B @ G[k] @ B.T
            diff = model - mats[k]
            D = W[k] * diff
            chi2 += 0.5 * float(np.sum(D * diff))
            grad_g[k] = B.T @ D @ B
            grad_B += D @ B @ G[k].T + D.T @ B @ G[k]
        grad_A = E.T @ grad_B  # (n_comp, n_states)
        za = x[:n_a]
        logterm = za - log_m
        Avec = np.exp(za)
        S = float(np.sum(Avec - m_prior.ravel() - Avec * logterm))
        Q = chi2 - S / lam
        dQ_dA = grad_A + (logterm.reshape(n_comp, n_states)) / lam
        dQ_dza = (dQ_dA * A).ravel()
        return Q, np.concatenate([dQ_dza, grad_g.ravel()])

    res = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-8},
    )
    A, G = unpack(res.x)

    chi2 = 0.0
    for k in range(n_lags):
        B = E @ A
        diff = (B @ G[k] @ B.T) - mats[k]
        chi2 += float(np.sum(W[k] * diff * diff))
    chi2 /= max(n_lags * n_data * n_data - n_a, 1)
    logger.info("[global-MEM] %d lags, %d states, Q=%.4g", n_lags, n_states, res.fun)
    return GlobalMEMResult(
        amplitudes=A,
        correlations=G,
        tau_grid=np.asarray(tau_grid, float),
        marginal=A.sum(axis=1),
        chi2=chi2,
    )
