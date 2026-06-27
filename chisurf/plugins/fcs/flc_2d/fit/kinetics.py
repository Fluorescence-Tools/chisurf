"""Rate-matrix kinetics from species-resolved correlation decays.

Port of the analysis behind ``TK_FitF_CorrelationDecay_RateMat``: given the lifetime-
filtered species auto/cross-correlations ``G_ij(tau)``, recover the relaxation kinetics.

The species correlations of an ``n``-state exchange process are sums of exponentials whose
rates are the **non-zero eigenvalues of the rate matrix** ``K`` (shared across all curves),
with curve-specific amplitudes set by the eigenvectors and equilibrium populations::

    G_ij(tau) = offset_ij + sum_k  a_ij^k  exp(-lambda_k tau)

This module fits the shared relaxation rates by **variable projection**: for trial rates
the per-curve amplitudes/offsets are obtained by a linear least-squares solve, and only the
``n-1`` rates are optimized non-linearly (robust, few parameters). For a two-state system
the single rate is ``lambda = k12 + k21``; combined with the equilibrium populations it
yields the individual rate constants ``k12 = lambda * p2``, ``k21 = lambda * p1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "fit_rate_matrix",
    "RateMatrixResult",
    "make_generator_matrix",
    "equilibrium_populations",
]


def make_generator_matrix(rate_matrix: np.ndarray) -> np.ndarray:
    """Return the master-equation generator from an off-diagonal rate matrix.

    Port of ``TK_RateEq_MakeExpMatrix``: given ``K`` with ``K[n, m]`` the rate ``n -> m``,
    the generator ``G`` has ``G[m, n] = K[n, m]`` for the gain terms and
    ``G[i, i] = -sum_{m} K[i, m]`` for the loss terms, so that ``dp/dt = G p`` and
    ``p(t) = expm(G t) p(0)``.
    """
    K = np.asarray(rate_matrix, dtype=float)
    n = K.shape[0]
    G = np.zeros((n, n))
    for i in range(n):
        for m in range(n):
            if m != i:
                G[m, i] += K[i, m]  # gain of state m from state i
                G[i, i] -= K[i, m]  # loss of state i to state m
    return G


def equilibrium_populations(rate_matrix: np.ndarray) -> np.ndarray:
    """Equilibrium populations (normalized null space of the generator)."""
    G = make_generator_matrix(rate_matrix)
    w, v = np.linalg.eig(G)
    idx = int(np.argmin(np.abs(w)))  # eigenvalue closest to 0
    p = np.real(v[:, idx])
    p = np.abs(p)
    s = p.sum()
    return p / s if s > 0 else np.full(G.shape[0], 1.0 / G.shape[0])


@dataclass
class RateMatrixResult:
    """Result of a rate-matrix correlation-decay fit."""

    relaxation_rates: np.ndarray  # (n_states-1,) non-zero eigenvalue magnitudes (1/s)
    relaxation_times_s: np.ndarray  # 1 / relaxation_rates
    rate_matrix: np.ndarray | None  # (n_states, n_states) reconstructed K (2-state only)
    populations: np.ndarray | None  # equilibrium populations used
    r2: float  # global goodness of fit
    amplitudes: dict = field(default_factory=dict)  # {(i,j): per-rate amplitudes}


def _project_amplitudes(rates: np.ndarray, lag: np.ndarray, curves: list[np.ndarray]):
    """Linear least-squares amplitudes+offset per curve for fixed rates; return residual stack."""
    basis = np.empty((lag.size, rates.size + 1))
    basis[:, 0] = 1.0
    for k, r in enumerate(rates):
        basis[:, k + 1] = np.exp(-r * lag)
    resid = []
    amps = []
    for y in curves:
        coef, _res, _rank, _sv = np.linalg.lstsq(basis, y, rcond=None)
        amps.append(coef)
        resid.append(basis @ coef - y)
    return np.concatenate(resid), amps


def fit_rate_matrix(
    lag_s: np.ndarray,
    curves: dict[tuple[int, int], np.ndarray],
    *,
    n_states: int = 2,
    populations: np.ndarray | None = None,
    t_min: float = 5e-4,
    t_max: float = 0.5,
    rate_bounds: tuple[float, float] = (1e-2, 1e6),
) -> RateMatrixResult:
    """Fit shared relaxation rates to species correlation decays and reconstruct ``K``.

    Parameters
    ----------
    lag_s
        Correlation lag axis (seconds).
    curves
        ``{(i, j): G_ij(tau)}`` species auto/cross-correlations (e.g. from
        :func:`chisurf.plugins.fcs.flc_2d.fit.dynamics.filtered_correlation`).
    n_states
        Number of kinetic states (``n_states - 1`` relaxation rates are fitted).
    populations
        Equilibrium populations; used to split a two-state ``lambda`` into ``k12``/``k21``.
        If ``None`` the rate matrix is left as ``None``.
    t_min, t_max
        Fit window in seconds.
    rate_bounds
        Lower/upper bounds (1/s) for the fitted relaxation rates.
    """
    from scipy.optimize import least_squares

    lag = np.asarray(lag_s, dtype=float)
    sel = np.isfinite(lag) & (lag > t_min) & (lag < t_max)
    x = lag[sel]
    curve_list = [np.asarray(g, dtype=float)[sel] for g in curves.values()]
    keys = list(curves.keys())
    if x.size < n_states + 1:
        raise ValueError("not enough points in the fit window")

    n_exp = max(1, n_states - 1)
    # log-spaced initial guesses spanning the window
    lo, hi = 1.0 / t_max, 1.0 / max(t_min, x.min())
    p0 = np.geomspace(max(lo, rate_bounds[0]), min(hi, rate_bounds[1]), n_exp)

    def residual(log_rates):
        rates = np.exp(log_rates)
        r, _ = _project_amplitudes(rates, x, curve_list)
        return r

    res = least_squares(
        residual,
        np.log(p0),
        bounds=(np.log(rate_bounds[0]) * np.ones(n_exp), np.log(rate_bounds[1]) * np.ones(n_exp)),
        method="trf",
        max_nfev=5000,
    )
    rates = np.sort(np.exp(res.x))[::-1]
    _r, amps = _project_amplitudes(rates, x, curve_list)
    ss_res = float(np.sum(_r**2))
    ss_tot = float(np.sum([(y - y.mean()) ** 2 for y in curve_list]).sum()) + 1e-300
    r2 = 1.0 - ss_res / ss_tot

    rate_matrix = None
    pops = None if populations is None else np.asarray(populations, dtype=float)
    if n_states == 2 and pops is not None and pops.size == 2:
        pops = pops / pops.sum()
        lam = float(rates[0])
        k12 = lam * pops[1]  # 1 -> 2
        k21 = lam * pops[0]  # 2 -> 1
        rate_matrix = np.array([[0.0, k12], [k21, 0.0]])

    return RateMatrixResult(
        relaxation_rates=rates,
        relaxation_times_s=1.0 / np.maximum(rates, 1e-300),
        rate_matrix=rate_matrix,
        populations=pops,
        r2=r2,
        amplitudes={keys[i]: amps[i] for i in range(len(keys))},
    )
