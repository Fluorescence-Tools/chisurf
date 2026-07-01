from __future__ import annotations

"""Native Tikhonov (model-free) distance-distribution inversion for DEER.

Solves the non-negatively constrained, smoothness-regularised least-squares
problem

    P = argmin_{P >= 0} || A @ P - b ||^2 + alpha^2 || L @ P ||^2

with ``L`` the second-derivative operator, via :func:`scipy.optimize.nnls` on
the augmented system ``[A; alpha*L] P = [b; 0]``. The regularisation weight
``alpha`` is chosen by generalised cross-validation (GCV) on a log grid, using
the (unconstrained) Tikhonov influence matrix as a robust, cheap heuristic.

Self-contained (numpy/scipy only).
"""

import numpy as np
from scipy.optimize import nnls
from scipy.integrate import trapezoid


def second_derivative_operator(n: int) -> np.ndarray:
    """Return the ``(n-2, n)`` discrete second-derivative matrix ``L``."""
    if n < 3:
        return np.eye(n)
    L = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        L[i, i] = 1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] = 1.0
    return L


def _gcv_score(A: np.ndarray, b: np.ndarray, L: np.ndarray, alpha: float) -> float:
    """GCV functional for the unconstrained Tikhonov solution at ``alpha``."""
    n = A.shape[0]
    AtA = A.T @ A
    LtL = L.T @ L
    try:
        inv = np.linalg.inv(AtA + alpha ** 2 * LtL)
    except np.linalg.LinAlgError:
        return np.inf
    # Influence (hat) matrix H = A (AtA + a^2 LtL)^-1 A^T; only its trace and
    # the residual of the corresponding solution are needed.
    solve = inv @ (A.T @ b)
    resid = A @ solve - b
    trace_h = np.trace(A @ inv @ A.T)
    denom = (n - trace_h) ** 2
    if denom <= 0:
        return np.inf
    return n * float(resid @ resid) / denom


def select_alpha(A: np.ndarray, b: np.ndarray, L: np.ndarray,
                 alphas: np.ndarray | None = None) -> float:
    """Return the GCV-optimal regularisation weight ``alpha`` over a log grid."""
    if alphas is None:
        alphas = np.logspace(-4, 1, 24)
    scores = [_gcv_score(A, b, L, a) for a in alphas]
    return float(alphas[int(np.argmin(scores))])


def lcurve(A: np.ndarray, b: np.ndarray, L: np.ndarray,
           alphas: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the L-curve of the non-negative Tikhonov problem.

    Returns ``(alphas, rho, eta)`` where ``rho = ||A P_a - b||`` (residual norm)
    and ``eta = ||L P_a||`` (solution roughness) for each ``alpha``, using the
    non-negative solution at each point.
    """
    if alphas is None:
        alphas = np.logspace(-4, 1, 24)
    rho = np.empty(alphas.size, dtype=float)
    eta = np.empty(alphas.size, dtype=float)
    for i, a in enumerate(alphas):
        p = solve_tikhonov(A, b, a, L)
        rho[i] = float(np.linalg.norm(A @ p - b))
        eta[i] = float(np.linalg.norm(L @ p))
    return np.asarray(alphas, dtype=float), rho, eta


def select_alpha_lcurve(A: np.ndarray, b: np.ndarray, L: np.ndarray,
                        alphas: np.ndarray | None = None) -> float:
    """Return the L-curve-corner regularisation weight ``alpha``.

    The corner (point of maximum curvature in log-log residual/roughness space)
    is located with :func:`chisurf.core.math.regularization.discrete_lcurve_corner`,
    falling back to GCV if no corner can be determined.
    """
    from chisurf.core.math.regularization import discrete_lcurve_corner

    a_grid, rho, eta = lcurve(A, b, L, alphas)
    k = discrete_lcurve_corner(rho, eta)
    if k is None:
        return select_alpha(A, b, L, a_grid)
    return float(a_grid[int(k)])


def solve_tikhonov(A: np.ndarray, b: np.ndarray, alpha: float,
                   L: np.ndarray | None = None) -> np.ndarray:
    """Solve the non-negative Tikhonov problem for a fixed ``alpha``.

    Parameters
    ----------
    A : numpy.ndarray
        Design matrix ``(nt, nr)`` mapping the distribution onto the signal.
    b : numpy.ndarray
        Target vector ``(nt,)``.
    alpha : float
        Regularisation weight.
    L : numpy.ndarray, optional
        Regularisation operator; second-derivative operator when ``None``.

    Returns
    -------
    numpy.ndarray
        Non-negative solution ``P`` of shape ``(nr,)``.
    """
    nr = A.shape[1]
    if L is None:
        L = second_derivative_operator(nr)
    aug_A = np.vstack([A, alpha * L])
    aug_b = np.concatenate([b, np.zeros(L.shape[0])])
    p, _ = nnls(aug_A, aug_b)
    return p


def tikhonov_distance_distribution(
    kernel: np.ndarray,
    r: np.ndarray,
    v_target: np.ndarray,
    alpha: float | None = None,
    method: str = "gcv",
) -> tuple[np.ndarray, float]:
    """Invert a form factor ``K @ P = v_target`` for a non-negative ``P(r)``.

    Parameters
    ----------
    kernel : numpy.ndarray
        Dipolar kernel ``K(t, r)`` of shape ``(nt, nr)`` (without ``dr``
        weights; they are applied internally).
    r : numpy.ndarray
        Distance axis (Å).
    v_target : numpy.ndarray
        Intramolecular form factor to fit, shape ``(nt,)``.
    alpha : float, optional
        Regularisation weight. When ``None`` or ``<= 0`` it is selected
        automatically according to ``method``.
    method : str
        Auto-selection criterion when ``alpha`` is not given: ``'gcv'``
        (generalised cross-validation) or ``'lcurve'`` (L-curve corner).

    Returns
    -------
    (numpy.ndarray, float)
        The area-normalised distribution ``P(r)`` and the ``alpha`` used.
    """
    r = np.asarray(r, dtype=float)
    dr = float(np.mean(np.diff(r))) if r.size > 1 else 1.0
    A = np.asarray(kernel, dtype=float) * dr  # so A @ P ~ trapz(K P, r)
    b = np.asarray(v_target, dtype=float)
    L = second_derivative_operator(r.size)
    if alpha is None or alpha <= 0:
        alpha = (select_alpha_lcurve(A, b, L) if str(method).lower() == "lcurve"
                 else select_alpha(A, b, L))
    p = solve_tikhonov(A, b, alpha, L)
    area = trapezoid(p, r)
    if area > 0:
        p = p / area
    return p, float(alpha)
