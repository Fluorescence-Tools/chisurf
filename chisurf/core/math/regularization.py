"""Regularization utilities, including discrete L-curve corner detection.

This module provides a lightweight implementation of a corner-finding
algorithm for *discrete* L-curves, i.e. a sequence of points
``(rho_k, eta_k)`` sampled along a regularization path.  It is inspired by
Per Christian Hansen's REGU toolbox for MATLAB, but implemented in a
simplified form suitable for use inside ChiSurf.

The main entry point is :func:`discrete_lcurve_corner`, which returns the
index of the point with the largest geometric curvature in log-log space.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class LCurveData:
    """A sampled L-curve plus its detected corner — the general, GUI-free L-curve model.

    This is the shared data model behind the reusable L-curve component: any regularized
    inversion (Tikhonov, NNLS, MEM, ...) can populate it via :func:`sample_lcurve` and any
    view (the AutoForm ``lcurve`` section) can render it, so the L-curve is consistent
    across tools and fits.

    Attributes
    ----------
    reg
        Regularization weights tested (linear).
    residual_norm
        Misfit norm ``||A x - b||`` at each weight (the L-curve x-axis).
    solution_norm
        Solution (semi-)norm ``||L x||`` at each weight (the L-curve y-axis).
    corner_index
        Index of the detected corner (``None`` if undetermined).
    """

    reg: np.ndarray
    residual_norm: np.ndarray
    solution_norm: np.ndarray
    corner_index: int | None = None

    @property
    def corner_reg(self) -> float | None:
        """Regularization weight at the corner (``None`` if undetermined)."""
        if self.corner_index is None:
            return None
        return float(self.reg[self.corner_index])

    @property
    def corner_point(self) -> tuple[float, float] | None:
        """``(residual_norm, solution_norm)`` at the corner (``None`` if undetermined)."""
        if self.corner_index is None:
            return None
        return float(self.residual_norm[self.corner_index]), float(
            self.solution_norm[self.corner_index]
        )


def sample_lcurve(
    solve: Callable[[float], tuple[float, float]],
    regs: Sequence[float],
) -> LCurveData:
    """Sample an L-curve from a solver and locate its corner.

    Parameters
    ----------
    solve
        Callable mapping a regularization weight to ``(residual_norm, solution_norm)``.
    regs
        Regularization weights to evaluate (typically log-spaced).

    Returns
    -------
    LCurveData
        The sampled curve with the corner index from :func:`discrete_lcurve_corner`.
    """
    regs = np.asarray(list(regs), dtype=float)
    rho = np.empty(regs.size)
    eta = np.empty(regs.size)
    for i, r in enumerate(regs):
        rho[i], eta[i] = solve(float(r))
    corner = discrete_lcurve_corner(rho, eta)
    return LCurveData(reg=regs, residual_norm=rho, solution_norm=eta, corner_index=corner)


def csvd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the compact singular value decomposition.

    Parameters
    ----------
    A : array_like
        Input matrix.

    Returns
    -------
    U, s, V : numpy.ndarray
        Compact left singular vectors, singular values, and right singular
        vectors such that ``A == U @ diag(s) @ V.T``.
    """
    U, s, VT = np.linalg.svd(np.asarray(A, dtype=float), full_matrices=False)
    return U, s, VT.T


def tikhonov(
    U: np.ndarray,
    s: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, float, float]:
    """Solve a Tikhonov-regularized least-squares problem in SVD form.

    Parameters
    ----------
    U, s, V : array_like
        Compact SVD factors of the forward matrix.
    b : array_like
        Right-hand side vector.
    lam : float
        Regularization parameter.

    Returns
    -------
    x : numpy.ndarray
        Regularized solution.
    rho : float
        Residual norm ``||A x - b||``.
    eta : float
        Solution norm ``||x||``.
    """
    U = np.asarray(U, dtype=float)
    s = np.asarray(s, dtype=float)
    V = np.asarray(V, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    lam = float(lam)
    filt = s / (s * s + lam * lam)
    x = V @ (filt * (U.T @ b))
    rho = float(np.linalg.norm((U @ (s * (V.T @ x))) - b))
    eta = float(np.linalg.norm(x))
    return x, rho, eta


def tsvd(
    U: np.ndarray,
    s: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    k: int,
) -> tuple[np.ndarray, float, float]:
    """Solve a truncated-SVD regularized least-squares problem.

    Parameters
    ----------
    U, s, V : array_like
        Compact SVD factors of the forward matrix.
    b : array_like
        Right-hand side vector.
    k : int
        Number of singular components to keep.

    Returns
    -------
    x : numpy.ndarray
        Truncated-SVD solution.
    rho : float
        Residual norm ``||A x - b||``.
    eta : float
        Solution norm ``||x||``.
    """
    U = np.asarray(U, dtype=float)
    s = np.asarray(s, dtype=float)
    V = np.asarray(V, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    k = max(min(int(k), s.size), 0)
    x = V[:, :k] @ ((U[:, :k].T @ b) / s[:k])
    rho = float(np.linalg.norm((U @ (s * (V.T @ x))) - b))
    eta = float(np.linalg.norm(x))
    return x, rho, eta


def gcv(
    U: np.ndarray,
    s: np.ndarray,
    b: np.ndarray,
    n_points: int = 100,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate a Tikhonov parameter using a simple GCV scan.

    Parameters
    ----------
    U, s : array_like
        Compact SVD left singular vectors and singular values.
    b : array_like
        Right-hand side vector.
    n_points : int, optional
        Number of log-spaced regularization parameters to scan.

    Returns
    -------
    reg_min : float
        Regularization parameter with the smallest scanned GCV score.
    G : numpy.ndarray
        GCV scores for the scanned parameters.
    reg_param : numpy.ndarray
        Scanned regularization parameters.
    """
    U = np.asarray(U, dtype=float)
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    n = max(int(n_points), 2)
    s_pos = s[s > 0.0]
    if s_pos.size == 0:
        reg_param = np.array([1.0, 10.0], dtype=float)
        return float(reg_param[0]), np.ones_like(reg_param), reg_param
    lo = max(float(s_pos[0]) * 1.0e-6, np.finfo(float).tiny)
    hi = max(float(s_pos[0]) * 1.0e2, lo * 10.0)
    reg_param = np.logspace(np.log10(lo), np.log10(hi), n)
    G = np.empty_like(reg_param)
    for i, lam in enumerate(reg_param):
        filt = s * s / (s * s + lam * lam)
        resid_coef = 1.0 - filt
        numer = float(np.sum((resid_coef * (U.T @ b)) ** 2))
        denom = float(np.sum(resid_coef)) ** 2
        G[i] = numer / denom if denom > 0.0 else np.inf
    idx = int(np.nanargmin(G))
    return float(reg_param[idx]), G, reg_param


def l_curve(
    U: np.ndarray,
    s: np.ndarray,
    b: np.ndarray,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Compute a Tikhonov L-curve in residual/solution norm space.

    Parameters
    ----------
    U, s : array_like
        Compact SVD left singular vectors and singular values.
    b : array_like
        Right-hand side vector.
    n_points : int, optional
        Number of log-spaced regularization parameters.

    Returns
    -------
    rho, eta : numpy.ndarray
        Residual and solution norms for each regularization parameter.
    corner : int or None
        Detected L-curve corner index.
    """
    U = np.asarray(U, dtype=float)
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    n = max(int(n_points), 2)
    s_pos = s[s > 0.0]
    if s_pos.size == 0:
        reg_param = np.array([1.0, 10.0], dtype=float)
        return np.ones(2), np.ones(2), None
    lo = max(float(s_pos[0]) * 1.0e-6, np.finfo(float).tiny)
    hi = max(float(s_pos[0]) * 1.0e2, lo * 10.0)
    reg_param = np.logspace(np.log10(lo), np.log10(hi), n)
    rho = np.empty_like(reg_param)
    eta = np.empty_like(reg_param)
    UTb = U.T @ b
    for i, lam in enumerate(reg_param):
        filt = s * s / (s * s + lam * lam)
        rho[i] = float(np.linalg.norm((1.0 - filt) * UTb))
        eta[i] = float(np.linalg.norm(filt * UTb))
    return rho, eta, discrete_lcurve_corner(rho, eta)


def _clean_lcurve_points(
    rho: np.ndarray,
    eta: np.ndarray,
    eps: float = 1.0e-300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite log-log L-curve points with consecutive duplicates removed.

    Parameters
    ----------
    rho, eta : array_like
        Residual and solution norms.
    eps : float, optional
        Positive floor used for zero-valued norms before taking logs.

    Returns
    -------
    tuple of numpy.ndarray
        Original indices, log10(rho), and log10(eta) for the cleaned points.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    eta = np.asarray(eta, dtype=float).ravel()
    if rho.size != eta.size:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    mask = np.isfinite(rho) & np.isfinite(eta)
    idx_all = np.nonzero(mask)[0]
    if idx_all.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    rho = np.clip(rho[idx_all], eps, None)
    eta = np.clip(eta[idx_all], eps, None)
    lrho = np.log10(rho)
    leta = np.log10(eta)

    keep = [0]
    for i in range(1, idx_all.size):
        if lrho[i] != lrho[keep[-1]] or leta[i] != leta[keep[-1]]:
            keep.append(i)

    keep_arr = np.asarray(keep, dtype=int)
    return idx_all[keep_arr], lrho[keep_arr], leta[keep_arr]


def _normalized_score(values: np.ndarray) -> np.ndarray:
    """Normalize a non-negative score to the range [0, 1].

    Parameters
    ----------
    values : array_like
        Score values.

    Returns
    -------
    numpy.ndarray
        Normalized score array.
    """
    values = np.asarray(values, dtype=float)
    max_value = np.nanmax(values)
    if not np.isfinite(max_value) or max_value <= 0.0:
        return np.zeros_like(values, dtype=float)
    return values / max_value


def discrete_lcurve_corner(
    rho: np.ndarray,
    eta: np.ndarray,
) -> int | None:
    """Locate the corner of a discrete L-curve.

    Parameters
    ----------
    rho : array_like
        Residual norms ``||A x - b||`` for different regularization
        parameters.
    eta : array_like
        Solution (semi-)norms ``||x||`` or ``||L x||`` for the same
        regularization parameters as in ``rho``.

    Returns
    -------
    int or None
        Index ``k`` such that ``(rho[k], eta[k])`` is the corner of the
        L-curve in log-log space, or ``None`` if no reasonable corner can
        be determined.

    Notes
    -----
    The algorithm works on the discrete polyline defined by the L-curve in
    log-log space. Zero-valued norms are floored before taking logs so that
    degenerate L-shapes with points on an axis can still be detected. For
    each interior point it combines a normalized curvature estimate with a
    normalized distance-to-chord score and returns the index with the largest
    combined score.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    eta = np.asarray(eta, dtype=float).ravel()

    if rho.size != eta.size or rho.size < 3:
        return None

    idx_all, lrho, leta = _clean_lcurve_points(rho, eta)
    if idx_all.size < 3:
        return None

    P = np.vstack((lrho, leta)).T  # shape (n, 2)
    n = P.shape[0]

    curv = np.zeros(n, dtype=float)
    for k in range(1, n - 1):
        p0 = P[k - 1]
        p1 = P[k]
        p2 = P[k + 1]

        v1 = p0 - p1
        v2 = p2 - p1

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        n3 = np.linalg.norm(v2 - v1)
        if n1 <= 0.0 or n2 <= 0.0 or n3 <= 0.0:
            curv[k] = 0.0
            continue

        area2 = abs(v1[0] * v2[1] - v1[1] * v2[0])
        curv[k] = area2 / (n1 * n2 * n3)

    chord = P[-1] - P[0]
    chord_norm = np.linalg.norm(chord)
    if chord_norm <= 0.0:
        return int(idx_all[0])
    dist = np.abs(np.cross(chord, P - P[0])) / chord_norm

    score = 0.5 * _normalized_score(curv) + 0.5 * _normalized_score(dist)
    if not np.any(np.isfinite(score)):
        return None

    k_local = int(np.nanargmax(score))
    return int(idx_all[k_local])


def corner(
    rho: np.ndarray,
    eta: np.ndarray,
) -> int | None:
    """Locate the corner of a discrete L-curve.

    Parameters
    ----------
    rho, eta : array_like
        Residual and solution norms sampled along a regularization path.

    Returns
    -------
    int or None
        Index of the detected L-curve corner, or ``None``.
    """
    return discrete_lcurve_corner(rho, eta)
