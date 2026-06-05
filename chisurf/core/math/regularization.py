from __future__ import annotations

"""Regularization utilities, including discrete L-curve corner detection.

This module provides a lightweight implementation of a corner-finding
algorithm for *discrete* L-curves, i.e. a sequence of points
``(rho_k, eta_k)`` sampled along a regularization path.  It is inspired by
Per Christian Hansen's REGU toolbox for MATLAB, but implemented in a
simplified form suitable for use inside ChiSurf.

The main entry point is :func:`discrete_lcurve_corner`, which returns the
index of the point with the largest geometric curvature in log–log space.
"""

import numpy as np
from typing import Optional


def discrete_lcurve_corner(
    rho: np.ndarray,
    eta: np.ndarray,
) -> Optional[int]:
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
        L-curve in log–log space, or ``None`` if no reasonable corner can
        be determined.

    Notes
    -----
    The algorithm works purely on the discrete polyline defined by the
    L-curve in log–log space.  For each interior point it estimates a
    curvature-like quantity based on three consecutive points and returns
    the index with maximal curvature.  This mirrors the geometric spirit
    of the REGU toolbox corner detectors while keeping the implementation
    compact and dependency-free.
    """

    rho = np.asarray(rho, dtype=float).ravel()
    eta = np.asarray(eta, dtype=float).ravel()

    if rho.size != eta.size or rho.size < 3:
        return None

    # Remove non-finite or non-positive entries; L-curve is defined in log
    # coordinates, so we require strictly positive values.
    mask = np.isfinite(rho) & np.isfinite(eta) & (rho > 0.0) & (eta > 0.0)
    idx_all = np.nonzero(mask)[0]
    if idx_all.size < 3:
        return None

    lrho = np.log10(rho[idx_all])
    leta = np.log10(eta[idx_all])
    P = np.vstack((lrho, leta)).T  # shape (n, 2)

    n = P.shape[0]
    curv = np.zeros(n, dtype=float)

    # Discrete curvature estimate based on three consecutive points in the
    # polyline, using the area of the triangle spanned by the edge vectors
    # as a proxy for curvature.
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

        # Twice the signed area of the triangle spanned by v1 and v2.
        area2 = abs(v1[0] * v2[1] - v1[1] * v2[0])
        # Normalize by edge lengths to obtain a scale-invariant measure.
        curv[k] = area2 / (n1 * n2 * n3)

    if not np.any(np.isfinite(curv)):
        return int(idx_all[0])

    k_local = int(np.nanargmax(curv))
    return int(idx_all[k_local])
