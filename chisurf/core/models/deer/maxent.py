from __future__ import annotations

"""Maximum-entropy (MaxEnt) model-free distance-distribution inversion for DEER.

An alternative to Tikhonov regularisation: recover a non-negative ``P(r)`` from
the intramolecular form factor ``K @ P = V`` by maximising the Shannon–Jaynes
entropy ``S = -sum_i p_i log(p_i / m_i)`` subject to the data, i.e. minimising

    Q(P) = chi2(P) / 2 - alpha * S(P),   P >= 0.

The exponential (Cambridge-style) fixed-point update ``p = m * exp(-grad/alpha)``
keeps the solution positive automatically. The regularisation weight ``alpha``
is chosen from an L-curve corner (residual norm vs. distribution roughness),
reusing :func:`chisurf.core.math.regularization.discrete_lcurve_corner`.

Self-contained (numpy/scipy only).
"""

import numpy as np

from .tikhonov import second_derivative_operator


def maxent_inversion(
    kernel: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    prior: np.ndarray | None = None,
    p_init: np.ndarray | None = None,
    n_iter: int = 1000,
    damping: float = 0.3,
    tol: float = 1e-10,
) -> np.ndarray:
    """Return MaxEnt probability masses ``p`` (``sum(p) = 1``) for ``K @ p = b``.

    Parameters
    ----------
    kernel : numpy.ndarray
        Dipolar kernel ``K`` of shape ``(nt, nr)`` (probability-mass convention:
        ``K @ p`` is the form factor when ``p`` are masses summing to one).
    b : numpy.ndarray
        Target form factor ``(nt,)``.
    weights : numpy.ndarray
        Per-point weights ``1/sigma**2`` (scalar-broadcast or length ``nt``).
    alpha : float
        Entropy regularisation weight (larger -> smoother/flatter).
    prior : numpy.ndarray, optional
        Prior masses ``m`` (the entropy reference measure); uniform when ``None``.
    p_init : numpy.ndarray, optional
        Warm-start masses for the iteration; the prior ``m`` is used when ``None``.
    n_iter, damping, tol : int, float, float
        Iteration budget, update damping and convergence tolerance.
    """
    K = np.asarray(kernel, dtype=float)
    b = np.asarray(b, dtype=float)
    nr = K.shape[1]
    w = np.broadcast_to(np.asarray(weights, dtype=float), b.shape)

    m = (np.ones(nr) / nr) if prior is None else np.clip(prior, 1e-12, None)
    m = m / m.sum()
    if p_init is None:
        p = m.copy()
    else:
        p = np.clip(np.asarray(p_init, dtype=float), 1e-12, None)
        p = p / p.sum()
    KtW = K.T * w  # (nr, nt)

    for _ in range(int(n_iter)):
        grad = KtW @ (K @ p - b)          # data-misfit gradient
        expo = -grad / max(alpha, 1e-12)
        expo -= expo.max()                # overflow guard
        p_new = m * np.exp(expo)
        s = p_new.sum()
        if s <= 0 or not np.isfinite(s):
            break
        p_new /= s
        p_next = (1.0 - damping) * p + damping * p_new
        if np.max(np.abs(p_next - p)) < tol:
            p = p_next
            break
        p = p_next
    return p


def maxent_distance_distribution(
    kernel: np.ndarray,
    r: np.ndarray,
    v_target: np.ndarray,
    sigma: float = 1.0,
    alpha: float | None = None,
    n_iter: int = 1000,
    n_alpha: int = 20,
    method: str = "discrepancy",
    return_lcurve: bool = False,
):
    """Invert ``K @ P = v_target`` for a non-negative ``P(r)`` by MaxEnt.

    Parameters
    ----------
    kernel : numpy.ndarray
        Dipolar kernel ``K(t, r)`` of shape ``(nt, nr)``.
    r : numpy.ndarray
        Distance axis (Å).
    v_target : numpy.ndarray
        Intramolecular form factor to fit ``(nt,)``.
    sigma : float
        Noise level; sets the weight ``1/sigma**2``.
    alpha : float, optional
        Entropy weight; auto-selected when ``None`` or ``<= 0``.
    n_iter : int
        MaxEnt iterations per solve.
    n_alpha : int
        Number of log-spaced ``alpha`` values sampled for auto-selection.
    method : str
        Auto-selection criterion: ``'discrepancy'`` (default — the smoothest
        solution whose data misfit is within a small tolerance of the best
        achievable, so the fit is as good as an unregularised inversion) or
        ``'lcurve'`` (the L-curve corner). ``'discrepancy'`` is far more robust
        inside the outer fit loop.
    return_lcurve : bool
        When True, also return an ``info`` dict with the sampled ``alphas``,
        residual norms ``rho``, roughness ``eta`` and the selected index.

    Returns
    -------
    (numpy.ndarray, float[, dict])
        The area-normalised distribution ``P(r)``, the ``alpha`` used and,
        when ``return_lcurve`` is set, the L-curve ``info`` dict.
    """
    r = np.asarray(r, dtype=float)
    dr = float(np.mean(np.diff(r))) if r.size > 1 else 1.0
    K = np.asarray(kernel, dtype=float)
    b = np.asarray(v_target, dtype=float)
    # Solve the *unweighted* form-factor problem (as Tikhonov does): the entropy
    # weight ``alpha`` then lives on the same scale for both methods and the
    # discrepancy criterion behaves well. ``sigma`` is accepted for API symmetry
    # but does not rescale the entropy balance.
    w = 1.0
    L = second_derivative_operator(r.size)

    info: dict | None = None
    if alpha is not None and alpha > 1e-8:
        # A concrete, non-trivial alpha was requested (floored for stability).
        p = maxent_inversion(K, b, w, max(float(alpha), 1e-6), n_iter=n_iter)
        alpha = float(alpha)
    else:
        alphas = np.logspace(-3, 1.3, int(n_alpha))  # small -> large smoothing
        rho = np.empty(alphas.size)
        eta = np.empty(alphas.size)
        masses: list[np.ndarray] = []
        for i, a in enumerate(alphas):
            p_i = maxent_inversion(K, b, w, a, n_iter=n_iter)
            masses.append(p_i)
            rho[i] = float(np.linalg.norm(K @ p_i - b))
            eta[i] = float(np.linalg.norm(L @ p_i))

        if str(method).lower() == "lcurve":
            from chisurf.core.math.regularization import discrete_lcurve_corner

            k = discrete_lcurve_corner(rho, eta)
            idx = int(k) if k is not None else int(np.argmin(rho))
        else:
            # Discrepancy: pick the LARGEST alpha (smoothest P) whose misfit is
            # still within 2% of the best achievable, so the reconstruction is
            # as faithful as an unregularised fit (comparable to Tikhonov) while
            # staying as smooth as the data allow.
            rho_min = float(np.min(rho))
            ok = np.where(rho <= rho_min * 1.02)[0]
            idx = int(ok.max()) if ok.size else int(np.argmin(rho))

        alpha = float(alphas[idx])
        # Refine the chosen alpha with a longer, warm-started solve.
        p = maxent_inversion(K, b, w, alpha, p_init=masses[idx], n_iter=n_iter * 3)
        info = {"alphas": alphas, "rho": rho, "eta": eta, "corner": idx}

    # masses (sum=1) -> density, area-normalised on r with the trapezoidal rule
    # (consistent with the Tikhonov path).
    from scipy.integrate import trapezoid

    density = p / dr
    area = float(trapezoid(density, r))
    if area > 0:
        density = density / area
    if return_lcurve:
        return density, float(alpha), info
    return density, float(alpha)
