from __future__ import annotations

"""Thin compute wrappers for the DEER models (Qt-free).

These build the distance distribution and the time-domain signal from plain
numbers, delegating the physics to :mod:`chisurf.core.models.deer.kernel` and
:mod:`chisurf.core.models.deer.tikhonov`. Keeping them here (separate from the
``FittingParameterGroup``/``ModelCurve`` glue in ``deer.py``) mirrors the
RICS/PDA split and makes the math independently unit-testable.
"""

import numpy as np

from .kernel import dd_gauss_multi, dd_rice, deer_signal, dipolar_kernel
from .tikhonov import tikhonov_distance_distribution


def default_distance_axis(t_max: float, n: int = 150) -> np.ndarray:
    """Return a sensible distance grid (Å) for a trace of length ``t_max`` µs.

    The reliably resolvable upper distance scales as ``(t_max)**(1/3)``; the
    classic DEER rule of thumb ``r_max = 5 * (t_max/2)**(1/3)`` nm is used,
    expressed here in Ångström (× 10).
    """
    t_max = max(float(t_max), 1e-3)
    r_max = float(np.clip(50.0 * (t_max / 2.0) ** (1.0 / 3.0), 30.0, 120.0))
    return np.linspace(15.0, r_max, int(n))


def gaussian_signal(t, r, means, sigmas, amplitudes, mod_depth,
                    bg_model, bg_k, bg_d, scale, kernel=None):
    """Time-domain signal for a (multi-)Gaussian distance distribution."""
    p = dd_gauss_multi(r, means, sigmas, amplitudes)
    return deer_signal(t, r, p, mod_depth, bg_model, bg_k, bg_d, scale, kernel), p


def rice_signal(t, r, nu, sigma, mod_depth, bg_model, bg_k, bg_d, scale, kernel=None):
    """Time-domain signal for a 3D-Rice distance distribution."""
    p = dd_rice(r, nu, sigma)
    return deer_signal(t, r, p, mod_depth, bg_model, bg_k, bg_d, scale, kernel), p


def _form_factor(v_data, b, lam, scale):
    """Return the intramolecular form factor ``F = (V/(scale*B) - (1-lam))/lam``."""
    s = float(scale) if scale else 1.0
    return (np.asarray(v_data, dtype=float) / (s * np.clip(b, 1e-9, None)) - (1.0 - lam)) / lam


def tikhonov_signal(t, r, v_data, mod_depth, bg_model, bg_k, bg_d, scale,
                    alpha=None, method="gcv", kernel=None):
    """Model-free signal: invert P(r) by Tikhonov regularisation, rebuild ``V(t)``.

    The background ``B(t)`` and modulation depth ``lambda`` come from the outer
    fit; ``P(r)`` is obtained by a non-negative Tikhonov inversion of the
    intramolecular form factor. ``method`` selects the automatic ``alpha``
    criterion (``'gcv'`` or ``'lcurve'``) when ``alpha`` is not given.
    """
    from .kernel import background as _bg

    r = np.asarray(r, dtype=float)
    k_mat = dipolar_kernel(t, r) if kernel is None else kernel
    b = _bg(t, bg_model, bg_k, bg_d)
    lam = float(np.clip(mod_depth, 1e-3, 1.0))
    s = float(scale) if scale else 1.0
    form_factor = _form_factor(v_data, b, lam, s)
    p, alpha_used = tikhonov_distance_distribution(
        k_mat, r, form_factor, alpha=alpha, method=method)
    v_model = deer_signal(t, r, p, lam, bg_model, bg_k, bg_d, s, kernel=k_mat)
    return v_model, p, alpha_used


def maxent_signal(t, r, v_data, mod_depth, bg_model, bg_k, bg_d, scale,
                  sigma=1.0, alpha=None, kernel=None):
    """Model-free signal via maximum-entropy inversion of ``P(r)``.

    Like :func:`tikhonov_signal` but uses the MaxEnt inversion in
    :mod:`chisurf.core.models.deer.maxent`; ``alpha`` is L-curve-selected when
    not given.
    """
    from .kernel import background as _bg
    from .maxent import maxent_distance_distribution

    r = np.asarray(r, dtype=float)
    k_mat = dipolar_kernel(t, r) if kernel is None else kernel
    b = _bg(t, bg_model, bg_k, bg_d)
    lam = float(np.clip(mod_depth, 1e-3, 1.0))
    s = float(scale) if scale else 1.0
    form_factor = _form_factor(v_data, b, lam, s)
    # Noise on the form factor is amplified from the V-space noise by 1/lambda.
    sigma_form = float(sigma) / lam
    p, alpha_used = maxent_distance_distribution(
        k_mat, r, form_factor, sigma=sigma_form, alpha=alpha)
    v_model = deer_signal(t, r, p, lam, bg_model, bg_k, bg_d, s, kernel=k_mat)
    return v_model, p, alpha_used
