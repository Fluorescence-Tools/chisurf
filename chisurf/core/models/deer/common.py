from __future__ import annotations

"""Qt-free plot accessors for the DEER models (referenced from ``*.view.json``).

Each accessor takes a fit (or fit group) and returns plain arrays the GUI plot
widgets consume; no Qt imports here.
"""

import numpy as np


def _model(fit_group):
    """Return the selected fit's model, or ``None``."""
    fit = getattr(fit_group, "selected_fit", fit_group)
    return getattr(fit, "model", None)


def get_deer_distance_distribution(fit_group, **kwargs):
    """Return ``(P, r)`` — distribution density and distance axis (Å).

    The distribution plot consumes accessors as ``(y, x)``, so the density is
    returned first and the distance axis (Å) second.
    """
    model = _model(fit_group)
    r = getattr(model, "_r", None)
    p = getattr(model, "_p_r", None)
    if r is None or p is None:
        return np.zeros(0), np.zeros(0)
    return np.asarray(p, dtype=float), np.asarray(r, dtype=float)


def get_deer_pr_ci(fit_group, n_boot: int = 120, **kwargs):
    """Return ``(r, p_best, p_lo, p_hi)`` — P(r) with a bootstrap confidence band.

    Falls back to ``(r, P, P, P)`` (zero-width band) when uncertainty is
    unavailable. Distances are in Å.
    """
    model = _model(fit_group)
    fn = getattr(model, "compute_uncertainty", None)
    if not callable(fn):
        r, p = get_deer_distance_distribution(fit_group)
        return r, p, p, p
    try:
        out = fn(n_boot=n_boot)
    except Exception:
        out = None
    if out is None:
        p, r = get_deer_distance_distribution(fit_group)
        return r, p, p, p
    return out


def get_deer_background(fit_group, **kwargs):
    """Return ``(t, B)`` — the intermolecular background trace over the data axis."""
    from chisurf.core.models.deer.kernel import background as _bg

    model = _model(fit_group)
    if model is None:
        return np.zeros(0), np.zeros(0)
    fit = getattr(fit_group, "selected_fit", fit_group)
    data = getattr(fit, "data", None)
    t = getattr(data, "x", None)
    bg = getattr(model, "background", None)
    mo = getattr(model, "modulation", None)
    if t is None or bg is None or mo is None:
        return np.zeros(0), np.zeros(0)
    t = np.asarray(t, dtype=float)
    tc = t - mo.zero_time
    return t, mo.scale * _bg(tc, bg.model, bg.k, bg.d)
