from __future__ import annotations

import numpy as np

from chisurf.core.models.fcs.maxent import compute_fcs_maxent_l_curve


def test_compute_fcs_maxent_l_curve_core() -> None:
    """Compute an FCS MaxEnt L-curve from core code."""
    tau = np.logspace(-3, 1, 16)
    td = 0.05
    g = 1.0 + 1.0 / (1.0 + tau / td) / np.sqrt(1.0 + tau / (td * 3.5**2))

    result = compute_fcs_maxent_l_curve(
        tau,
        g,
        log10_min=-2.0,
        log10_max=1.0,
        n_points=5,
        n_td=8,
        num_iter=5,
    )

    assert result.log10_reg.shape == (5,)
    assert result.reg.shape == (5,)
    assert result.chi2r.shape == (5,)
    assert result.solution_norm.shape == (5,)
    assert np.all(np.isfinite(result.log10_reg))
    assert np.all(result.reg > 0.0)
