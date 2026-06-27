"""Species-resolved correlation must recover the interconversion relaxation (~25 ms)."""

from __future__ import annotations

import numpy as np
import pytest

from chisurf.plugins.fcs.flc_2d.api import species_correlation, species_decay_patterns
from chisurf.plugins.fcs.flc_2d.fit.dynamics import fit_relaxation


def test_fit_relaxation_recovers_known_rate():
    t = np.geomspace(1e-4, 1.0, 200)
    rate = 40.0
    g = 2.0 * np.exp(-rate * t) + 1.0
    out = fit_relaxation(t, g, t_min=1e-4, t_max=1.0)
    assert abs(out["rate"] - rate) / rate < 0.1
    assert abs(out["relaxation_time_s"] - 1.0 / rate) < 0.1 / rate


@pytest.mark.slow
def test_species_correlation_reference(reference_photons):
    """FFCS of the two reference species yields a ~25 ms (40 1/s) relaxation."""
    patterns = species_decay_patterns(
        (1.0, 3.0),
        n_microtime_bins=reference_photons["n_microtime_bins"],
        micro_time_resolution_ns=reference_photons["micro_resolution_ns"],
        irf=reference_photons["irf"],
        irf_time_ns=reference_photons["irf_time_ns"],
    )
    dyn = species_correlation(
        reference_photons["macro_ticks"],
        reference_photons["micro_ticks"],
        species_decays=patterns,
        total_decay=None,
        macro_time_resolution_s=reference_photons["macro_resolution_s"],
        n_microtime_bins=reference_photons["n_microtime_bins"],
        n_bins=8,
        n_casc=25,
    )
    assert dyn.relaxation, "relaxation fit did not converge"
    rate = dyn.relaxation["rate"]
    # ground truth: k12 + k21 = 40 1/s (25 ms). Allow a generous single-exp tolerance.
    assert 25.0 <= rate <= 60.0, f"relaxation rate {rate:.1f}/s outside expected band"

    # the two species cross-correlation must be anti-correlated at short lag
    cross = dyn.correlation.cross[(0, 1)]
    lag = dyn.correlation.lag_s
    g_short = float(np.interp(1e-3, lag, cross))
    g_long = float(np.interp(0.2, lag, cross))
    assert g_short < g_long
