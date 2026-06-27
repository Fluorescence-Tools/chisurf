"""Correctness of the numba 2D-FDC builder (core.py) against a brute-force reference.

The 2D fluorescence-decay correlation counts photon pairs separated by a macro-time lag
window ``dT +/- ddT/2`` and histograms them by the two photons' micro-times. These tests
verify the total pair count and the micro-time placement independently of the internal
binning/trimming conventions.
"""

from __future__ import annotations

import numpy as np
import pytest

from chisurf.plugins.fcs.flc_2d.api import two_d_fdc


def _brute_force_pair_count(macro, micro, dT, ddT, tMin, tMax):
    """Replicate the kernel's pair-selection semantics in plain numpy."""
    half = ddT // 2
    valid = (micro > tMin) & (micro < tMax)
    total = 0
    last = macro[-1]
    for i in range(macro.size):
        if not valid[i]:
            continue
        dt_start = macro[i] + dT - half
        dt_end = macro[i] + dT + half
        if dt_end > last:  # kernel breaks once the window passes the last photon
            break
        k0 = np.searchsorted(macro, dt_start, side="left")
        k1 = np.searchsorted(macro, dt_end, side="right")
        total += int(np.count_nonzero(valid[k0:k1]))
    return total


def test_fdc_total_count_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 4000
    macro = np.sort(rng.integers(0, 50_000, size=n)).astype(np.int64)
    micro = rng.integers(1, 64, size=n).astype(np.int64)
    dT, ddT, tMin, tMax = 200, 100, 0, 64

    out = two_d_fdc(macro, micro, dT=dT, ddT=ddT, tMin=tMin, tMax=tMax, logt_imax=16)
    expected = _brute_force_pair_count(macro, micro, dT, ddT, tMin, tMax)

    assert out["mat_lin"].sum() == expected
    assert out["mat_log"].sum() <= expected  # log bins drop tau outside the log range


def test_fdc_single_microtime_lands_in_one_lane():
    """All photons sharing one micro-time => all pair mass in a single row and column."""
    macro = np.arange(0, 2000, 2, dtype=np.int64)
    micro = np.full(macro.size, 20, dtype=np.int64)
    out = two_d_fdc(macro, micro, dT=10, ddT=8, tMin=0, tMax=64, logt_imax=16)
    M = out["mat_lin"]
    nz_rows = np.flatnonzero(M.sum(axis=1))
    nz_cols = np.flatnonzero(M.sum(axis=0))
    assert nz_rows.size == 1 and nz_cols.size == 1
    assert M.sum() > 0


def test_fdc_requires_sorted_macro():
    macro = np.array([5, 1, 9, 3], dtype=np.int64)
    micro = np.array([2, 3, 4, 5], dtype=np.int64)
    with pytest.raises(ValueError):
        two_d_fdc(macro, micro, dT=2, ddT=2, tMin=0, tMax=8, logt_imax=4)


@pytest.mark.slow
def test_fdc_on_reference_subset(reference_photons):
    """Total pair count matches brute force on a slice of the real simulated stream."""
    n = 30_000
    macro = np.ascontiguousarray(reference_photons["macro_ticks"][:n])
    micro = np.ascontiguousarray(reference_photons["micro_ticks"][:n])
    dT, ddT, tMin, tMax = 1000, 2000, 1, 3127
    out = two_d_fdc(macro, micro, dT=dT, ddT=ddT, tMin=tMin, tMax=tMax, logt_imax=60)
    expected = _brute_force_pair_count(macro, micro, dT, ddT, tMin, tMax)
    assert out["mat_lin"].sum() == expected
