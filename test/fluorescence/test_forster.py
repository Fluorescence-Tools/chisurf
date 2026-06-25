"""PRD-06 Task 1: Förster radius from spectral overlap.

The canonical overlap integral J and R0 = 0.02108·(κ²·Q_D·n⁻⁴·J)^(1/6)·10 [Å]. Tests
anchor the closed-form R0, the physical scaling invariants, fail-loud on bad inputs, and
a realistic Gaussian-spectra sanity range.
"""

from __future__ import annotations

import numpy as np
import pytest

from chisurf.core.fluorescence.fret.forster import (
    forster_radius,
    forster_radius_from_spectra,
    overlap_integral,
)


def _gaussian(x, center, width):
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def test_forster_radius_closed_form_anchor():
    # R0 = 0.02108·(κ²·Q_D·n⁻⁴·J)^(1/6)·10, with J=1e15, Q_D=0.8, κ²=2/3, n=1.33.
    r0 = forster_radius(1.0e15, donor_quantum_yield=0.8)
    assert r0 == pytest.approx(49.6, abs=0.2)


def test_forster_radius_scales_as_J_sixth_root():
    base = forster_radius(1.0e15, donor_quantum_yield=0.8)
    # multiplying J by 2^6 must double R0
    scaled = forster_radius(1.0e15 * (2.0**6), donor_quantum_yield=0.8)
    assert scaled == pytest.approx(2.0 * base, rel=1e-9)


def test_overlap_integral_is_linear_in_acceptor_extinction():
    wl = np.linspace(500.0, 700.0, 401)
    donor = _gaussian(wl, 580.0, 12.0)
    eps_a = 200000.0 * _gaussian(wl, 620.0, 15.0)
    j1 = overlap_integral(wl, donor, eps_a)
    j2 = overlap_integral(wl, donor, 3.0 * eps_a)
    assert j2 == pytest.approx(3.0 * j1, rel=1e-12)


def test_overlap_integral_ignores_donor_emission_scale():
    # donor emission is area-normalized internally -> absolute scale is irrelevant
    wl = np.linspace(500.0, 700.0, 401)
    donor = _gaussian(wl, 580.0, 12.0)
    eps_a = 200000.0 * _gaussian(wl, 620.0, 15.0)
    assert overlap_integral(wl, donor, eps_a) == pytest.approx(
        overlap_integral(wl, 7.3 * donor, eps_a), rel=1e-12
    )


def test_from_spectra_matches_two_step():
    wl = np.linspace(500.0, 750.0, 501)
    donor = _gaussian(wl, 570.0, 14.0)
    eps_a = 250000.0 * _gaussian(wl, 650.0, 18.0)
    r0, j = forster_radius_from_spectra(wl, donor, eps_a, donor_quantum_yield=0.9)
    assert j == pytest.approx(overlap_integral(wl, donor, eps_a), rel=1e-12)
    assert r0 == pytest.approx(
        forster_radius(j, donor_quantum_yield=0.9), rel=1e-12
    )


def test_realistic_pair_gives_sensible_r0():
    # Well-overlapping donor emission / acceptor absorption (donor emits where the
    # acceptor absorbs) with typical dye values -> R0 in a physical range. Poorly
    # overlapping spectra correctly give a *small* R0; good overlap gives ~tens of Å.
    # Broad physical envelope: catches unit/prefactor bugs (e.g. a nm-vs-Å factor of
    # 10) without being fragile to the exact synthetic spectral shape. The precise
    # formula is anchored by test_forster_radius_closed_form_anchor.
    wl = np.linspace(450.0, 800.0, 701)
    donor_em = _gaussian(wl, 570.0, 25.0)            # donor emission band
    eps_acc = 270000.0 * _gaussian(wl, 600.0, 30.0)  # acceptor ε(λ), ε_max ~2.7e5
    r0, _ = forster_radius_from_spectra(wl, donor_em, eps_acc, donor_quantum_yield=0.8)
    assert 20.0 < r0 < 120.0


def test_zero_donor_emission_raises():
    wl = np.linspace(500.0, 700.0, 201)
    with pytest.raises(ValueError, match="non-positive area"):
        overlap_integral(wl, np.zeros_like(wl), np.ones_like(wl))


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="same length"):
        overlap_integral(np.linspace(500, 700, 10), np.ones(10), np.ones(9))


def test_negative_inputs_raise():
    with pytest.raises(ValueError):
        forster_radius(-1.0, donor_quantum_yield=0.8)
    with pytest.raises(ValueError):
        forster_radius(1e15, donor_quantum_yield=0.8, refractive_index=0.0)
