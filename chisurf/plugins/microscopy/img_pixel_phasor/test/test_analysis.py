"""Headless tests for the pure-numpy phasor analysis toolkit (PRD-55 / PRD-56 §7)."""

from __future__ import annotations

import numpy as np
import pytest

from chisurf.plugins.microscopy.img_pixel_phasor import analysis


def _semicircle_point(tau_ns: float, frequency_mhz: float) -> tuple[float, float]:
    """(g, s) of a single-exponential lifetime — lies on the universal semicircle."""
    g, s = analysis.lifetime_to_phasor(tau_ns, frequency_mhz)
    return float(g), float(s)


def test_apparent_lifetime_recovers_true_tau_on_semicircle():
    freq = 80.0  # MHz
    for tau in (0.5, 1.0, 2.0, 4.0):
        g, s = _semicircle_point(tau, freq)
        tau_phi, tau_m = analysis.phasor_to_apparent_lifetime(
            np.array([g]), np.array([s]), freq
        )
        assert tau_phi[0] == pytest.approx(tau, rel=1e-6)
        assert tau_m[0] == pytest.approx(tau, rel=1e-6)


def test_apparent_lifetime_matches_phasorpy_reference_values():
    # From phasorpy.lifetime.phasor_to_apparent_lifetime docstring example.
    tau_phi, tau_m = analysis.phasor_to_apparent_lifetime(
        np.array([0.5, 0.5]), np.array([0.5, 0.45]), 80.0
    )
    np.testing.assert_allclose(tau_phi, [1.989, 1.79], atol=1e-2)
    np.testing.assert_allclose(tau_m, [1.989, 2.188], atol=1e-2)


def test_component_fraction_recovers_known_mixture():
    freq = 80.0
    c1 = _semicircle_point(1.0, freq)
    c2 = _semicircle_point(4.0, freq)
    true_f = 0.3
    g = true_f * c1[0] + (1 - true_f) * c2[0]
    s = true_f * c1[1] + (1 - true_f) * c2[1]
    frac = analysis.phasor_component_fraction(np.array([g]), np.array([s]), c1, c2)
    assert frac[0] == pytest.approx(true_f, abs=1e-6)


def test_component_fraction_is_clipped_to_unit_interval():
    c1, c2 = (0.2, 0.3), (0.6, 0.4)
    # A point well beyond c2 projects to < 0 -> clipped to 0.
    frac = analysis.phasor_component_fraction(np.array([1.0]), np.array([0.45]), c1, c2)
    assert 0.0 <= frac[0] <= 1.0


def test_unmix_two_components_matches_fraction():
    freq = 80.0
    c1 = _semicircle_point(1.0, freq)
    c2 = _semicircle_point(4.0, freq)
    true_f = 0.65
    g = np.array([true_f * c1[0] + (1 - true_f) * c2[0]])
    s = np.array([true_f * c1[1] + (1 - true_f) * c2[1]])
    fractions = analysis.phasor_unmix(g, s, [c1, c2])
    assert len(fractions) == 2
    assert fractions[0][0] + fractions[1][0] == pytest.approx(1.0, abs=1e-6)
    assert fractions[0][0] == pytest.approx(true_f, abs=1e-2)


def test_unmix_three_components_sums_to_one():
    freq = 80.0
    comps = [
        _semicircle_point(0.5, freq),
        _semicircle_point(2.0, freq),
        _semicircle_point(6.0, freq),
    ]
    weights = np.array([0.2, 0.5, 0.3])
    g = np.array([sum(w * c[0] for w, c in zip(weights, comps))])
    s = np.array([sum(w * c[1] for w, c in zip(weights, comps))])
    fractions = analysis.phasor_unmix(g, s, comps)
    total = sum(f[0] for f in fractions)
    assert total == pytest.approx(1.0, abs=1e-6)
    assert all(f[0] >= -1e-9 for f in fractions)


def test_median_filter_reduces_variance_preserves_mean():
    rng = np.random.default_rng(0)
    g_true, s_true = 0.4, 0.3
    g = g_true + rng.normal(0, 0.05, size=(32, 32))
    s = s_true + rng.normal(0, 0.05, size=(32, 32))
    gf, sf = analysis.phasor_filter_median(g, s, size=3, repeat=1)
    assert gf.var() < g.var()
    assert sf.var() < s.var()
    assert gf.mean() == pytest.approx(g_true, abs=0.02)
    assert sf.mean() == pytest.approx(s_true, abs=0.02)


def test_median_filter_is_nan_safe():
    g = np.full((8, 8), 0.5)
    s = np.full((8, 8), 0.2)
    g[0, 0] = np.nan
    gf, sf = analysis.phasor_filter_median(g, s, size=3, repeat=1)
    # The NaN neighbourhood is filled from valid neighbours; no NaN propagation.
    assert np.isfinite(gf).all()
    assert gf[1, 1] == pytest.approx(0.5, abs=1e-9)


def test_circular_cursor_selects_exact_pixels():
    g = np.array([[0.10, 0.50], [0.51, 0.90]])
    s = np.array([[0.10, 0.50], [0.49, 0.10]])
    mask = analysis.mask_from_circular_cursor(g, s, center=(0.5, 0.5), radius=0.05)
    expected = np.array([[False, True], [True, False]])
    np.testing.assert_array_equal(mask, expected)


def test_elliptic_cursor_axis_aligned():
    g = np.array([0.5, 0.7, 0.5])
    s = np.array([0.3, 0.3, 0.5])
    mask = analysis.mask_from_elliptic_cursor(
        g, s, center=(0.5, 0.3), radii=(0.25, 0.05)
    )
    # (0.5,0.3) center in; (0.7,0.3) within wide axis; (0.5,0.5) beyond narrow axis.
    np.testing.assert_array_equal(mask, [True, True, False])


def test_pseudo_color_shapes_and_colors():
    m0 = np.array([[True, False], [False, False]])
    m1 = np.array([[False, True], [True, False]])
    rgb = analysis.pseudo_color([m0, m1], colors=[(1, 0, 0), (0, 0, 1)])
    assert rgb.shape == (2, 2, 3)
    np.testing.assert_allclose(rgb[0, 0], [1, 0, 0])
    np.testing.assert_allclose(rgb[0, 1], [0, 0, 1])
    np.testing.assert_allclose(rgb[1, 1], [0, 0, 0])


def test_semicircle_polyline_on_unit_circle():
    x, y = analysis.universal_semicircle_polyline(n_points=101)
    # All points satisfy (x-0.5)^2 + y^2 = 0.25 and s >= 0.
    np.testing.assert_allclose((x - 0.5) ** 2 + y**2, 0.25, atol=1e-9)
    assert (y >= -1e-12).all()


def test_lifetime_markers_on_semicircle():
    freq = 80.0
    g, s = analysis.lifetime_tick_markers(freq, taus=[1.0, 2.0, 4.0])
    np.testing.assert_allclose((g - 0.5) ** 2 + s**2, 0.25, atol=1e-9)


def test_iso_lifetime_contours_structure():
    contours = analysis.iso_lifetime_contours(80.0, taus=[2.0])
    kinds = {c["kind"] for c in contours}
    assert kinds == {"iso_phase", "iso_modulation"}
    for c in contours:
        assert len(c["x"]) == len(c["y"]) > 1


def test_fret_trajectory_starts_at_donor_lifetime():
    freq = 80.0
    tau_d0 = 4.0
    x, y = analysis.fret_trajectory(freq, tau_d0=tau_d0, e_range=(0.0, 0.9))
    g0, s0 = analysis.lifetime_to_phasor(tau_d0, freq)
    assert x[0] == pytest.approx(float(g0), abs=1e-9)
    assert y[0] == pytest.approx(float(s0), abs=1e-9)
    # Trajectory lies on the semicircle (mono-exponential quenched donor).
    np.testing.assert_allclose((np.array(x) - 0.5) ** 2 + np.array(y) ** 2, 0.25, atol=1e-9)


def test_angular_frequency_zero_frequency_raises():
    with pytest.raises(ValueError):
        analysis.phasor_to_apparent_lifetime(np.array([0.5]), np.array([0.5]), 0.0)


# --- ported phasorpy overlays: polar grid / components / cursor / contours -------------
def test_polar_grid_default_has_circles_and_spokes():
    grid = analysis.polar_grid_polylines()
    # default = 3 circles (1/3, 2/3, 1) + 12 spokes
    circles = [g for g in grid if g["x"][0] != 0.0 or len(g["x"]) > 2]
    spokes = [g for g in grid if len(g["x"]) == 2 and g["x"][0] == 0.0]
    assert len(spokes) == 12
    unit = [g for g in grid if g["major"]]
    assert len(unit) == 1  # only the unit circle is major
    # every point of the unit circle lies at radius 1
    u = unit[0]
    np.testing.assert_allclose(np.hypot(u["x"], u["y"]), 1.0, atol=1e-9)


def test_component_mixing_polygon_without_fractions():
    comps = [[0.2, 0.3], [0.6, 0.4], [0.5, 0.1]]
    out = analysis.component_mixing(comps)
    names = [o["name"] for o in out]
    assert "mixing region" in names and "components" in names
    region = next(o for o in out if o["name"] == "mixing region")
    # closed polygon: first == last vertex
    assert (region["x"][0], region["y"][0]) == (region["x"][-1], region["y"][-1])


def test_component_mixing_weighted_average():
    comps = [[0.2, 0.4], [0.8, 0.4]]
    out = analysis.component_mixing(comps, fractions=[0.25, 0.75])
    mixture = next(o for o in out if o["name"] == "mixture")
    assert mixture["x"][0] == pytest.approx(0.65)  # 0.25*0.2 + 0.75*0.8
    assert mixture["y"][0] == pytest.approx(0.4)


def test_component_mixing_rejects_bad_shape():
    with pytest.raises(ValueError):
        analysis.component_mixing([[0.5, 0.3]])  # needs >= 2 components


def test_circular_cursor_polyline_is_closed_circle():
    x, y = analysis.cursor_polyline([0.5, 0.3], kind="circular", radius=0.07)
    np.testing.assert_allclose(np.hypot(x - 0.5, y - 0.3), 0.07, atol=1e-9)


def test_elliptic_cursor_semi_axes():
    x, y = analysis.cursor_polyline([0.4, 0.2], kind="elliptic", radii=[0.1, 0.05], angle=0.0)
    assert (x.max() - x.min()) == pytest.approx(0.2, abs=1e-3)  # 2a (discretized)
    assert (y.max() - y.min()) == pytest.approx(0.1, abs=1e-3)  # 2b (discretized)


def test_elliptic_cursor_requires_radii():
    with pytest.raises(ValueError):
        analysis.cursor_polyline([0.4, 0.2], kind="elliptic")


def test_density_contours_encircle_peak():
    g = np.linspace(0.0, 1.0, 60)[:, None]
    s = np.linspace(0.0, 1.0, 60)[None, :]
    density = np.exp(-((g - 0.5) ** 2 + (s - 0.3) ** 2) / 0.01)
    contours = analysis.density_contours(density, (0.0, 1.0), (0.0, 1.0), levels=3)
    assert contours, "expected at least one contour segment"
    # the highest-level contour should sit near the (0.5, 0.3) peak
    top = max(contours, key=lambda c: c["level"])
    assert np.mean(top["x"]) == pytest.approx(0.5, abs=0.1)
    assert np.mean(top["y"]) == pytest.approx(0.3, abs=0.1)


def test_build_overlays_accepts_new_sets():
    ov = analysis.build_overlays(
        80.0,
        sets=["polar_grid", "components", "cursor"],
        components=[[0.2, 0.3], [0.6, 0.4]],
        fractions=[0.5, 0.5],
        cursors=[{"center": [0.5, 0.3], "radius": 0.05, "name": "gate"}],
    )
    names = {o["name"] for o in ov}
    assert "unit circle" in names and "components" in names and "gate" in names
