"""Unit tests for the native DEER dipolar physics (self-contained, numpy/scipy only).

Anchor the reimplemented kernel/distributions/background to analytic limits and
verify the Tikhonov inversion recovers a known distance.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from chisurf.core.models.deer import kernel as K
from chisurf.core.models.deer import tikhonov as T


def test_kernel_zero_time_is_one():
    t = np.linspace(0.0, 3.0, 128)
    r = np.linspace(15.0, 80.0, 100)  # Å
    Kmat = K.dipolar_kernel(t, r)
    assert Kmat.shape == (t.size, r.size)
    assert np.allclose(Kmat[0], 1.0)
    assert np.all(np.isfinite(Kmat))


def test_kernel_matches_numerical_powder_average():
    """The Fresnel closed form equals the direct powder-average integral."""
    t = np.linspace(0.0, 2.5, 64)
    r = np.array([25.0, 35.0, 50.0])  # Å
    Kmat = K.dipolar_kernel(t, r)
    xi = np.linspace(0.0, 1.0, 4001)
    for j, rj in enumerate(r):
        w = K.DIPOLAR_CONSTANT / rj ** 3
        ref = np.array([trapezoid(np.cos((3 * xi ** 2 - 1) * w * ti), xi) for ti in t])
        assert np.allclose(Kmat[:, j], ref, atol=2e-3)


def test_distributions_normalized():
    r = np.linspace(10.0, 100.0, 400)  # Å
    assert abs(trapezoid(K.dd_gauss(r, 35.0, 3.0), r) - 1.0) < 1e-6
    assert abs(trapezoid(K.dd_rice(r, 40.0, 4.0), r) - 1.0) < 1e-6
    p = K.dd_gauss_multi(r, [30.0, 50.0], [2.0, 3.0], [0.6, 0.4])
    assert abs(trapezoid(p, r) - 1.0) < 1e-6


def test_background_limits():
    t = np.linspace(-1.0, 3.0, 50)
    assert np.allclose(K.background(t, "hom3d", 0.0), 1.0)
    assert np.allclose(K.background(t, "strexp", 0.1, d=3.0),
                       K.background(t, "exp", 0.1))  # d=3 -> mono-exponential
    b = K.background(t, "hom3d", 0.2)
    assert np.all((b > 0) & (b <= 1.0 + 1e-9))


def test_deer_signal_shape_and_modulation():
    t = np.linspace(0.0, 3.0, 200)
    r = np.linspace(15.0, 80.0, 150)  # Å
    p = K.dd_gauss(r, 35.0, 3.0)
    V = K.deer_signal(t, r, p, mod_depth=0.4, bg_model="hom3d", bg_k=0.05, scale=1.0)
    assert V.shape == t.shape
    assert np.all(np.isfinite(V))
    assert abs(V[0] - 1.0) < 1e-6           # V(0) = 1 for B(0)=1, form factor 1
    assert V.min() < V[0]                    # modulation dips below the origin


def test_tikhonov_recovers_known_distance():
    t = np.linspace(0.0, 3.0, 256)
    r = np.linspace(15.0, 80.0, 150)  # Å
    Kmat = K.dipolar_kernel(t, r)
    p_true = K.dd_gauss(r, 35.0, 3.0)
    form_factor = trapezoid(Kmat * p_true.reshape(1, -1), r, axis=1)
    p_rec, alpha = T.tikhonov_distance_distribution(Kmat, r, form_factor)
    assert alpha > 0
    assert np.all(p_rec >= -1e-9)
    r_peak = r[int(np.argmax(p_rec))]
    assert abs(r_peak - 35.0) < 2.0


def test_tikhonov_lcurve_selection_recovers_distance():
    """L-curve alpha-selection also recovers the known distance."""
    t = np.linspace(0.0, 3.0, 256)
    r = np.linspace(15.0, 80.0, 150)  # Å
    Kmat = K.dipolar_kernel(t, r)
    form_factor = trapezoid(Kmat * K.dd_gauss(r, 35.0, 3.0).reshape(1, -1), r, axis=1)
    p_rec, alpha = T.tikhonov_distance_distribution(Kmat, r, form_factor, method="lcurve")
    assert alpha > 0
    assert abs(r[int(np.argmax(p_rec))] - 35.0) < 3.0


def test_maxent_recovers_known_distance():
    """The MaxEnt inversion recovers a known distance with L-curve alpha."""
    from chisurf.core.models.deer import maxent as ME

    t = np.linspace(0.0, 3.0, 200)
    r = np.linspace(15.0, 80.0, 120)  # Å
    Kmat = K.dipolar_kernel(t, r)
    form_factor = trapezoid(Kmat * K.dd_gauss(r, 40.0, 3.0).reshape(1, -1), r, axis=1)
    p_rec, alpha = ME.maxent_distance_distribution(Kmat, r, form_factor, sigma=1e-2)
    assert alpha > 0
    assert np.all(p_rec >= -1e-12)
    assert abs(trapezoid(p_rec, r) - 1.0) < 1e-6
    assert abs(r[int(np.argmax(p_rec))] - 40.0) < 4.0
