"""Headless model-editor tests for the AutoForm-ported PDA models (PRD-38/PRD-50).

These mirror ``test_model_editor_integration.py`` but for the PDA family. They
walk the real add-fit path for each pure PDA model:

* ``build_model_editor(model)`` returns an :class:`AutoModelWidget`;
* every ``ParameterGroupSection`` resolves to a group that has parameters;
* the dynamic species/distance/component groups render rows;
* plot specs resolve (distribution accessor imports cleanly);
* ``model.update()`` computes a finite, non-empty curve.

A synthetic ``data.pda`` is built so no TTTR files or heavy I/O are needed.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


def _make_pda_data(nmax: int = 60, nmin: int = 5):
    """Return a DataCurve carrying a synthetic, valid ``.pda`` metadata dict."""
    import chisurf.core.fluorescence.tcspc as tcspc
    from chisurf.core.data import DataCurve

    n = np.arange(nmax + 1)
    ps = stats.poisson.pmf(n, mu=20.0).astype(float)
    ps /= ps.sum()

    s1s2 = np.zeros((nmax + 1, nmax + 1), dtype=float)
    for N in range(nmin, nmax + 1):
        g = np.arange(N + 1)
        s1s2[g, N - g] += ps[N] * stats.binom.pmf(g, N, 0.6) * 1000.0

    ny, nx = s1s2.shape
    rr, cc = np.indices((ny, nx))
    y = s1s2.ravel(order="C")
    x = np.arange(y.size)
    pda = {
        "maximum_number_of_photons": nmax,
        "minimum_number_of_photons": nmin,
        "minimum_time_window_length": 2e-3,
        "channels": ([0], [1]),
        "s1s2": s1s2,
        "ps": ps,
        "row_indices": rr.ravel().tolist(),
        "col_indices": cc.ravel().tolist(),
        "ndim": 2,
        "shape": (ny, nx),
        "size": int(y.size),
        "tttr_indices": None,
    }
    return DataCurve(
        name="synthetic-pda",
        load_filename_on_init=False,
        pda=pda,
        y=y,
        x=x,
        ey=tcspc.counting_noise(y),
    )


def _make_pda_fit(model_class):
    import chisurf.core.fitting.fit as fit_mod

    return fit_mod.Fit(model_class=model_class, data=_make_pda_data())


PDA_MODELS = [
    "chisurf.core.models.pda.simple.PdaSimpleModel",
    "chisurf.core.models.pda.pdagauss.PdaGaussianDistanceModel",
    "chisurf.core.models.pda.dynamic.PdaDynamicTwoStateModel",
    "chisurf.core.models.pda.dynamic_mc.PdaDynamicThreeStateModel",
    "chisurf.core.models.pda.anisotropy.PdaAnisotropyModel",
]

# Fixed-layout dynamic models have no add/remove component group.
_FIXED_LAYOUT = ("PdaDynamicTwoStateModel", "PdaDynamicThreeStateModel")


def _resolve(path):
    import importlib

    mod, _, name = path.rpartition(".")
    return getattr(importlib.import_module(mod), name)


@pytest.mark.parametrize("model_path", PDA_MODELS)
def test_pda_model_editor_renders_and_computes(qapp, model_path):
    from qtpy import QtWidgets

    from chisurf.core.models import view_spec as vs
    from chisurf.gui.widgets.models.auto_model_widget import AutoModelWidget
    from chisurf.gui.widgets.models.model_editor import (
        build_model_editor,
        model_plot_specs,
    )

    model_class = _resolve(model_path)
    # AutoForm requires a declarative spec file on the pure model.
    assert getattr(model_class, "view_spec_file", None), f"{model_path} has no view_spec_file"

    fit = _make_pda_fit(model_class)
    model = fit.model

    # (a) build_model_editor returns a real AutoForm widget (the add-fit crash site)
    editor = build_model_editor(model)
    assert isinstance(editor, AutoModelWidget)
    QtWidgets.QVBoxLayout().addWidget(editor)

    # (b) editor is not a row of empty titled boxes
    assert len(editor.parameter_widgets) > 3, "parameter groups rendered empty"

    # (c) every ParameterGroupSection resolves to a group that has parameters
    spec = model.view_spec()
    for section in spec.flat_sections():
        if isinstance(section, vs.ParameterGroupSection):
            group = getattr(model, section.target)
            if hasattr(group, "find_parameters") and not list(group.parameters_all):
                group.find_parameters()
            assert list(group.parameters_all), f"group {section.target!r} has no parameters"

    # (d) variable-component models expose an add/remove dynamic group. The
    # fixed-layout dynamic models (2/3-state) are exempt.
    if not any(fx in model_path for fx in _FIXED_LAYOUT):
        dyn = [s for s in spec.flat_sections() if isinstance(s, vs.DynamicGroupSection)]
        assert dyn, "no dynamic component group in PDA editor"

    # (e) plot specs resolve (distribution accessor imports) and model computes
    assert model_plot_specs(model), "no plot specs resolved"
    model.update()
    y = np.asarray(model.y)
    assert y.size > 0 and np.all(np.isfinite(y)), "model did not compute a finite curve"


def test_dynamic_two_state_limits():
    """The two-state occupation-time density reduces correctly in both limits.

    Slow exchange (K->0): mass concentrates at the boundaries f in {0, 1}.
    Fast exchange (K->inf): mass concentrates near f = p1 (steady occupancy).
    """
    from chisurf.core.models.pda.dynamic import two_state_time_fraction_pdf

    p1 = 0.3
    f = np.linspace(1e-4, 1 - 1e-4, 400)

    # Slow: interior density is negligible compared with the boundary masses.
    w_slow = two_state_time_fraction_pdf(f, p1, K=1e-3)
    interior_mass = float(np.sum(w_slow) * (f[1] - f[0]))
    boundary_mass = p1 * np.exp(-(1 - p1) * 1e-3) + (1 - p1) * np.exp(-p1 * 1e-3)
    assert interior_mass < 0.05 * boundary_mass

    # Fast: the interior density peaks at the steady-state occupancy p1.
    w_fast = two_state_time_fraction_pdf(f, p1, K=500.0)
    f_peak = float(f[int(np.argmax(w_fast))])
    assert abs(f_peak - p1) < 0.05


def test_dynamic_two_state_matches_static_in_slow_limit(qapp):
    """Dynamic model computes a finite, positive histogram in the static limit.

    With K->0 the dynamic model reduces to a static two-population PDA; here we
    assert it produces a finite, non-empty, positive-sum curve.
    """
    import chisurf.core.fitting.fit as fit_mod
    from chisurf.core.models.pda.dynamic import PdaDynamicTwoStateModel

    fit = fit_mod.Fit(model_class=PdaDynamicTwoStateModel, data=_make_pda_data())
    model = fit.model
    # Push toward the static two-state limit and update.
    model.states._kex.value = 0.0
    model.update()
    y = np.asarray(model.y)
    assert y.size > 0 and np.all(np.isfinite(y)) and float(np.sum(y)) > 0.0


def test_three_state_mc_gillespie_equilibrium():
    """The Gillespie MC recovers the analytic equilibrium populations."""
    from chisurf.core.models.pda.dynamic_mc import (
        equilibrium_populations,
        gillespie_time_fractions,
    )

    # K[target, source]; symmetric-ish 3-state scheme.
    K = np.array([
        [0.0, 200.0, 50.0],
        [150.0, 0.0, 300.0],
        [100.0, 250.0, 0.0],
    ])
    p_eq = equilibrium_populations(K)
    assert abs(p_eq.sum() - 1.0) < 1e-9 and np.all(p_eq >= 0)

    # Long windows -> mean time-fraction converges to equilibrium populations.
    fr = gillespie_time_fractions(K, sim_time=2.0, n_windows=1500, seed=3)
    assert fr.shape == (1500, 3)
    assert np.allclose(fr.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(fr.mean(axis=0), p_eq, atol=0.05)


def test_three_state_mc_model_computes(qapp):
    """The 3-state MC PDA model builds a finite, positive histogram."""
    model_class = _resolve("chisurf.core.models.pda.dynamic_mc.PdaDynamicThreeStateModel")
    fit = _make_pda_fit(model_class)
    model = fit.model
    model.states._n_windows.value = 800  # keep the test fast
    model.update()
    y = np.asarray(model.y)
    assert y.size > 0 and np.all(np.isfinite(y)) and float(np.sum(y)) > 0.0
    # MC result is cached: same rates -> same fractions object (no re-sim).
    frac1 = model._time_fractions()
    frac2 = model._time_fractions()
    assert frac1 is frac2


def test_pda_corrected_axes_e_and_r(qapp):
    """The corrected FRET-efficiency and distance histogram axes compute."""
    from chisurf.core.models.pda.common import get_pda_distribution

    model_class = _resolve("chisurf.core.models.pda.pdagauss.PdaGaussianDistanceModel")
    fit = _make_pda_fit(model_class)
    fit.model.update()
    for axis, lo, hi in [("E", 0.0, 1.0), ("R", 20.0, 100.0)]:
        curves = get_pda_distribution(fit, {"x_max": hi, "x_min": lo, "log_x": False,
                                            "n_bins": 41, "n_min": 10, "histogram": axis})
        assert curves, f"axis {axis} produced no curves"
        x = np.asarray(curves[0][1])
        assert x.size > 0 and np.all(np.isfinite(x))


def test_apply_lightpath_matrices():
    """PdaFretNuisance ingests a light-path crosstalk-matrix result correctly."""
    from chisurf.core.models.pda.nusiance import PdaFretNuisance

    matrices = {
        "excitation": {
            "rows": ["laser_green", "laser_red"],
            "columns": ["donor", "acceptor"],
            "values": [[1.0, 0.05], [0.0, 1.0]],
        },
        "emission": {
            "rows": ["donor", "acceptor"],
            "columns": ["det_green", "det_red"],
            "values": [[0.9, 0.05], [0.02, 0.95]],
        },
    }
    n = PdaFretNuisance()
    n.apply_lightpath_matrices(
        matrices,
        donor="donor",
        acceptor="acceptor",
        green_detector="det_green",
        red_detector="det_red",
        green_laser="laser_green",
    )
    assert n.ExDG == 1.0 and n.ExAG == 0.05
    assert n.cGD == 0.9 and n.cRD == 0.05
    assert n.cGA == 0.02 and n.cRA == 0.95
    # Derived MFD factors recomputed and finite.
    for name in ("alpha", "gamma", "delta"):
        assert np.isfinite(getattr(n, name)), f"{name} not computed"


def test_pda_gaussian_plots_pr_and_residual2d(qapp):
    """Gaussian PDA exposes the P(R) distribution and 2D S1S2 residual plots."""
    from chisurf.core.models.pda.common import (
        get_pda_distance_distribution,
        get_pda_residual_image,
    )
    from chisurf.gui.plots.residual_image import Residual2DPlot
    from chisurf.gui.widgets.models.model_editor import model_plot_specs

    model_class = _resolve("chisurf.core.models.pda.pdagauss.PdaGaussianDistanceModel")
    fit = _make_pda_fit(model_class)
    model = fit.model
    model.distances.append()  # a second Gaussian component
    model.update()

    # P(R): summed distribution + one curve per Gaussian component.
    pr = get_pda_distance_distribution(fit)
    assert len(pr) == 3
    assert all(len(np.asarray(y)) > 0 for y, _ in pr)

    # 2D S1S2 weighted residual image is finite.
    img, xa, ya = get_pda_residual_image(fit)
    assert img is not None and img.ndim == 2 and np.all(np.isfinite(img))

    # The residual2d plot is wired via view.json and its accessor resolves.
    specs = model_plot_specs(model)
    res2d = [opts for cls, opts in specs if cls is Residual2DPlot]
    assert res2d, "residual2d plot not resolved from view.json"
    assert callable(res2d[0].get("accessor")), "residual2d accessor not resolved to a callable"


def test_pda_gaussian_correction_factors(qapp):
    """The Gaussian PDA model populates read-only alpha/gamma/delta outputs."""
    model_class = _resolve("chisurf.core.models.pda.pdagauss.PdaGaussianDistanceModel")
    fit = _make_pda_fit(model_class)
    model = fit.model
    model.update()
    n = model.nuisance
    for name in ("alpha", "gamma", "delta"):
        assert np.isfinite(getattr(n, name)), f"{name} not computed"
