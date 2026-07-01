"""Headless model-editor tests for the AutoForm DEER models (PRD-38).

Mirror ``test_pda_model_editor.py`` for the native DEER family:

* ``build_model_editor(model)`` returns an :class:`AutoModelWidget`;
* every ``ParameterGroupSection`` resolves to a group that has parameters;
* the Gaussian model exposes an add/remove dynamic component group;
* plot specs resolve (distance-distribution accessor imports cleanly);
* ``model.update()`` computes a finite, non-empty curve;
* a least-squares run recovers a known mean distance.

Synthetic DEER data is built from the native kernel so no files are needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from chisurf.core.models.deer import kernel as K


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_deer_data(r_mean: float = 40.0, sigma: float = 3.0, lam: float = 0.35,
                    bg_k: float = 0.05, noise: float = 0.0):
    """Return a DataCurve carrying a synthetic DEER trace + ``meta_data['deer']`` (Å)."""
    from chisurf.core.data import DataCurve

    t = np.linspace(0.0, 3.0, 256)
    r = np.linspace(15.0, 80.0, 200)  # Å
    p = K.dd_gauss(r, r_mean, sigma)
    v = K.deer_signal(t, r, p, mod_depth=lam, bg_model="hom3d", bg_k=bg_k, scale=1.0)
    if noise:
        rng = np.random.default_rng(1)
        v = v + rng.normal(0.0, noise, size=v.shape)
    deer = {"t": t, "V": v, "V_imag": np.zeros_like(t), "phase": 0.0,
            "t0": 0.0, "exp_type": "4pDEER", "scale": 1.0}
    return DataCurve(name="synthetic-deer", load_filename_on_init=False,
                     x=t, y=v, ey=np.full_like(v, max(noise, 1e-3)),
                     meta_data={"deer": deer})


def _make_deer_fit(model_class, **kw):
    import chisurf.core.fitting.fit as fit_mod

    return fit_mod.Fit(model_class=model_class, data=_make_deer_data(**kw))


DEER_MODELS = [
    "chisurf.core.models.deer.deer.DeerGaussianModel",
    "chisurf.core.models.deer.deer.DeerRiceModel",
    "chisurf.core.models.deer.deer.DeerTikhonovModel",
    "chisurf.core.models.deer.deer.DeerMaxEntModel",
]


def _resolve(path):
    import importlib

    mod, _, name = path.rpartition(".")
    return getattr(importlib.import_module(mod), name)


@pytest.mark.parametrize("model_path", DEER_MODELS)
def test_deer_model_editor_renders_and_computes(qapp, model_path):
    from qtpy import QtWidgets

    from chisurf.core.models import view_spec as vs
    from chisurf.gui.widgets.models.auto_model_widget import AutoModelWidget
    from chisurf.gui.widgets.models.model_editor import (
        build_model_editor,
        model_plot_specs,
    )

    model_class = _resolve(model_path)
    assert getattr(model_class, "view_spec_file", None), f"{model_path} has no view_spec_file"

    fit = _make_deer_fit(model_class)
    model = fit.model

    # (a) build_model_editor returns a real AutoForm widget (add-fit crash site)
    editor = build_model_editor(model)
    assert isinstance(editor, AutoModelWidget)
    QtWidgets.QVBoxLayout().addWidget(editor)

    # (b) editor is not a row of empty titled boxes
    assert len(editor.parameter_widgets) > 1, "parameter groups rendered empty"

    # (c) every ParameterGroupSection resolves to a group that has parameters
    spec = model.view_spec()
    for section in spec.flat_sections():
        if isinstance(section, vs.ParameterGroupSection):
            group = getattr(model, section.target)
            if hasattr(group, "find_parameters") and not list(group.parameters_all):
                group.find_parameters()
            assert list(group.parameters_all), f"group {section.target!r} has no parameters"

    # (d) the Gaussian model exposes a dynamic component group
    if "DeerGaussianModel" in model_path:
        dyn = [s for s in spec.flat_sections() if isinstance(s, vs.DynamicGroupSection)]
        assert dyn, "no dynamic component group in DEER Gaussian editor"

    # (e) plot specs resolve (distribution accessor imports) and model computes
    assert model_plot_specs(model), "no plot specs resolved"
    model.update()
    y = np.asarray(model.y)
    assert y.size > 0 and np.all(np.isfinite(y)), "model did not compute a finite curve"


def test_deer_distance_accessor_returns_distribution(qapp):
    from chisurf.core.models.deer.common import get_deer_distance_distribution

    fit = _make_deer_fit(_resolve(DEER_MODELS[0]))
    fit.model.update()
    p, r = get_deer_distance_distribution(fit)  # plot convention: (y, x)
    assert r.size == p.size and r.size > 0
    assert np.all(np.isfinite(p)) and np.all(p >= -1e-9)
    # Distances are in Ångström (chisurf FRET convention): tens of Å, not nm.
    assert r[0] > 10.0 and r[-1] < 200.0


@pytest.mark.parametrize("model_path", [
    "chisurf.core.models.deer.deer.DeerTikhonovModel",
    "chisurf.core.models.deer.deer.DeerMaxEntModel",
])
def test_deer_lcurve_plot_and_compute(qapp, model_path):
    """The model-free models expose an L-curve plot that renders finite points."""
    from chisurf.gui.plots.lcurve import LCurvePlot
    from chisurf.gui.widgets.models.model_editor import model_plot_specs

    fit = _make_deer_fit(_resolve(model_path))
    model = fit.model
    model.update()

    # compute_lcurve returns residual/roughness arrays + a corner index
    lc = model.compute_lcurve()
    assert lc is not None
    rho, eta = np.asarray(lc["rho"]), np.asarray(lc["eta"])
    assert rho.size == eta.size > 2
    assert np.all(np.isfinite(rho)) and np.all(np.isfinite(eta))

    # the "lcurve" plot key resolves to the LCurvePlot widget and renders
    assert any(cls is LCurvePlot for cls, _ in model_plot_specs(model))
    plot = LCurvePlot(fit=fit)
    plot.update()
    xs = plot._curve.getData()[0]
    assert xs is not None and len(xs) > 2


def test_deer_gaussian_fit_recovers_distance(qapp):
    """A least-squares run recovers the true mean distance from clean data."""
    model_class = _resolve("chisurf.core.models.deer.deer.DeerGaussianModel")
    fit = _make_deer_fit(model_class, r_mean=40.0, sigma=3.0, lam=0.35)
    model = fit.model
    # Start away from the truth (Å).
    model.gaussians._means[0].value = 32.0
    model.modulation._lam.value = 0.25
    fit.xmin, fit.xmax = 0, len(fit.data.y)
    fit.run()
    assert abs(model.gaussians.means[0] - 40.0) < 2.0
