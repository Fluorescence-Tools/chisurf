"""Headless model-editor tests for the AutoForm RICS models (PRD-38).

Mirror the PDA editor tests: each pure RICS model builds through the real
AutoForm seam, every parameter-group section renders, the 2D residual plot
resolves, and the model computes a finite surface. A synthetic RICS dataset
(lag grid + ics_mean in ``meta_data['rics']``) avoids any file I/O.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_rics_data():
    """Return a DataCurve with a synthetic RICS lag grid + ics_mean map."""
    from chisurf.core.data import DataCurve
    from chisurf.core.models.rics.models import rics_simple

    xi = np.arange(-8, 9)          # fast (pixel) lags
    psi = np.arange(0, 16)         # slow (line) lags
    pixel_shift, line_shift = np.meshgrid(xi, psi)
    ics_mean = rics_simple(
        line_shift=line_shift.astype(float), pixel_shift=pixel_shift.astype(float),
        n=2.0, diffusion_coefficient=1.5, offset=0.0,
        pixel_duration=11.1, line_duration=3.33, pixel_size=50.0, w_r=0.25, w_z=1.0,
    )
    y = ics_mean.ravel()
    meta = {
        "rics": {
            "line_shift": line_shift.astype(float),
            "pixel_shift": pixel_shift.astype(float),
            "ics_mean": ics_mean,
            "pixel_duration_us": 11.1,
            "line_duration_ms": 3.33,
        }
    }
    return DataCurve(name="synthetic-rics", load_filename_on_init=False,
                     y=y, x=np.arange(y.size, dtype=float), meta_data=meta)


def _make_rics_fit(model_class):
    import chisurf.core.fitting.fit as fit_mod

    return fit_mod.Fit(model_class=model_class, data=_make_rics_data())


RICS_MODELS = [
    "chisurf.core.models.rics.rics.RicsSimpleModel",
    "chisurf.core.models.rics.rics.RicsTripletModel",
    "chisurf.core.models.rics.rics.RicsImmobileModel",
    "chisurf.core.models.rics.rics.RicsFlowModel",
    "chisurf.core.models.rics.rics.RicsFullModel",
    "chisurf.core.models.rics.rics.IcsGaussian2DModel",
]


def _resolve(path):
    import importlib

    mod, _, name = path.rpartition(".")
    return getattr(importlib.import_module(mod), name)


@pytest.mark.parametrize("model_path", RICS_MODELS)
def test_rics_model_editor_renders_and_computes(qapp, model_path):
    from qtpy import QtWidgets

    from chisurf.core.models import view_spec as vs
    from chisurf.gui.plots.residual_image import Residual2DPlot
    from chisurf.gui.widgets.models.auto_model_widget import AutoModelWidget
    from chisurf.gui.widgets.models.model_editor import (
        build_model_editor,
        model_plot_specs,
    )

    model_class = _resolve(model_path)
    assert getattr(model_class, "view_spec_file", None), f"{model_path} has no view_spec_file"

    fit = _make_rics_fit(model_class)
    model = fit.model

    editor = build_model_editor(model)
    assert isinstance(editor, AutoModelWidget)
    QtWidgets.QVBoxLayout().addWidget(editor)
    assert len(editor.parameter_widgets) > 3, "parameter groups rendered empty"

    spec = model.view_spec()
    for section in spec.flat_sections():
        if isinstance(section, vs.ParameterGroupSection):
            group = getattr(model, section.target)
            if hasattr(group, "find_parameters") and not list(group.parameters_all):
                group.find_parameters()
            assert list(group.parameters_all), f"group {section.target!r} has no parameters"

    # 2D residual plot resolves with a callable accessor.
    specs = model_plot_specs(model)
    res2d = [opts for cls, opts in specs if cls is Residual2DPlot]
    assert res2d and callable(res2d[0].get("accessor")), "residual2d accessor not resolved"

    # Model computes a finite 2D surface (and flattened 1D curve).
    model.update()
    y = np.asarray(model.y)
    assert y.size > 0 and np.all(np.isfinite(y))
    assert model.rics_model_2d is not None and model.rics_model_2d.ndim == 2


def test_rics_compute_is_finite_for_adversarial_params():
    """RICS compute functions stay finite for zero/negative optimizer excursions.

    The fit divides by N and the beam waists, so unconstrained excursions to
    zero/negative values previously produced inf/nan and crashed leastsq. The
    functions now take magnitudes / floor divisors (PAM |N|,|D| convention).
    """
    from chisurf.core.models.rics.models import (
        ics_gaussian_2d,
        rics_diffusion_triplet,
        rics_flow,
        rics_full,
        rics_immobile,
        rics_simple,
    )

    xi = np.arange(-6, 7)
    ps, ls = np.meshgrid(xi, xi)
    ps = ps.astype(float)
    ls = ls.astype(float)
    bad = [
        dict(n=0.0, diffusion_coefficient=1.0),
        dict(n=-5.0, diffusion_coefficient=-2.0),
        dict(n=1e-9, diffusion_coefficient=0.0, w_r=0.0, w_z=0.0),
    ]
    for kw in bad:
        for fn, extra in [
            (rics_simple, {}),
            (rics_immobile, {"a_immobile": -3.0}),
            (rics_flow, {"v_x": -50.0, "v_y": 1e4}),
            (rics_diffusion_triplet, {"tauT": -1.0, "aT": 1.5}),
            (rics_full, {"tauT": -1.0, "aT": 1.5, "n_immobile": -2.0, "w_immobile": 0.0, "shift_x": 30.0}),
            (rics_full, {"n_immobile": -2.0, "two_d": True}),
        ]:
            out = fn(line_shift=ls, pixel_shift=ps, **{**kw, **extra})
            assert np.all(np.isfinite(out)), f"{fn.__name__} not finite for {kw}"
    # anisotropic Gaussian with degenerate widths / angle
    g = ics_gaussian_2d(ls, ps, amplitude=-1.0, sigma_1=0.0, sigma_2=0.0, angle=9.0)
    assert np.all(np.isfinite(g))


def test_rics_fit_is_stable(qapp):
    """Each RICS model runs a real fit to completion with finite parameters."""
    import chisurf.core.fitting.fit as fit_mod

    for path in RICS_MODELS:
        fit = fit_mod.Fit(model_class=_resolve(path), data=_make_rics_data())
        m = fit.model
        fit.xmin, fit.xmax = 0, int(np.asarray(fit.data.y).size)
        m.update()
        fit.run()  # must not raise (previously: "array must not contain infs or NaNs")
        assert np.all(np.isfinite(np.asarray(m.y)))
        assert np.isfinite(m.diffusion.n) and np.isfinite(m.diffusion.D)


def test_rics_immobile_and_flow_change_the_surface(qapp):
    """The immobile and flow terms actually perturb the RICS surface."""
    from chisurf.core.models.rics.rics import RicsFlowModel, RicsImmobileModel

    imm = _make_rics_fit(RicsImmobileModel).model
    imm.update()
    base = imm.rics_model_2d.copy()
    imm.immobile._a_imm.value = 0.5
    imm.update()
    assert not np.allclose(base, imm.rics_model_2d), "immobile amplitude had no effect"

    flow = _make_rics_fit(RicsFlowModel).model
    flow.update()
    base_f = flow.rics_model_2d.copy()
    flow.flow._vx.value = 200.0
    flow.update()
    assert not np.allclose(base_f, flow.rics_model_2d), "flow velocity had no effect"
