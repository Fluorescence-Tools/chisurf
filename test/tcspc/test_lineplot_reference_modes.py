from __future__ import annotations

import numpy as np

import chisurf.core.plot_transforms as plot_transforms
from chisurf.gui.widgets.models.tcspc.lifetime import LifetimeModelWidgetBase


class _Curve:
    """Small curve object for transform tests."""

    def __init__(self, x, y):
        """Initialize the curve.

        Parameters
        ----------
        x : array_like
            X-values.
        y : array_like
            Y-values.
        """
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)


class _Fit:
    """Small fit object for transform tests."""

    def __init__(self, x, y):
        """Initialize the fit.

        Parameters
        ----------
        x : array_like
            X-values.
        y : array_like
            Y-values.
        """
        self.data = _Curve(x, y)
        self.xmin = 1
        self.xmax = 3


def _context(model, y, parameters=None, curve_key="data"):
    """Create a plot reference context for TCSPC tests.

    Parameters
    ----------
    model : object
        Model object.
    y : array_like
        Curve values.
    parameters : dict, optional
        Plot-only parameters.
    curve_key : str
        Curve key.

    Returns
    -------
    PlotReferenceContext
        Context object.
    """
    x = np.arange(len(y), dtype=float)
    fit = _Fit(x, y)
    return plot_transforms.PlotReferenceContext(
        fit=fit,
        model=model,
        curve_key=curve_key,
        x=x,
        y=np.asarray(y, dtype=float),
        curves={},
        parameters=parameters or {},
    )


def test_tcspc_total_photon_mode_uses_fit_range_parameter():
    """TCSPC total-photon mode should use optional fit-range denominator."""
    model = LifetimeModelWidgetBase.__new__(LifetimeModelWidgetBase)
    context = _context(model, [1.0, 2.0, 3.0, 4.0], {"fit_range_only": True})

    result = model._tcspc_total_photons_mode(context)

    np.testing.assert_allclose(result.y, np.array([1.0, 2.0, 3.0, 4.0]) / 5.0)


def test_tcspc_peak_photon_mode_uses_peak_denominator():
    """TCSPC peak-photon mode should divide by the selected peak."""
    model = LifetimeModelWidgetBase.__new__(LifetimeModelWidgetBase)
    context = _context(model, [1.0, 2.0, 3.0, 4.0], {"fit_range_only": False})

    result = model._tcspc_peak_photons_mode(context)

    np.testing.assert_allclose(result.y, np.array([1.0, 2.0, 3.0, 4.0]) / 4.0)


def test_tcspc_donor_reference_mode_supports_scaling_parameter():
    """Donor-reference mode should use the plot-controller scaling option."""
    model = LifetimeModelWidgetBase.__new__(LifetimeModelWidgetBase)

    class _Reference:
        """Reference model stub."""

        y = np.array([1.0, 2.0, 4.0])

        def update_model(self):
            """No-op update."""

    model._reference = _Reference()
    context = _context(model, [2.0, 4.0, 8.0], {"scale": "reference_peak"})

    result = model._tcspc_donor_reference_mode(context)

    np.testing.assert_allclose(result.y, np.array([8.0, 8.0, 8.0]))


def test_tcspc_anisotropy_rt_uses_plot_parameters():
    """Anisotropy r(t) mode should use plot-only correction parameters."""
    model = LifetimeModelWidgetBase.__new__(LifetimeModelWidgetBase)

    class _Anisotropy:
        """Anisotropy component stub."""

        g = 1.0
        l1 = 0.0
        l2 = 0.0

        def _extract_vv_vh_raw_for_diag(self):
            """Return simple VV/VH traces."""
            t = np.array([0.0, 1.0])
            return t, np.array([4.0, 4.0]), np.array([1.0, 1.0]), {}

        def _shift_trace_to_reference(self, t, y, delta_t):
            """Return unshifted trace for this test."""
            return y

    model.anisotropy = _Anisotropy()
    context = _context(
        model,
        [0.0, 0.0],
        {
            "g": 1.0,
            "l1": 0.0,
            "l2": 0.0,
            "bg_vv": 0.0,
            "bg_vh": 0.0,
            "vh_shift": 0.0,
            "variant": "corrected",
        },
    )

    result = model._tcspc_anisotropy_rt_mode(context)

    np.testing.assert_allclose(result.y, np.array([0.5, 0.5]))

