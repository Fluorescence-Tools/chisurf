from __future__ import annotations

import numpy as np

import chisurf.core.plot_transforms as plot_transforms
from chisurf.core.fluorescence.fcs import fcs_diffusion_reference
from chisurf.gui.widgets.models.fcs.parse_fcs_widget import ParseFCSWidget


class _Parameter:
    """Small fitting-parameter wrapper used by FCS reference tests."""

    def __init__(self, name: str, value: float):
        """Initialize the parameter wrapper.

        Parameters
        ----------
        name : str
            Parameter name.
        value : float
            Parameter value.
        """
        self.name = name
        self.value = value


def _fcs_widget() -> ParseFCSWidget:
    """Create a ParseFCSWidget without initializing Qt widgets."""
    widget = ParseFCSWidget.__new__(ParseFCSWidget)
    x = np.array([0.0, 0.25, 1.0])
    widget.__dict__["d"] = [x, np.zeros_like(x)]
    widget.__dict__["_parameters"] = [
        _Parameter("N", 2.0),
        _Parameter("td", 0.5),
        _Parameter("s", 3.5),
        _Parameter("b", 1.2),
    ]
    widget.x = x
    return widget


def test_fcs_diffusion_reference_excludes_baseline():
    """Verify the FCS diffusion mode uses only ``diffusion / abs(N)``."""
    widget = _fcs_widget()
    x = widget.x
    expected = (
        1.0 / abs(2.0)
        * (1.0 + x / 0.5) ** (-1.0)
        * (1.0 + x / (3.5 * 3.5 * 0.5)) ** (-0.5)
    )

    reference = fcs_diffusion_reference(
        x,
        {"N": 2.0, "td": 0.5, "s": 3.5, "b": 1.2},
    )
    np.testing.assert_allclose(reference, expected)


def test_fcs_diffusion_mode_normalizes_as_g_minus_b_over_gdiff():
    """Verify FCS diffusion mode uses ``(G - b) / Gdiff``."""
    widget = _fcs_widget()
    g = np.array([2.2, 2.0, 1.7])
    mode = {mode.key: mode for mode in widget.get_plot_reference_modes()}["fcs_diffusion"]
    context = plot_transforms.PlotReferenceContext(
        fit=None,
        model=widget,
        curve_key="data",
        x=widget.x,
        y=g,
        curves={},
        parameters={"b": 1.2},
    )

    result = mode.callback(context)
    gdiff = fcs_diffusion_reference(
        widget.x,
        {"N": 2.0, "td": 0.5, "s": 3.5, "b": 1.2},
    )
    expected = (g - 1.2) / gdiff

    np.testing.assert_allclose(result.y, expected)


def test_fcs_molecule_mode_uses_parameter_overrides():
    """Verify FCS molecule mode uses plot-controller N and b values."""
    widget = _fcs_widget()
    g = np.array([2.2, 2.0, 1.7])
    mode = {mode.key: mode for mode in widget.get_plot_reference_modes()}["fcs_molecules"]
    context = plot_transforms.PlotReferenceContext(
        fit=None,
        model=widget,
        curve_key="data",
        x=widget.x,
        y=g,
        curves={},
        parameters={"N": 4.0, "b": 1.0},
    )

    result = mode.callback(context)

    np.testing.assert_allclose(result.y, 4.0 * (g - 1.0))


def test_fcs_reference_missing_parameters_returns_none():
    """Verify FCS reference is unavailable when required parameters are missing."""
    widget = ParseFCSWidget.__new__(ParseFCSWidget)
    x = np.array([0.1, 1.0])
    widget.__dict__["d"] = [x, np.zeros_like(x)]
    widget.__dict__["_parameters"] = [
        _Parameter("td", 0.5),
        _Parameter("s", 3.5),
    ]
    widget.x = x

    mode = {mode.key: mode for mode in widget.get_plot_reference_modes()}["fcs_diffusion"]
    context = plot_transforms.PlotReferenceContext(
        fit=None,
        model=widget,
        curve_key="data",
        x=widget.x,
        y=np.array([1.0, 1.0]),
        curves={},
        parameters={"b": 1.0},
    )

    try:
        mode.callback(context)
    except ValueError:
        missing_reference = True
    else:
        missing_reference = False

    assert missing_reference is True
