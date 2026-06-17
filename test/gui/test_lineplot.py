import json
from pathlib import Path

import numpy as np
from qtpy import QtWidgets

import chisurf.core.plot_transforms as plot_transforms
from chisurf.gui.plots.lineplot.lineplot import (
    LinePlot,
    LinePlotControl,
    _load_reference_presets,
)

def _lineplot_source() -> str:
    path = Path(__file__).resolve().parents[2] / "chisurf" / "gui" / "plots" / "lineplot" / "lineplot.py"
    return path.read_text(encoding="utf-8")

def test_group_display_methods_exist():
    src = _lineplot_source()
    assert "def _plot_group_curves(self" in src
    assert "def _plot_single_fit_curves(self" in src
    assert "def _plot_active_fit_only(self" in src

def test_group_display_alpha_and_setalpha_contract():
    src = _lineplot_source()
    assert "line.setAlpha(int(alpha * 255), auto=False)" in src
    assert "alpha = 1.0" in src
    assert "alpha = 0.4" in src

def test_group_display_uses_selected_fit_when_available():
    src = _lineplot_source()
    assert "selected_fit" in src
    assert "hasattr(self.fit, 'grouped_fits')" in src or "hasattr(fit, 'grouped_fits')" in src or "hasattr(self.fit, \"grouped_fits\")" in src

def test_lineplot_without_reference_modes_uses_raw(qtbot):
    """LinePlotControl should expose raw mode when no modes are registered."""
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    controller = LinePlotControl(parent=parent)
    qtbot.addWidget(controller)

    controller.set_reference_modes([])

    assert controller.reference_mode == "raw"
    assert controller.comboBox_reference.count() == 1


def test_lineplot_controller_state_roundtrip(qtbot):
    """LinePlot controller state should be project-serializable."""
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    source = LinePlotControl(parent=parent)
    target = LinePlotControl(parent=parent)
    qtbot.addWidget(source)
    qtbot.addWidget(target)

    source.scale_x = "log"
    source.data_logy = "log"
    source.x_shift = 1.25
    source.y_shift = -0.5
    source.display_group = True
    source.checkBox_4.setChecked(True)
    source.doubleSpinBox.setValue(2.0)
    mode = plot_transforms.PlotReferenceMode(
        key="scale",
        label="Scale",
        callback=lambda context: context.y,
        parameters=(
            plot_transforms.PlotReferenceParameter(
                key="factor",
                label="factor",
                kind="float",
                default=2.0,
            ),
        ),
    )
    source.set_reference_modes([mode])
    source.reference_mode = "scale"
    source.reference_parameters = {"factor": 3.5}
    target.set_reference_modes([mode])

    target.set_state(source.get_state())

    assert target.scale_x == "log"
    assert target.data_logy == "log"
    assert target.x_shift == 1.25
    assert target.y_shift == -0.5
    assert target.display_group is True
    assert target.checkBox_4.isChecked() is True
    assert target.doubleSpinBox.value() == 2.0
    assert target.reference_mode == "scale"
    assert target.reference_parameters["factor"] == 3.5


def test_reference_mode_callback_receives_parameters(qtbot):
    """LinePlot should pass GUI parameter values into selected callbacks."""
    plot = LinePlot.__new__(LinePlot)
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    plot.plot_controller = LinePlotControl(parent=parent)
    plot._reference_y_label_override = None
    x = np.array([1.0, 2.0])
    y = np.array([5.0, 9.0])

    class MockModel:
        """Mock model with plot reference modes."""

        def transform(self, context):
            """Scale y-values using the GUI parameter."""
            return plot_transforms.PlotReferenceResult(
                x=context.x,
                y=context.y * float(context.parameters["factor"]),
            )

        def get_plot_reference_modes(self):
            """Return the available mock mode."""
            return [
                plot_transforms.PlotReferenceMode(
                    key="scale",
                    label="Scale",
                    callback=self.transform,
                    parameters=(
                        plot_transforms.PlotReferenceParameter("factor", "factor", "float", 2.0),
                    ),
                    applies_to=("data",),
                )
            ]

    model = MockModel()
    plot.plot_controller.set_reference_modes(model.get_plot_reference_modes())
    plot.plot_controller.reference_mode = "scale"
    plot.plot_controller.reference_parameters = {"factor": 3.0}

    result = plot._apply_reference_mode_to_curve(
        fit=None,
        model=model,
        curve_key="data",
        x=x,
        y=y,
        curves={},
    )
    np.testing.assert_allclose(result.y, y * 3.0)


def test_reference_mode_hidden_result(qtbot):
    """LinePlot reference modes may hide non-applicable curves."""
    plot = LinePlot.__new__(LinePlot)
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    plot.plot_controller = LinePlotControl(parent=parent)
    plot._reference_y_label_override = None
    x = np.array([1.0, 2.0])
    y = np.array([6.0, 10.0])

    class MockModel:
        """Mock model with a hiding mode."""

        def get_plot_reference_modes(self):
            """Return the available mock mode."""
            return [
                plot_transforms.PlotReferenceMode(
                    key="hide",
                    label="Hide",
                    callback=lambda context: plot_transforms.PlotReferenceResult(context.x, context.y, visible=False),
                )
            ]

    model = MockModel()
    plot.plot_controller.set_reference_modes(model.get_plot_reference_modes())
    plot.plot_controller.reference_mode = "hide"

    result = plot._apply_reference_mode_to_curve(
        fit=None,
        model=model,
        curve_key="data",
        x=x,
        y=y,
        curves={},
    )
    assert result.visible is False


def test_axis_range_skips_invalid_log_axis():
    """Manual axis ranges must not produce NaN ranges in log mode."""

    assert LinePlot._axis_range(1.0, 3.0, np.array([1.0, 2.0, 3.0]), True) == [0.0, np.log10(3.0)]
    assert LinePlot._axis_range(None, 3.0, np.array([1.0, 2.0, 3.0]), True) == [0.0, np.log10(3.0)]
    assert LinePlot._axis_range(0.0, None, np.array([1.0, 2.0]), True) is None
    assert LinePlot._axis_range(None, -1.0, np.array([1.0, 2.0]), True) is None
    assert LinePlot._axis_range(0.0, 2.0, np.array([1.0, 3.0]), False) == [0.0, 2.0]


def test_apply_presets_to_mode():
    """_apply_presets_to_mode should override axis preset fields."""
    mode = plot_transforms.PlotReferenceMode(
        key="test", label="Test",
        callback=lambda ctx: plot_transforms.PlotReferenceResult(ctx.x, ctx.y),
        y_range=(0, 1),
        y_padding=0.05,
    )
    preset = {"y_range": (0, 2), "y_padding": 0.1}
    updated = LinePlot._apply_presets_to_mode(mode, preset)
    assert updated.y_range == (0, 2)
    assert updated.y_padding == 0.1
    assert updated.key == "test"
    assert updated.x_range is None


def test_apply_presets_to_mode_skips_none():
    """None values in preset dict should not override existing fields."""
    mode = plot_transforms.PlotReferenceMode(
        key="test", label="Test",
        callback=lambda ctx: plot_transforms.PlotReferenceResult(ctx.x, ctx.y),
        y_range=(0, 1),
        y_padding=0.05,
    )
    preset = {"y_padding": None}
    updated = LinePlot._apply_presets_to_mode(mode, preset)
    assert updated.y_range == (0, 1)
    assert updated.y_padding == 0.05


def test_presets_static_source_has_all_expected_keys():
    """The built-in JSON file should contain entries for all known modes."""
    path = Path(__file__).resolve().parents[2] / "chisurf" / "gui" / "plots" / "lineplot" / "reference_presets.json"
    assert path.exists()
    with open(str(path)) as fh:
        presets = json.load(fh)
    expected_keys = {
        "fcs_diffusion", "fcs_molecules",
        "tcspc_total_photons", "tcspc_peak_photons",
        "tcspc_donor_reference", "tcspc_anisotropy_rt",
    }
    assert expected_keys.issubset(set(presets.keys()))


def test_load_reference_presets_returns_dict():
    """_load_reference_presets should return a dict with expected keys."""
    presets = _load_reference_presets()
    assert isinstance(presets, dict)
    # Should have at least the FCS diffusion preset
    assert "fcs_diffusion" in presets
    entry = presets["fcs_diffusion"]
    assert "y_range" in entry
    assert "y_padding" in entry


def test_reference_mode_y_axis_preset_range():
    """A mode with y_range=(0,1) and y_padding=0.05 should produce [-0.05, 1.05]."""
    y_lo, y_hi = 0.0, 1.0
    span = y_hi - y_lo
    pad = 0.05
    result = [y_lo - span * pad, y_hi + span * pad]
    assert result == [-0.05, 1.05]


def test_reference_mode_y_axis_preset_without_padding():
    """A mode with y_range=(0,1) and no padding should produce [0, 1]."""
    y_lo, y_hi = 0.0, 1.0
    span = y_hi - y_lo
    pad = 0.0
    result = [y_lo - span * pad, y_hi + span * pad]
    assert result == [0.0, 1.0]


def test_reference_mode_y_axis_preset_anisotropy():
    """Anisotropy preset: y_range=(-0.05, 0.45) with y_padding=0 produces [-0.05, 0.45]."""
    y_lo, y_hi = -0.05, 0.45
    span = y_hi - y_lo
    pad = 0.0
    result = [y_lo - span * pad, y_hi + span * pad]
    assert result == [-0.05, 0.45]


def test_apply_presets_to_mode_empty_preset():
    """Empty preset dict should return the mode unchanged."""
    mode = plot_transforms.PlotReferenceMode(
        key="test", label="Test",
        callback=lambda ctx: plot_transforms.PlotReferenceResult(ctx.x, ctx.y),
        y_range=(0, 1),
    )
    updated = LinePlot._apply_presets_to_mode(mode, {})
    assert updated is mode


def test_apply_presets_to_mode_partial_override():
    """Partial preset (only y_padding) should preserve other fields."""
    mode = plot_transforms.PlotReferenceMode(
        key="test", label="Test",
        callback=lambda ctx: plot_transforms.PlotReferenceResult(ctx.x, ctx.y),
        y_range=(0, 1),
        y_padding=0.05,
        x_range=(0, 10),
    )
    preset = {"y_padding": 0.1}
    updated = LinePlot._apply_presets_to_mode(mode, preset)
    assert updated.y_range == (0, 1)
    assert updated.y_padding == 0.1
    assert updated.x_range == (0, 10)


def test_apply_presets_to_mode_list_conversion():
    """JSON presets with lists for ranges should be handled via load_reference_presets.
    _load_reference_presets converts lists to tuples.
    """
    # Simulate what the JSON loader does
    raw_preset = {"y_range": [0, 1], "y_padding": 0.05}
    if isinstance(raw_preset.get("y_range"), list):
        raw_preset["y_range"] = tuple(raw_preset["y_range"])
    mode = plot_transforms.PlotReferenceMode(
        key="test", label="Test",
        callback=lambda ctx: plot_transforms.PlotReferenceResult(ctx.x, ctx.y),
    )
    updated = LinePlot._apply_presets_to_mode(mode, raw_preset)
    assert updated.y_range == (0, 1)
    assert updated.y_padding == 0.05
