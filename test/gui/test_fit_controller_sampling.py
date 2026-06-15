"""Tests for FittingControllerWidget sampling dispatch."""

from pathlib import Path
from types import SimpleNamespace

from qtpy import QtWidgets


def test_proteinmc_sample_button_runs_model_handler(qapp, qtbot, monkeypatch, tmp_path):
    """Sample on ProteinMC must invoke the model's run_sampling handler.

    It must not fall through to the generic emcee server sampler, which
    assumes a curve-based model.
    """
    import chisurf.gui.widgets
    from chisurf.gui.widgets.fitting.fit_controller import FittingControllerWidget

    calls = []

    class ProteinMCModel:
        name = "ProteinMC"

        def run_sampling(self, *, output_directory=None, run_count=1, n_iter=None):
            calls.append({
                "output_directory": output_directory,
                "run_count": run_count,
                "n_iter": n_iter,
            })

    model = ProteinMCModel()
    fit = SimpleNamespace(
        name="ProteinMC fit",
        model=model,
        unique_identifier="uid-1",
        plots=[],
    )

    controller = FittingControllerWidget.__new__(FittingControllerWidget)
    controller.fit = fit
    controller.comboBox = QtWidgets.QComboBox()
    controller.comboBox.addItem("ProteinMC")

    # Mock the directory picker to return our temp path
    monkeypatch.setattr(
        chisurf.gui.widgets,
        "get_directory",
        lambda *args, **kwargs: (tmp_path, ""),
    )

    # Mock spinboxes used by n_steps/n_runs properties
    controller.doubleSpinBox = QtWidgets.QDoubleSpinBox()
    controller.doubleSpinBox.setValue(2.0)  # n_steps = 2000
    controller.spinBox_5 = QtWidgets.QSpinBox()
    controller.spinBox_5.setValue(3)  # n_runs = 3

    controller.onErrorEstimate()

    assert len(calls) == 1, f"Expected one run_sampling call, got {calls}"
    assert Path(calls[0]["output_directory"]) == tmp_path
    assert calls[0]["run_count"] == 3
    assert calls[0]["n_iter"] == 2000


def test_generic_sample_button_still_uses_server_for_curve_models(qapp, qtbot, monkeypatch, tmp_path):
    """Curve models without a custom handler fall through to server sampling.

    This ensures the new model-handler branch does not intercept ordinary
    fits that should use the generic emcee path.
    """
    import chisurf.gui.widgets
    from chisurf.gui.widgets.fitting.fit_controller import FittingControllerWidget

    class CurveModel:
        name = "Curve"

    fit = SimpleNamespace(
        name="Curve fit",
        model=CurveModel(),
        unique_identifier="uid-2",
        plots=[],
    )

    controller = FittingControllerWidget.__new__(FittingControllerWidget)
    controller.fit = fit
    controller.comboBox = QtWidgets.QComboBox()
    controller.comboBox.addItem("Curve")

    monkeypatch.setattr(
        chisurf.gui.widgets,
        "get_directory",
        lambda *args, **kwargs: (tmp_path, ""),
    )

    controller.doubleSpinBox = QtWidgets.QDoubleSpinBox()
    controller.doubleSpinBox.setValue(1.0)
    controller.spinBox_5 = QtWidgets.QSpinBox()
    controller.spinBox_5.setValue(1)

    server_calls = []

    class FakeClient:
        def start_sampling(self, **kwargs):
            server_calls.append(kwargs)
            return {"ok": True}

    monkeypatch.setattr(
        "chisurf.gui.widgets.fitting.fit_controller.get_fitting_client",
        lambda: FakeClient(),
    )

    controller.onErrorEstimate()

    assert len(server_calls) == 1
    assert server_calls[0]["fit_uid"] == "uid-2"
