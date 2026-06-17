import numpy as np
from qtpy import QtWidgets

from chisurf.gui.widgets.fitting.fit_controller import FittingControllerWidget


class _Reader:
    def autofitrange(self, data):
        """Return a deterministic fit range for controller tests.

        Parameters
        ----------
        data : object
            Data object passed by the controller.

        Returns
        -------
        tuple of int
            Fit range indices.
        """
        return 2, 7


class _Data:
    def __init__(self):
        """Create synthetic one-dimensional data for controller tests."""
        self.y = np.arange(10, dtype=float)
        self.data_reader = _Reader()


class _Fit:
    def __init__(self):
        """Create a minimal fit object for auto-fit-range tests."""
        self.data = _Data()
        self.name = "test fit"
        self.unique_identifier = "fit-1"
        self.model = object()
        self.update_count = 0
        self.fit_range_values = []

    @property
    def fit_range(self):
        """Return the last range assigned by the controller.

        Returns
        -------
        tuple of int
            Last assigned range.
        """
        return self.fit_range_values[-1]

    @fit_range.setter
    def fit_range(self, value):
        """Record an assigned fit range.

        Parameters
        ----------
        value : tuple of int
            Fit range assigned by the controller.
        """
        self.fit_range_values.append(tuple(value))

    def update(self):
        """Count deferred fit updates."""
        self.update_count += 1


def _spinbox(qtbot):
    """Create a tracked spin box for a Qt controller test.

    Parameters
    ----------
    qtbot : pytestqt.qtbot.QtBot
        Test helper that owns the widget lifetime.

    Returns
    -------
    QtWidgets.QSpinBox
        Spin box with a broad integer range.
    """
    spinbox = QtWidgets.QSpinBox()
    spinbox.setRange(0, 99)
    qtbot.addWidget(spinbox)
    return spinbox


def test_auto_fit_range_defers_update_and_blocks_spinbox_signals(qtbot, monkeypatch):
    """Verify that auto-fit-range applies the range atomically before redraw."""
    import chisurf.gui.widgets.fitting.fit_controller as fit_controller

    monkeypatch.setattr(fit_controller, "get_fitting_client", lambda: None)

    widget = FittingControllerWidget.__new__(FittingControllerWidget)
    widget.fit = _Fit()
    widget.spinBox_2 = _spinbox(qtbot)
    widget.spinBox_4 = _spinbox(qtbot)
    widget.spinBox = _spinbox(qtbot)
    widget.spinBox_6 = _spinbox(qtbot)
    widget._is_2d_dataset = False
    widget._2d_shape = None
    widget._record_history = lambda *args, **kwargs: None

    signal_count = {"value": 0}

    def count_signal(_value):
        """Record an intermediate spin-box signal.

        Parameters
        ----------
        _value : int
            Emitted spin-box value.
        """
        signal_count["value"] += 1

    widget.spinBox_2.valueChanged.connect(count_signal)
    widget.spinBox.valueChanged.connect(count_signal)

    widget.onAutoFitRange()

    assert widget.fit.fit_range_values == [(2, 7)]
    assert signal_count["value"] == 0
    assert widget._auto_fit_range_in_progress is True
    assert widget.fit.update_count == 0

    qtbot.waitUntil(lambda: widget._auto_fit_range_in_progress is False)

    assert widget.fit.update_count == 1


def test_finalize_iteration_uses_current_fit_parameters_only():
    """Fit-controller finalization should not touch unrelated global parameters."""

    class _Controller:
        def __init__(self):
            """Track whether a controller was finalized."""
            self.finalized = False

        def finalize(self):
            """Mark the controller as finalized."""
            self.finalized = True

    class _Param:
        def __init__(self, name, controller=None):
            """Create a minimal fitting parameter."""
            self.name = name
            self.controller = controller

    class _Model:
        def __init__(self, parameters):
            """Create a minimal model with parameters."""
            self.parameters_all = parameters

    class _Fit:
        def __init__(self, model):
            """Create a minimal fit."""
            self.model = model

    widget = FittingControllerWidget.__new__(FittingControllerWidget)
    local_controller = _Controller()
    unrelated_controller = _Controller()
    fit = _Fit(_Model([_Param("a", local_controller), _Param("b")]))
    unrelated_fit = _Fit(_Model([_Param("c", unrelated_controller)]))
    fit.grouped_fits = []
    unrelated_fit.grouped_fits = []

    finalized = list(widget._iter_fit_parameters_to_finalize(fit))

    assert finalized == [fit.model.parameters_all[0], fit.model.parameters_all[1]]
    assert unrelated_fit.model.parameters_all[0] not in finalized
    assert local_controller.finalized is False
    assert unrelated_controller.finalized is False
