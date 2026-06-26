from __future__ import annotations

import pathlib

from qtpy import QtWidgets

import chisurf as cs
import chisurf.gui.widgets
from chisurf.core.experiments.core import reader
from chisurf.gui.widgets.sample_picker import show_sample_picker_dialog


class _FcsColumnsWidget(QtWidgets.QWidget):
    """Compact grid widget for x/y column index + error toggle settings."""

    is_form_field = False

    def __init__(self, model, target=None, parent=None, **kwargs):
        super().__init__(parent)
        self._model = model

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setSpacing(4)

        # Row 0: x col | y col
        grid.addWidget(QtWidgets.QLabel("x col"), 0, 0)
        self._col_x = QtWidgets.QSpinBox()
        self._col_x.setRange(0, 99)
        self._col_x.setValue(getattr(model, "col_x", 0))
        grid.addWidget(self._col_x, 0, 1)

        grid.addWidget(QtWidgets.QLabel("y col"), 0, 2)
        self._col_y = QtWidgets.QSpinBox()
        self._col_y.setRange(0, 99)
        self._col_y.setValue(getattr(model, "col_y", 1))
        grid.addWidget(self._col_y, 0, 3)

        # Row 1: x-error toggle + col | y-error toggle + col
        self._err_x = QtWidgets.QCheckBox("x-error")
        self._err_x.setChecked(getattr(model, "error_x_on", False))
        grid.addWidget(self._err_x, 1, 0)
        self._col_ex = QtWidgets.QSpinBox()
        self._col_ex.setRange(0, 99)
        self._col_ex.setValue(getattr(model, "col_ex", 2))
        grid.addWidget(self._col_ex, 1, 1)

        self._err_y = QtWidgets.QCheckBox("y-error")
        self._err_y.setChecked(getattr(model, "error_y_on", True))
        grid.addWidget(self._err_y, 1, 2)
        self._col_ey = QtWidgets.QSpinBox()
        self._col_ey.setRange(0, 99)
        self._col_ey.setValue(getattr(model, "col_ey", 3))
        grid.addWidget(self._col_ey, 1, 3)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self._col_x.valueChanged.connect(lambda v: setattr(model, "col_x", v))
        self._col_y.valueChanged.connect(lambda v: setattr(model, "col_y", v))
        self._col_ex.valueChanged.connect(lambda v: setattr(model, "col_ex", v))
        self._col_ey.valueChanged.connect(lambda v: setattr(model, "col_ey", v))
        self._err_x.toggled.connect(lambda v: setattr(model, "error_x_on", v))
        self._err_y.toggled.connect(lambda v: setattr(model, "error_y_on", v))


def _register_fcs_sections():
    from chisurf.gui.autoform.sections.registry import register_section
    register_section("fcs_columns")(_FcsColumnsWidget)


class FCSController(reader.ExperimentReaderController, QtWidgets.QWidget):

    def get_filename(self) -> pathlib.Path:
        """Return an FCS filename after optionally assigning a sample."""
        path = cs.gui.widgets.get_filename('FCS-CSV files', file_type=self.file_type)
        if path:
            self._set_reader_sample_id(show_sample_picker_dialog(db=self._db(), parent=self))
        return path

    def _db(self):
        """Return the MFDB connection from the current reader when available."""
        try:
            return self.db
        except Exception:
            return None

    def _set_reader_sample_id(self, sample_id: str | None) -> None:
        """Store the selected sample ID on the underlying FCS reader."""
        reader_obj = getattr(self, "experiment_reader", None)
        if reader_obj is None:
            return
        try:
            reader_obj.sample_id = sample_id
        except Exception:
            pass

    def __init__(
            self,
            file_type='Kristine files (*.cor)',
            *args,
            **kwargs
    ):
        _register_fcs_sections()
        super().__init__(*args, **kwargs)
        self.file_type = file_type

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.layout = layout

        # AutoForm covers Format, Columns, and Noise model panels.
        reader_obj = getattr(self, "experiment_reader", None)
        self._settings_form = None
        if reader_obj is not None and hasattr(reader_obj, "view_spec"):
            from chisurf.gui.autoform import AutoForm
            self._settings_form = AutoForm(reader_obj, parent=self)
            self.layout.addWidget(self._settings_form)

    def updateUI(self):
        """Refresh UI from current reader state."""
        if self._settings_form is not None:
            try:
                self._settings_form.rebuild()
            except Exception:
                pass

    def onParametersChanged(self):
        """Push current GUI parameters into cs.current_setup.

        The noise model is already live-bound via AutoForm; this is a no-op
        kept for API compatibility with the base class call sites.
        """
        pass


__all__ = ["FCSController"]
