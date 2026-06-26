from __future__ import annotations

import importlib
import pathlib

from chisurf.gui import QtCore, QtWidgets

import chisurf as cs
import chisurf.gui.widgets
import chisurf.gui.widgets.fio
from chisurf.core.experiments.core import reader


def _load_jordi_gfactor_calculator_class():
    mod = importlib.import_module("chisurf.plugins.jordi_g_factor")
    cls = getattr(mod, "JordiGFactorCalculator", None)
    if cls is None:
        raise ImportError(
            "JordiGFactorCalculator not found in chisurf.plugins.jordi_g_factor"
        )
    return cls


class _TcspcL1L2Widget(QtWidgets.QWidget):
    """Polarization-leakage controls (l1/l2 + link) and the Jordi g-factor tool.

    Bound directly to the reader (``model``). l1/l2/vh_shift are written live; the
    Jordi inspector additionally sets ``g_factor`` on the model and emits
    ``changed`` so the editor can rebuild and reflect the new values.
    """

    changed = QtCore.Signal()

    def __init__(self, model, target=None, parent=None, **kwargs):
        super().__init__(parent)
        self._model = model

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setSpacing(3)

        grid.addWidget(QtWidgets.QLabel("l1"), 0, 0)
        self.spin_l1 = QtWidgets.QDoubleSpinBox()
        self.spin_l1.setDecimals(6)
        self.spin_l1.setRange(-10.0, 10.0)
        self.spin_l1.setSingleStep(0.001)
        self.spin_l1.setValue(self._safe_float(getattr(model, "l1", 0.0)))
        self.spin_l1.setToolTip("Polarization leakage correction l1.")
        grid.addWidget(self.spin_l1, 0, 1)

        grid.addWidget(QtWidgets.QLabel("l2"), 0, 2)
        self.spin_l2 = QtWidgets.QDoubleSpinBox()
        self.spin_l2.setDecimals(6)
        self.spin_l2.setRange(-10.0, 10.0)
        self.spin_l2.setSingleStep(0.001)
        self.spin_l2.setValue(self._safe_float(getattr(model, "l2", 0.0)))
        self.spin_l2.setToolTip("Polarization leakage correction l2.")
        grid.addWidget(self.spin_l2, 0, 3)

        self.check_link = QtWidgets.QCheckBox("link")
        self.check_link.setToolTip("Keep l2 equal to l1.")
        grid.addWidget(self.check_link, 0, 4)

        grid.addWidget(QtWidgets.QLabel("VH shift [ch]"), 1, 0)
        self.spin_vh_shift = QtWidgets.QSpinBox()
        self.spin_vh_shift.setRange(-150, 150)
        self.spin_vh_shift.setValue(int(getattr(model, "vh_shift", 0) or 0))
        self.spin_vh_shift.setToolTip(
            "Channel shift applied to the VH decay relative to VV."
        )
        grid.addWidget(self.spin_vh_shift, 1, 1)

        self.btn_jordi = QtWidgets.QPushButton("g-factor")
        self.btn_jordi.setToolTip("Use a reference dye to adjust shift and g-factor.")
        grid.addWidget(self.btn_jordi, 1, 3, 1, 2)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.spin_l1.valueChanged.connect(self._on_l1_changed)
        self.spin_l2.valueChanged.connect(self._on_l2_changed)
        self.check_link.toggled.connect(self._on_link_toggled)
        self.spin_vh_shift.valueChanged.connect(self._on_vh_shift_changed)
        self.btn_jordi.clicked.connect(self._open_jordi_plugin)

        self._update_l2_enabled()

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _update_l2_enabled(self) -> None:
        self.spin_l2.setEnabled(not self.check_link.isChecked())

    def _on_l1_changed(self, v: float) -> None:
        try:
            self._model.l1 = float(v)
        except Exception:
            pass
        if self.check_link.isChecked():
            self.spin_l2.blockSignals(True)
            self.spin_l2.setValue(float(v))
            self.spin_l2.blockSignals(False)
            try:
                self._model.l2 = float(v)
            except Exception:
                pass

    def _on_l2_changed(self, v: float) -> None:
        try:
            self._model.l2 = float(v)
        except Exception:
            pass

    def _on_link_toggled(self, checked: bool) -> None:
        self._update_l2_enabled()
        if checked:
            self.spin_l2.blockSignals(True)
            self.spin_l2.setValue(self.spin_l1.value())
            self.spin_l2.blockSignals(False)
            try:
                self._model.l2 = float(self.spin_l1.value())
            except Exception:
                pass

    def _on_vh_shift_changed(self, v: int) -> None:
        try:
            self._model.vh_shift = int(v)
        except Exception:
            pass

    def _open_jordi_plugin(self) -> None:
        """Launch the Jordi g-factor calculator and apply its results to the model."""
        try:
            file_path = cs.gui.widgets.get_filename(
                description="Open stacked VV/VH file (fast rotating dye)",
                file_type="Data Files (*.dat *.txt *.csv);;All Files (*)",
            )
            if not file_path or not getattr(file_path, "name", ""):
                return
            file_path = str(file_path)

            JordiGFactorCalculator = _load_jordi_gfactor_calculator_class()
            plugin = JordiGFactorCalculator()

            # Pre-fill FP dt [ns/ch] from the reader's effective dt.
            try:
                dt_ns = float(getattr(self._model, "effective_dt", getattr(self._model, "dt", 1.0)))
                if hasattr(plugin, "fp_dt_spinbox") and plugin.fp_dt_spinbox is not None:
                    plugin.fp_dt_spinbox.setValue(float(dt_ns))
            except Exception:
                pass

            plugin.load_jordi_file(file_path)

            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("G-Factor & Shift Inspector")
            vbox = QtWidgets.QVBoxLayout(dlg)
            vbox.addWidget(plugin)
            btns = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dlg
            )
            vbox.addWidget(btns)
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)

            def apply_from_plugin(*_):
                try:
                    try:
                        g_factor = float(plugin.g_factor) if plugin.g_factor is not None else None
                    except Exception:
                        g_factor = None
                    if g_factor is not None:
                        try:
                            self._model.g_factor = g_factor
                        except Exception:
                            pass
                    try:
                        vh_shift = int(round(float(plugin.decay_shift)))
                        self.spin_vh_shift.setValue(vh_shift)
                    except Exception:
                        pass
                    # Optional l1/l2 estimates
                    try:
                        if bool(getattr(plugin, "fp_estimate_available", False)):
                            l1_est = getattr(plugin, "l1_estimate", None)
                            l2_est = getattr(plugin, "l2_estimate", None)
                            if l1_est is not None:
                                self.spin_l1.setValue(float(l1_est))
                            if l2_est is not None:
                                self.spin_l2.setValue(float(l2_est))
                            elif l1_est is not None:
                                self.spin_l2.setValue(float(l1_est))
                    except Exception:
                        pass
                    # Tell the controller to rebuild so the declarative g-factor
                    # field reflects the new value.
                    self.changed.emit()
                except Exception:
                    pass

            dlg.finished.connect(apply_from_plugin)
            dlg.exec_()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "G-Factor Plugin Error", f"Failed to open the g-factor plugin: {e}"
            )


def _register_tcspc_sections() -> None:
    from chisurf.gui.autoform.sections.registry import register_section
    register_section("tcspc_l1l2")(_TcspcL1L2Widget)


class TCSPCReaderControlWidget(
    reader.ExperimentReaderController,
    QtWidgets.QWidget,
):
    def get_filename(self) -> pathlib.Path:
        return cs.gui.widgets.get_filename(
            description="CSV-TCSPC file",
            file_type="All files (*.*)",
            working_path=None,
        )

    def __init__(self, *args, **kwargs):
        _register_tcspc_sections()
        super().__init__(*args, **kwargs)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.layout = layout

        # Generic CSV input controls (columns/skiprows/delimiter) on top.
        self.csv_widget = cs.gui.widgets.fio.CsvWidget()
        self.layout.addWidget(self.csv_widget)

        # Declarative TCSPC-specific settings via AutoForm.
        reader_obj = getattr(self, "experiment_reader", None)
        self._settings_form = None
        self._l1l2_widget = None
        if reader_obj is not None and hasattr(reader_obj, "view_spec"):
            from chisurf.gui.autoform import AutoForm
            self._settings_form = AutoForm(reader_obj, parent=self)
            self.layout.addWidget(self._settings_form)
            self._bind_custom_section()

    def _bind_custom_section(self) -> None:
        if self._settings_form is None:
            return
        widgets = self._settings_form.findChildren(_TcspcL1L2Widget)
        if widgets:
            self._l1l2_widget = widgets[0]
            self._l1l2_widget.changed.connect(self._rebuild_settings_form)

    def _rebuild_settings_form(self) -> None:
        if self._settings_form is None:
            return
        try:
            self._settings_form.rebuild()
        finally:
            self._bind_custom_section()

    def updateUI(self):
        """Refresh the declarative editor from the current reader state."""
        if self._settings_form is not None:
            try:
                self._settings_form.rebuild()
                self._bind_custom_section()
            except Exception:
                pass
