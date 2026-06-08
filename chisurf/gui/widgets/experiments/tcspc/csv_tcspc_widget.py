from __future__ import annotations

import importlib

from chisurf.gui import QtWidgets

import chisurf as cs
import chisurf.gui.decorators
import chisurf.gui.widgets


def _load_jordi_gfactor_calculator_class():
    mod = importlib.import_module("chisurf.plugins.jordi_g_factor")
    cls = getattr(mod, "JordiGFactorCalculator", None)
    if cls is None:
        raise ImportError("JordiGFactorCalculator not found in chisurf.plugins.jordi_g_factor")
    return cls


class CsvTCSPCWidget(QtWidgets.QWidget):

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _sync_l2_from_l1(self) -> None:
        if not hasattr(self, 'checkBox_link_l1_l2') or not self.checkBox_link_l1_l2.isChecked():
            return
        if not hasattr(self, 'doubleSpinBox_l1') or not hasattr(self, 'doubleSpinBox_l2'):
            return
        l1 = self._safe_float(self.doubleSpinBox_l1.value(), 0.0)
        if abs(self._safe_float(self.doubleSpinBox_l2.value(), 0.0) - l1) > 1e-15:
            self.doubleSpinBox_l2.blockSignals(True)
            self.doubleSpinBox_l2.setValue(l1)
            self.doubleSpinBox_l2.blockSignals(False)

    def _update_l2_enabled_state(self) -> None:
        if not hasattr(self, 'doubleSpinBox_l2'):
            return
        jordi_enabled = True
        if hasattr(self, 'checkBox_3'):
            jordi_enabled = bool(self.checkBox_3.isChecked())
        linked = bool(getattr(self, 'checkBox_link_l1_l2', None) and self.checkBox_link_l1_l2.isChecked())
        self.doubleSpinBox_l2.setEnabled(jordi_enabled and (not linked))

    def _on_link_l1_l2_toggled(self, checked: bool) -> None:
        self._update_l2_enabled_state()
        if checked:
            self._sync_l2_from_l1()
            self.onParametersChanged()

    @cs.gui.decorators.init_with_ui("tcspc_csv.ui")
    def __init__(self, *args, **kwargs):
        self.actionDtChanged.triggered.connect(self.onParametersChanged)
        self.actionRebinChanged.triggered.connect(self.onParametersChanged)
        self.actionRepratechange.triggered.connect(self.onParametersChanged)
        self.actionPolarizationChange.triggered.connect(self.onParametersChanged)
        self.actionGfactorChanged.triggered.connect(self.onParametersChanged)
        if hasattr(self, 'actionL1L2Changed'):
            self.actionL1L2Changed.triggered.connect(self.onParametersChanged)
        if hasattr(self, 'checkBox_link_l1_l2'):
            self.checkBox_link_l1_l2.setChecked(False)
            self.checkBox_link_l1_l2.toggled.connect(self._on_link_l1_l2_toggled)
        if hasattr(self, 'checkBox_3'):
            self.checkBox_3.toggled.connect(lambda *_: self._update_l2_enabled_state())
        if hasattr(self, 'doubleSpinBox_l1'):
            self.doubleSpinBox_l1.valueChanged.connect(lambda *_: self._sync_l2_from_l1())
        self.actionIsjordiChanged.triggered.connect(self.onParametersChanged)
        self.actionMatrixColumnsChanged.triggered.connect(self.onParametersChanged)
        self.actionVhShiftChanged.triggered.connect(self.onParametersChanged)
        self.pushButton_inspect.clicked.connect(self.openJordiGFactorPlugin)

    def updateUI(self):
        """Update UI elements based on current_setup properties."""
        # Get the current setup
        setup = cs.cs.current_setup

        # Update is_jordi checkbox
        self.checkBox_3.setChecked(setup.is_jordi)

        # Update matrix_columns line edit
        self.lineEdit.setText(' '.join(map(str, setup.matrix_columns)) if setup.matrix_columns else '')

        # Update g_factor spin box
        self.doubleSpinBox_3.setValue(setup.g_factor)

        # Update l1/l2 controls
        try:
            l1 = float(getattr(setup, 'l1', 0.0) or 0.0)
        except Exception:
            l1 = 0.0
        try:
            l2 = float(getattr(setup, 'l2', 0.0) or 0.0)
        except Exception:
            l2 = 0.0
        if hasattr(self, 'doubleSpinBox_l1'):
            self.doubleSpinBox_l1.setValue(l1)
        if hasattr(self, 'doubleSpinBox_l2'):
            self.doubleSpinBox_l2.setValue(l2)
        if hasattr(self, 'checkBox_link_l1_l2') and self.checkBox_link_l1_l2.isChecked():
            self._sync_l2_from_l1()
        self._update_l2_enabled_state()

        # Update polarization radio buttons
        pol = setup.polarization
        if pol == 'vv':
            self.radioButton_3.setChecked(True)
        elif pol == 'vh':
            self.radioButton_2.setChecked(True)
        elif pol == 'vv/vh':
            self.radioButton_4.setChecked(True)
        else:  # 'vm'
            self.radioButton.setChecked(True)

        # Update rep_rate spin box
        self.doubleSpinBox.setValue(setup.rep_rate)

        # Update rebin combo boxes
        rebin_x, rebin_y = setup.rebin
        # Find and set the index for rebin_y
        index_y = self.comboBox.findText(str(rebin_y))
        if index_y >= 0:
            self.comboBox.setCurrentIndex(index_y)

        # Find and set the index for rebin_x
        index_x = self.comboBox_2.findText(str(rebin_x))
        if index_x >= 0:
            self.comboBox_2.setCurrentIndex(index_x)

        # Update dt spin box
        # Note: We need to handle the case where dt is scaled by rebin
        if self.checkBox_2.isChecked():
            self.doubleSpinBox_2.setValue(setup.dt / rebin_y)
        else:
            self.doubleSpinBox_2.setValue(setup.dt)

        # Update VH shift spinbox if present
        if hasattr(self, 'spinBox_vh_shift') and hasattr(setup, 'vh_shift'):
            try:
                self.spinBox_vh_shift.setValue(int(setup.vh_shift))
            except Exception:
                pass

    def openJordiGFactorPlugin(self):
        """
        Launch the Jordi G-Factor Calculator plugin.
        - Ask user to select a Jordi file (fast rotating dye)
        - Let user adjust parameters (g-factor, VH shift) in the plugin
        - On acceptance, update this controller's g-factor and VH shift
        """
        try:
            # 1) Ask for Jordi file
            file_path = cs.gui.widgets.get_filename(
                description="Open Jordi VV/VH file (fast rotating dye)",
                file_type="Data Files (*.dat *.txt *.csv);;All Files (*)",
            )
            if not file_path or not getattr(file_path, 'name', ''):
                return
            file_path = str(file_path)

            # 2) Create plugin widget dynamically and load file
            JordiGFactorCalculator = _load_jordi_gfactor_calculator_class()
            plugin = JordiGFactorCalculator()

            # Pre-fill FP dt [ns/ch] from current CSV reader settings.
            try:
                dt_base = float(self.doubleSpinBox_2.value()) if hasattr(self, 'doubleSpinBox_2') else float(getattr(cs.cs.current_setup, 'dt', 1.0))
                rebin_y = int(self.comboBox.currentText()) if hasattr(self, 'comboBox') else 1
                use_scaled_dt = bool(self.checkBox_2.isChecked()) if hasattr(self, 'checkBox_2') else False
                dt_ns = dt_base * rebin_y if use_scaled_dt else dt_base
                if hasattr(plugin, 'fp_dt_spinbox') and plugin.fp_dt_spinbox is not None:
                    plugin.fp_dt_spinbox.setValue(float(dt_ns))
            except Exception:
                pass

            plugin.load_jordi_file(file_path)

            # 3) Embed in dialog with OK/Cancel
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Jordi G-Factor & Shift Inspector")
            vbox = QtWidgets.QVBoxLayout(dlg)
            vbox.addWidget(plugin)
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dlg)
            vbox.addWidget(btns)
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)

            # Apply values when the dialog finishes (Accepted, Rejected, or closed via window button)
            def apply_from_plugin(*_):
                try:
                    try:
                        g_factor = float(plugin.g_factor) if plugin.g_factor is not None else float(self.doubleSpinBox_3.value())
                    except Exception:
                        g_factor = float(self.doubleSpinBox_3.value())
                    try:
                        vh_shift = int(round(float(plugin.decay_shift)))
                    except Exception:
                        vh_shift = int(self.spinBox_vh_shift.value()) if hasattr(self, 'spinBox_vh_shift') else 0

                    # Update UI controls (emits valueChanged -> triggers actions wired in .ui)
                    self.doubleSpinBox_3.setValue(g_factor)
                    if hasattr(self, 'spinBox_vh_shift'):
                        self.spinBox_vh_shift.setValue(vh_shift)

                    # Optional l1/l2 estimate propagation from plugin (if determined)
                    try:
                        fp_ready = bool(getattr(plugin, 'fp_estimate_available', False))
                        if fp_ready:
                            l1_est = getattr(plugin, 'l1_estimate', None)
                            l2_est = getattr(plugin, 'l2_estimate', None)
                            l1_val = float(l1_est) if l1_est is not None else None
                            l2_val = float(l2_est) if l2_est is not None else None
                            if l1_val is not None and hasattr(self, 'doubleSpinBox_l1'):
                                self.doubleSpinBox_l1.setValue(l1_val)
                            if hasattr(self, 'doubleSpinBox_l2'):
                                if l2_val is not None:
                                    self.doubleSpinBox_l2.setValue(l2_val)
                                elif l1_val is not None:
                                    self.doubleSpinBox_l2.setValue(l1_val)
                            if l1_val is not None:
                                self._sync_l2_from_l1()
                    except Exception:
                        pass

                    # Ensure parameter propagation if signals are blocked
                    try:
                        self.actionGfactorChanged.trigger()
                    except Exception:
                        pass
                    try:
                        self.actionVhShiftChanged.trigger()
                    except Exception:
                        pass
                    try:
                        if hasattr(self, 'actionL1L2Changed'):
                            self.actionL1L2Changed.trigger()
                    except Exception:
                        pass
                except Exception:
                    # Silently ignore application errors to avoid crashing on dialog close
                    pass

            dlg.finished.connect(apply_from_plugin)

            # Execute the dialog; values will be applied on any finish/close
            dlg.exec_()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Jordi Plugin Error", f"Failed to open Jordi plugin: {e}")

    def onParametersChanged(self):
        is_jordi = bool(self.checkBox_3.isChecked())
        try:
            matrix_columns = list(
                map(int, str(self.lineEdit.text()).strip().split(' '))
            )
        except ValueError:
            matrix_columns = []
        gfactor = float(self.doubleSpinBox_3.value())
        l1 = self._safe_float(getattr(cs.cs.current_setup, 'l1', 0.0), 0.0)
        l2 = self._safe_float(getattr(cs.cs.current_setup, 'l2', 0.0), 0.0)
        if hasattr(self, 'doubleSpinBox_l1'):
            l1 = self._safe_float(self.doubleSpinBox_l1.value(), l1)
        if hasattr(self, 'doubleSpinBox_l2'):
            l2 = self._safe_float(self.doubleSpinBox_l2.value(), l2)
        if hasattr(self, 'checkBox_link_l1_l2') and self.checkBox_link_l1_l2.isChecked():
            l2 = l1
            self._sync_l2_from_l1()
        pol = 'vm'
        if self.radioButton_3.isChecked():
            pol = 'vv'
        elif self.radioButton_2.isChecked():
            pol = 'vh'
        elif self.radioButton_4.isChecked():
            pol = 'vv/vh'
        elif self.radioButton.isChecked():
            pol = 'vm'
        rep_rate = self.doubleSpinBox.value()
        rebin_y = int(self.comboBox.currentText())
        rebin_x = int(self.comboBox_2.currentText())
        rebin = int(self.comboBox.currentText())
        dt_base = float(self.doubleSpinBox_2.value())
        dt = dt_base * rebin if self.checkBox_2.isChecked() else dt_base
        cs.run(
            "\n".join(
                [
                    f"cs.current_setup.is_jordi = {is_jordi}",
                    f"cs.current_setup.use_header = {(not is_jordi)}",
                    f"cs.current_setup.matrix_columns = {matrix_columns}",
                    f"cs.current_setup.g_factor = {gfactor:f}",
                    f"cs.current_setup.l1 = {l1:f}",
                    f"cs.current_setup.l2 = {l2:f}",
                    f"cs.current_setup.polarization = '{pol}'",
                    f"cs.current_setup.rep_rate = {rep_rate}",
                    f"cs.current_setup.rebin = ({rebin_x}, {rebin_y})",
                    f"cs.current_setup.vh_shift = {int(self.spinBox_vh_shift.value()) if hasattr(self, 'spinBox_vh_shift') else 0}",
                    f"cs.current_setup.dt = {dt}"
                ]
            )
        )
