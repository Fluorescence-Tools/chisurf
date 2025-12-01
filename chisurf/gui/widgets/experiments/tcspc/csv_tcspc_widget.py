from __future__ import annotations

from chisurf.gui import QtWidgets

import chisurf
import chisurf.gui.decorators
from chisurf.plugins.jordi_g_factor import JordiGFactorCalculator


class CsvTCSPCWidget(QtWidgets.QWidget):

    @chisurf.gui.decorators.init_with_ui("tcspc_csv.ui")
    def __init__(self, *args, **kwargs):
        self.actionDtChanged.triggered.connect(self.onParametersChanged)
        self.actionRebinChanged.triggered.connect(self.onParametersChanged)
        self.actionRepratechange.triggered.connect(self.onParametersChanged)
        self.actionPolarizationChange.triggered.connect(self.onParametersChanged)
        self.actionGfactorChanged.triggered.connect(self.onParametersChanged)
        self.actionIsjordiChanged.triggered.connect(self.onParametersChanged)
        self.actionMatrixColumnsChanged.triggered.connect(self.onParametersChanged)
        self.actionVhShiftChanged.triggered.connect(self.onParametersChanged)
        self.pushButton_inspect.clicked.connect(self.openJordiGFactorPlugin)

    def updateUI(self):
        """Update UI elements based on current_setup properties."""
        import chisurf
        # Get the current setup
        setup = chisurf.cs.current_setup

        # Update is_jordi checkbox
        self.checkBox_3.setChecked(setup.is_jordi)

        # Update matrix_columns line edit
        self.lineEdit.setText(' '.join(map(str, setup.matrix_columns)) if setup.matrix_columns else '')

        # Update g_factor spin box
        self.doubleSpinBox_3.setValue(setup.g_factor)

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
            # Use ChiSurf working directory if available
            try:
                import chisurf as _cs
                start_dir = str(getattr(_cs, 'working_path', '') or '')
            except Exception:
                start_dir = ""
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open Jordi VV/VH file (fast rotating dye)",
                start_dir,
                "Data Files (*.dat *.txt *.csv);;All Files (*)"
            )
            if not file_path:
                return

            # 2) Create plugin widget and load file
            plugin = JordiGFactorCalculator()
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

                    # Ensure parameter propagation if signals are blocked
                    try:
                        self.actionGfactorChanged.trigger()
                    except Exception:
                        pass
                    try:
                        self.actionVhShiftChanged.trigger()
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
        dt = float(
            self.doubleSpinBox_2.value()
        ) * rebin if self.checkBox_2.isChecked() else 1.0 * rebin
        chisurf.run(
            "\n".join(
                [
                    f"cs.current_setup.is_jordi = {is_jordi}",
                    f"cs.current_setup.use_header = {(not is_jordi)}",
                    f"cs.current_setup.matrix_columns = {matrix_columns}",
                    f"cs.current_setup.g_factor = {gfactor:f}",
                    f"cs.current_setup.polarization = '{pol}'",
                    f"cs.current_setup.rep_rate = {rep_rate}",
                    f"cs.current_setup.rebin = ({rebin_x}, {rebin_y})",
                    f"cs.current_setup.vh_shift = {int(self.spinBox_vh_shift.value()) if hasattr(self, 'spinBox_vh_shift') else 0}",
                    f"cs.current_setup.dt = {dt}"
                ]
            )
        )
