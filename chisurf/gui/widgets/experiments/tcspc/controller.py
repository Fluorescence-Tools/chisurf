from __future__ import annotations

from chisurf.gui import QtWidgets

import pathlib

import chisurf.decorators
import chisurf.gui.decorators
import chisurf.gui.widgets
from chisurf.experiments import reader
import chisurf.data
import chisurf.gui.widgets.fio
import chisurf.gui.widgets.experiments.widgets
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


class TCSPCReaderControlWidget(
    reader.ExperimentReaderController,
    QtWidgets.QWidget
):
    def get_filename(self) -> pathlib.Path:
        return chisurf.gui.widgets.get_filename(
            description='CSV-TCSPC file',
            file_type='All files (*.*)',
            working_path=None
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.layout = layout
        csv_widget = chisurf.gui.widgets.fio.CsvWidget()
        self.layout.addWidget(csv_widget)
        self.csv_tcspc_widget = CsvTCSPCWidget()
        self.layout.addWidget(self.csv_tcspc_widget)

    def updateUI(self):
        """Update UI elements based on current_setup properties."""
        # Call updateUI on the CsvTCSPCWidget
        self.csv_tcspc_widget.updateUI()


class TCSPCTTTRReaderControlWidget(
    QtWidgets.QWidget,
    reader.ExperimentReaderController
):
    def get_filename(self) -> pathlib.Path:
        self.onParametersChanged()
        return chisurf.gui.widgets.get_filename(
            description='TTTR files',
            file_type='All files (*.*)',
            working_path=None
        )

    @chisurf.gui.decorators.init_with_ui(ui_filename="tcspc_tttr.ui")
    def __init__(self, *args, **kwargs):
        self.actionParametersChanged.triggered.connect(self.onParametersChanged)
        self.onParametersChanged()

    def show(self):
        super().show()
        # Call onParametersChanged on show to make sure that the
        # settings match the UI
        self.onParametersChanged()

    def updateUI(self):
        """Update UI elements based on current_setup properties."""
        import chisurf
        # Get the current setup
        setup = chisurf.cs.current_setup

        # Update channel_numbers line edit
        self.lineEdit.setText(', '.join(map(str, setup.channel_numbers)) if hasattr(setup, 'channel_numbers') else '')

        # Update reading_routine combo box
        if hasattr(setup, 'reading_routine'):
            index = self.comboBox.findText(setup.reading_routine)
            if index >= 0:
                self.comboBox.setCurrentIndex(index)

        # Update micro_time_coarsening combo box
        if hasattr(setup, 'micro_time_coarsening'):
            index = self.comboBox_2.findText(str(setup.micro_time_coarsening))
            if index >= 0:
                self.comboBox_2.setCurrentIndex(index)

    def onParametersChanged(self):
        micro_time_coarsening = self.comboBox_2.currentText()
        channel_numbers = self.lineEdit.text()
        reading_routine = self.comboBox.currentText()
        chisurf.run(
            "\n".join(
                [
                    f"cs.current_setup.channel_numbers = np.array([{channel_numbers}], dtype=np.int8)",
                    f"cs.current_setup.reading_routine = '{reading_routine}'",
                    f"cs.current_setup.micro_time_coarsening = {micro_time_coarsening}"
                ]
            )
        )

class TCSPCSimulatorSetupWidget(QtWidgets.QWidget):

    @chisurf.gui.decorators.init_with_ui("tcspc_simulator.ui")
    def __init__(self, *args, **kwargs):
        self.selector = chisurf.gui.widgets.experiments.ExperimentalDataSelector(
            click_close=False,
            parent=self,
            context_menu_enabled=False,
            experiment=chisurf.experiments.types['tcspc']
        )
        self.verticalLayout_2.addWidget(self.selector)
        self.actionParametersChanged.triggered.connect(self.onParametersChanged)
        self.onParametersChanged()

    def get_filename(self) -> pathlib.Path:
        return pathlib.Path(self.lineEdit.text())

    def updateUI(self):
        """Update UI elements based on current_setup properties."""
        import chisurf
        # Get the current setup
        setup = chisurf.cs.current_setup

        # Update sample_name line edit
        if hasattr(setup, 'sample_name'):
            self.lineEdit.setText(setup.sample_name)

        # Update dt spin box
        if hasattr(setup, 'dt'):
            self.doubleSpinBox.setValue(setup.dt)

        # Update n_tac spin box
        if hasattr(setup, 'n_tac'):
            self.spinBox.setValue(setup.n_tac)

        # Update p0 spin box
        if hasattr(setup, 'p0'):
            self.spinBox_2.setValue(setup.p0)

        # Update lifetime_spectrum line edit
        if hasattr(setup, 'lifetime_spectrum'):
            self.lineEdit_2.setText(', '.join(map(str, setup.lifetime_spectrum)) if setup.lifetime_spectrum.size > 0 else '')

    def onParametersChanged(self):
        dt = self.doubleSpinBox.value()
        n_tac = self.spinBox.value()
        p0 = self.spinBox_2.value()
        sample_name = str(self.lineEdit.text())
        lt_text = self.lineEdit_2.text()
        chisurf.run(
            "\n".join(
                [
                    f"cs.current_setup.sample_name = '{sample_name}'",
                    f"cs.current_setup.dt = {dt}",
                    f"cs.current_setup.lifetime_spectrum = np.array([{lt_text}], dtype=np.float64)",
                    f"cs.current_setup.n_tac = {n_tac}",
                    f"cs.current_setup.p0 = {p0}"
                ]
            )
        )
