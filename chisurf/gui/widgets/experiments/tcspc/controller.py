from __future__ import annotations

from chisurf.gui import QtWidgets

from qtpy import QtCore, QtGui
import pathlib

import numpy as np
import pyqtgraph as pg

import chisurf.decorators
import chisurf.gui.decorators
import chisurf.gui.widgets
from chisurf.experiments import reader
import chisurf.data
import chisurf.gui.widgets.fio
import chisurf.gui.widgets.experiments.widgets
from chisurf.plugins.jordi_g_factor import JordiGFactorCalculator
from chisurf.gui.widgets.wizard.tttr_channel_definition import load_detector_setups


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
    reader.ExperimentReaderController,
    QtWidgets.QWidget,
):
    """TCSPC TTTR controller with RICS-like TTTR histogram preview.

    This controller replaces the old .ui-based widget and provides:

    - Detector setup + detector selection from detector_setups.json
    - TTTR reading routine and routing channel selection
    - Micro-time coarsening (binning) and optional integer time shift (bins)
    - Drag-and-drop TTTR preview into a log-scaled decay plot
    - An "Add" button that forwards the current TTTR file to
      chisurf.macros.add_dataset(...), mirroring RICS/PCH controllers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label = QtWidgets.QLabel(
            "TCSPC loader (TTTR → decay histogram)\n"
            "Drop TTTR files (e.g. PTU/HT3/SPC) here. The reader will compute a\n"
            "microtime histogram using the configured coarsening and optional\n"
            "integer time shift (bins)."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        # Detector setup + logical detector selection (from channel wizard JSON)
        setup_row = QtWidgets.QHBoxLayout()
        setup_row.setContentsMargins(0, 0, 0, 0)
        setup_row.setSpacing(0)
        setup_row.addWidget(QtWidgets.QLabel("Setup:"))
        self.combo_setup = QtWidgets.QComboBox(self)
        setup_row.addWidget(self.combo_setup, 1)
        setup_row.addWidget(QtWidgets.QLabel("Detector:"))
        self.combo_detector = QtWidgets.QComboBox(self)
        setup_row.addWidget(self.combo_detector, 1)
        layout.addLayout(setup_row)

        # Parameter row: TTTR container type + routing channels + coarsening + shift
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        row.addWidget(QtWidgets.QLabel("Routine:"))
        self.combo_routine = QtWidgets.QComboBox(self)
        try:
            import tttrlib  # type: ignore[import]

            supported = []
            try:
                supported = list(tttrlib.TTTR.get_supported_container_names())
            except Exception:
                supported = []
            if not supported:
                supported = ["PTU", "HT3", "SPC"]
            self.combo_routine.addItems([str(s) for s in supported])
        except Exception:
            self.combo_routine.addItems(["PTU", "HT3", "SPC"])
        row.addWidget(self.combo_routine, 1)

        row.addWidget(QtWidgets.QLabel("Routing chs:"))
        self.lineedit_channels = QtWidgets.QLineEdit(self)
        # Allow manual override when no detector setups are used.
        self.lineedit_channels.setPlaceholderText("e.g. 0,3 or from detector setup")
        row.addWidget(self.lineedit_channels, 1)

        row.addWidget(QtWidgets.QLabel("Coarsening:"))
        self.spin_coarsen = QtWidgets.QSpinBox(self)
        self.spin_coarsen.setRange(1, 4096)
        self.spin_coarsen.setValue(1)
        row.addWidget(self.spin_coarsen)

        row.addWidget(QtWidgets.QLabel("Shift [bins]:"))
        self.spin_shift = QtWidgets.QSpinBox(self)
        self.spin_shift.setRange(-1_000_000, 1_000_000)
        self.spin_shift.setValue(0)
        row.addWidget(self.spin_shift)

        layout.addLayout(row)

        # Preview group: simple decay plot with Compute/Add/Clear buttons
        preview_group = QtWidgets.QGroupBox("Preview (drop TTTR here)")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(0)
        header.addStretch(1)

        self.toolbtn_compute = QtWidgets.QToolButton(preview_group)
        self.toolbtn_compute.setText("Compute")
        self.toolbtn_compute.setToolTip("Compute decay preview for the current file")
        header.addWidget(self.toolbtn_compute)

        self.toolbtn_add = QtWidgets.QToolButton(preview_group)
        self.toolbtn_add.setText("Add")
        self.toolbtn_add.setToolTip("Add TCSPC dataset for the current file")
        header.addWidget(self.toolbtn_add)

        self.toolbtn_clear = QtWidgets.QToolButton(preview_group)
        self.toolbtn_clear.setText("Clear")
        self.toolbtn_clear.setToolTip("Clear preview")
        header.addWidget(self.toolbtn_clear)
        preview_layout.addLayout(header)

        self.preview_plot = pg.PlotWidget(title="TCSPC decay preview")
        self.preview_plot.setLabel("bottom", "Time (ns)")
        self.preview_plot.setLabel("left", "Counts")
        try:
            self.preview_plot.setLogMode(y=True)
        except Exception:
            pass
        preview_layout.addWidget(self.preview_plot)

        layout.addWidget(preview_group)

        # Internal state for preview
        self._preview_filename: pathlib.Path | None = None
        self._preview_t = None
        self._preview_y = None

        # Cache of detector setups loaded from the central JSON
        self._detector_setups = {}

        # Accept drops anywhere in the controller
        self.setAcceptDrops(True)

        # Wire parameter changes to current_setup updates
        try:
            self.combo_routine.currentTextChanged.connect(self.onParametersChanged)
        except Exception:
            pass
        try:
            self.lineedit_channels.editingFinished.connect(self.onParametersChanged)
        except Exception:
            pass
        for w in (self.spin_coarsen, self.spin_shift):
            try:
                w.valueChanged.connect(self.onParametersChanged)
            except Exception:
                pass

        # Buttons
        try:
            self.toolbtn_clear.clicked.connect(self._on_clear_preview_clicked)
        except Exception:
            pass
        try:
            self.toolbtn_add.clicked.connect(self._on_add_clicked)
        except Exception:
            pass
        try:
            self.toolbtn_compute.clicked.connect(self._on_compute_clicked)
        except Exception:
            pass

        # Detector setup / detector combos
        try:
            self.combo_setup.currentIndexChanged.connect(self._on_setup_combo_changed)
            self.combo_detector.currentIndexChanged.connect(self._on_detector_combo_changed)
        except Exception:
            pass

        # Load available detector setups once at construction
        try:
            self._reload_detector_setups()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ExperimentReaderController API integration
    # ------------------------------------------------------------------

    def get_filename(self) -> pathlib.Path:
        """Return a TTTR filename, preferring the last previewed file.

        This mirrors RICS/PCH controllers: first ensure parameters are
        pushed into cs.current_setup, then prefer the last dropped file,
        otherwise open a dialog.
        """

        import chisurf

        try:
            self.onParametersChanged()
        except Exception:
            pass

        try:
            fn_prev = getattr(self, "_preview_filename", None)
        except Exception:
            fn_prev = None
        if fn_prev:
            return pathlib.Path(fn_prev)

        fn = chisurf.gui.widgets.open_files(
            description='TCSPC TTTR file',
            file_type='TTTR files (*.ptu *.ht3 *.spc *.phu *.photonhdf5);;All files (*.*)',
            working_path=None,
        )
        if isinstance(fn, (list, tuple)):
            return pathlib.Path(fn[0]) if fn else pathlib.Path("")
        return pathlib.Path(fn) if fn else pathlib.Path("")

    def updateUI(self):
        """Update UI elements based on cs.current_setup properties."""

        import chisurf

        try:
            setup = chisurf.cs.current_setup
        except Exception:
            return

        # reading routine
        try:
            rr = getattr(setup, 'reading_routine', None)
            if isinstance(rr, str) and rr:
                idx = self.combo_routine.findText(rr)
                if idx >= 0:
                    self.combo_routine.setCurrentIndex(idx)
        except Exception:
            pass

        # channels
        try:
            chs = getattr(setup, 'channel_numbers', None)
            if chs is None:
                chs = [getattr(setup, 'channel', 0)]
            try:
                seq = list(chs)
            except TypeError:
                seq = [chs]
            text = ", ".join(str(int(c)) for c in seq)
            self.lineedit_channels.setText(text)
        except Exception:
            pass

        # micro-time coarsening (binning factor)
        try:
            mtc = int(getattr(setup, 'micro_time_coarsening', self.spin_coarsen.value()) or 1)
            if mtc < self.spin_coarsen.minimum():
                mtc = self.spin_coarsen.minimum()
            if mtc > self.spin_coarsen.maximum():
                mtc = self.spin_coarsen.maximum()
            self.spin_coarsen.setValue(mtc)
        except Exception:
            pass

        # micro-time shift (integer bins)
        try:
            s = int(getattr(setup, 'micro_time_shift', self.spin_shift.value()) or 0)
            if s < self.spin_shift.minimum():
                s = self.spin_shift.minimum()
            if s > self.spin_shift.maximum():
                s = self.spin_shift.maximum()
            self.spin_shift.setValue(s)
        except Exception:
            pass

    def onParametersChanged(self):
        """Push TTTR parameters into cs.current_setup via CLI-style strings."""

        import chisurf

        routine = self.combo_routine.currentText()

        # Parse routing channel numbers from the line edit (comma/semicolon separated)
        try:
            ch_text = self.lineedit_channels.text().strip()
        except Exception:
            ch_text = ""
        channels: list[int] = []
        if ch_text:
            for part in ch_text.replace(";", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    channels.append(int(part))
                except Exception:
                    continue
        if not channels:
            channels = [0]
        first_channel = int(channels[0])
        channel_numbers_expr = ", ".join(str(int(c)) for c in channels)

        coarsen = int(self.spin_coarsen.value())
        shift = int(self.spin_shift.value())

        try:
            chisurf.run(
                "\n".join(
                    [
                        f"cs.current_setup.reading_routine = '{routine}'",
                        f"cs.current_setup.channel_numbers = np.array([{channel_numbers_expr}], dtype=np.int8)",
                        f"cs.current_setup.channel = {first_channel}",
                        f"cs.current_setup.micro_time_coarsening = {coarsen}",
                        f"cs.current_setup.micro_time_shift = {shift}",
                    ]
                )
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Detector setup / detector selection based on detector_setups.json
    # ------------------------------------------------------------------

    def _reload_detector_setups(self) -> None:
        """Load detector setups from the central JSON and populate combo_setup."""

        try:
            data = load_detector_setups()
            setups = data.get("setups", {}) if isinstance(data, dict) else {}
        except Exception:
            setups = {}

        self._detector_setups = setups if isinstance(setups, dict) else {}

        try:
            self.combo_setup.blockSignals(True)
        except Exception:
            pass
        self.combo_setup.clear()
        for name in sorted(self._detector_setups.keys()):
            self.combo_setup.addItem(str(name))
        try:
            self.combo_setup.blockSignals(False)
        except Exception:
            pass

        if self.combo_setup.count() > 0:
            self._on_setup_combo_changed(0)

    def _on_setup_combo_changed(self, _idx: int) -> None:
        """Populate detector combo when the setup changes and sync routine."""

        try:
            setup_name = self.combo_setup.currentText().strip()
        except Exception:
            setup_name = ""
        sd = self._detector_setups.get(setup_name) if setup_name else None
        dets = sd.get("detectors", {}) if isinstance(sd, dict) else {}

        try:
            self.combo_detector.blockSignals(True)
        except Exception:
            pass
        self.combo_detector.clear()
        for det_name in sorted(dets.keys()):
            self.combo_detector.addItem(str(det_name))
        try:
            self.combo_detector.blockSignals(False)
        except Exception:
            pass

        # Update reading routine from tttr_reading.file_type when available
        try:
            reading = sd.get("tttr_reading", {}) if isinstance(sd, dict) else {}
            routine_name = reading.get("file_type") if isinstance(reading, dict) else None
        except Exception:
            routine_name = None
        if isinstance(routine_name, str) and routine_name:
            try:
                self.combo_routine.blockSignals(True)
            except Exception:
                pass
            try:
                idx = self.combo_routine.findText(routine_name)
                if idx >= 0:
                    self.combo_routine.setCurrentIndex(idx)
            finally:
                try:
                    self.combo_routine.blockSignals(False)
                except Exception:
                    pass

        if self.combo_detector.count() > 0:
            self._on_detector_combo_changed(0)
        else:
            # Even without a detector, propagate setup/routine changes.
            try:
                self.onParametersChanged()
            except Exception:
                pass

    def _on_detector_combo_changed(self, _idx: int) -> None:
        """Update routing channel display from the chosen detector definition."""

        try:
            setup_name = self.combo_setup.currentText().strip()
        except Exception:
            setup_name = ""
        try:
            det_name = self.combo_detector.currentText().strip()
        except Exception:
            det_name = ""

        sd = self._detector_setups.get(setup_name) if setup_name else None
        dets = sd.get("detectors", {}) if isinstance(sd, dict) else {}
        info = dets.get(det_name) if det_name else None
        if not isinstance(info, dict):
            return

        chs = info.get("chs", [])

        try:
            text = ", ".join(str(int(c)) for c in chs)
        except Exception:
            text = ""
        try:
            self.lineedit_channels.setText(text)
        except Exception:
            pass

        # Propagate updated detector selection into current_setup
        try:
            self.onParametersChanged()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Drag-and-drop + preview handling
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
            if urls:
                path = pathlib.Path(str(urls[0].toLocalFile()))
                try:
                    self._preview_filename = path
                except Exception:
                    self._preview_filename = None
                try:
                    self._load_preview_from_file(path)
                except Exception:
                    pass
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def _on_clear_preview_clicked(self) -> None:
        try:
            self._preview_filename = None
        except Exception:
            pass
        try:
            self._preview_t = None
            self._preview_y = None
        except Exception:
            pass
        try:
            self.preview_plot.clear()
        except Exception:
            pass

    def _on_compute_clicked(self) -> None:
        import pathlib as _pathlib

        try:
            self.onParametersChanged()
        except Exception:
            pass

        try:
            path = getattr(self, "_preview_filename", None)
        except Exception:
            path = None
        if not path:
            try:
                path = self.get_filename()
            except Exception:
                path = None
        if not path:
            return

        try:
            p = _pathlib.Path(path)
        except Exception:
            try:
                p = _pathlib.Path(str(path))
            except Exception:
                return

        try:
            self._preview_filename = p
        except Exception:
            pass

        try:
            self._load_preview_from_file(p)
        except Exception:
            pass

    def _on_add_clicked(self) -> None:
        import chisurf
        import pathlib as _pathlib

        try:
            path = getattr(self, "_preview_filename", None)
        except Exception:
            path = None
        if not path:
            try:
                path = self.get_filename()
            except Exception:
                path = None
        if not path:
            return

        try:
            p = _pathlib.Path(path)
        except Exception:
            try:
                p = _pathlib.Path(str(path))
            except Exception:
                return

        try:
            self.onParametersChanged()
        except Exception:
            pass

        try:
            s = p.as_posix().replace("\\", "/")
        except Exception:
            return

        try:
            chisurf.run(f"chisurf.macros.add_dataset(filename=r\"{s}\")")
        except Exception:
            pass

    def _load_preview_from_file(self, path: pathlib.Path) -> None:
        import chisurf

        try:
            self.onParametersChanged()
        except Exception:
            pass

        try:
            reader_obj = chisurf.cs.current_setup
        except Exception:
            reader_obj = None
        if reader_obj is None:
            return

        try:
            group = reader_obj.read(filename=path.as_posix())
        except Exception:
            return

        try:
            from chisurf.data import ExperimentDataCurveGroup as _Group

            if isinstance(group, _Group) and len(group) > 0:
                data_obj = group[0]
            else:
                data_obj = group
        except Exception:
            data_obj = group

        try:
            t = np.asarray(getattr(data_obj, 'x', []), dtype=float)
            y = np.asarray(getattr(data_obj, 'y', []), dtype=float)
        except Exception:
            return
        if t.size == 0 or y.size == 0:
            return

        self._preview_t = t
        self._preview_y = y
        self._refresh_preview_plot()

    def _refresh_preview_plot(self) -> None:
        t = getattr(self, "_preview_t", None)
        y = getattr(self, "_preview_y", None)
        if t is None or y is None:
            return
        if np.size(t) == 0 or np.size(y) == 0:
            return
        try:
            self.preview_plot.clear()
            self.preview_plot.plot(t, y, pen='y')
            try:
                self.preview_plot.setLogMode(y=True)
            except Exception:
                pass
        except Exception:
            pass

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
