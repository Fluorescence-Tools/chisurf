from __future__ import annotations

from chisurf.gui import QtWidgets
from qtpy import QtGui
import pathlib

import numpy as np
import pyqtgraph as pg

import chisurf as cs
from chisurf.core.experiments.core import reader
from chisurf.gui.widgets.experiments.sample_selector_widget import SampleSelectorWidget
from chisurf.gui.widgets.wizard.tttr_channeldefinition import load_detector_setups


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
    - An "Add" button that forwards the current TTTR file via
      cs.core.actions.dispatch("dataset.add", ...), mirroring RICS/PCH controllers.
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

        self._pol_widget = QtWidgets.QWidget(self)
        pol_row = QtWidgets.QGridLayout(self._pol_widget)
        pol_row.setContentsMargins(0, 0, 0, 0)
        pol_row.setHorizontalSpacing(4)
        pol_row.setVerticalSpacing(0)

        self.label_pol_mode = QtWidgets.QLabel("Pol:", self._pol_widget)
        pol_row.addWidget(self.label_pol_mode, 0, 0)
        self.combo_polarization = QtWidgets.QComboBox(self._pol_widget)
        self.combo_polarization.addItems(["vm", "vv", "vh", "vv/vh"])
        pol_row.addWidget(self.combo_polarization, 0, 1)

        self.label_gfactor = QtWidgets.QLabel("G:", self._pol_widget)
        pol_row.addWidget(self.label_gfactor, 0, 2)
        self.spin_gfactor = QtWidgets.QDoubleSpinBox(self._pol_widget)
        self.spin_gfactor.setDecimals(4)
        self.spin_gfactor.setRange(0.0, 10.0)
        self.spin_gfactor.setValue(1.0)
        pol_row.addWidget(self.spin_gfactor, 0, 3)

        self.label_vh_shift = QtWidgets.QLabel("VH shift [bins]:", self._pol_widget)
        pol_row.addWidget(self.label_vh_shift, 0, 4)
        self.spin_vh_shift = QtWidgets.QSpinBox(self._pol_widget)
        self.spin_vh_shift.setRange(-1_000_000, 1_000_000)
        self.spin_vh_shift.setValue(0)
        pol_row.addWidget(self.spin_vh_shift, 0, 5)

        # Give the editors more stretch than the labels
        try:
            pol_row.setColumnStretch(1, 2)
            pol_row.setColumnStretch(3, 2)
            pol_row.setColumnStretch(5, 2)
        except Exception:
            pass

        layout.addWidget(self._pol_widget)
        try:
            self._pol_widget.setVisible(False)
        except Exception:
            pass

        # Preview group: simple decay plot with Compute/Add/Clear buttons
        preview_group = QtWidgets.QGroupBox("Preview (drop TTTR here)")
        preview_group.setMaximumHeight(250)
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
        self._preview_channels = None

        # Cache of detector setups loaded from the central JSON
        self._detector_setups = {}

        # Accept drops anywhere in the controller
        self.setAcceptDrops(True)

        # Wire parameter changes to current_setup updates and live preview
        try:
            self.combo_routine.currentTextChanged.connect(self._on_gui_parameters_changed)
        except Exception:
            pass
        try:
            self.lineedit_channels.editingFinished.connect(self._on_gui_parameters_changed)
        except Exception:
            pass
        for w in (self.spin_coarsen, self.spin_shift):
            try:
                w.valueChanged.connect(self._on_gui_parameters_changed)
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

        fn = cs.gui.widgets.open_files(
            description='TCSPC TTTR file',
            file_type='TTTR files (*.ptu *.ht3 *.spc *.phu *.photonhdf5);;All files (*.*)',
            working_path=None,
        )
        if isinstance(fn, (list, tuple)):
            return pathlib.Path(fn[0]) if fn else pathlib.Path("")
        return pathlib.Path(fn) if fn else pathlib.Path("")

    def updateUI(self):
        """Update UI elements based on cs.current_setup properties."""

        try:
            setup = cs.cs.current_setup
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

        # Polarization / G-factor / VH shift (used when multiple routing channels)
        try:
            pol = getattr(setup, 'polarization', None)
        except Exception:
            pol = None
        if isinstance(pol, str) and pol:
            try:
                idx_pol = self.combo_polarization.findText(pol)
                if idx_pol >= 0:
                    self.combo_polarization.setCurrentIndex(idx_pol)
            except Exception:
                pass

        try:
            g_val = float(getattr(setup, 'g_factor', self.spin_gfactor.value()))
            self.spin_gfactor.setValue(g_val)
        except Exception:
            pass

        try:
            vh = int(getattr(setup, 'vh_shift', self.spin_vh_shift.value()) or 0)
            if vh < self.spin_vh_shift.minimum():
                vh = self.spin_vh_shift.minimum()
            if vh > self.spin_vh_shift.maximum():
                vh = self.spin_vh_shift.maximum()
            self.spin_vh_shift.setValue(vh)
        except Exception:
            pass

        # Show polarization controls only when more than one routing channel is used
        try:
            chs = getattr(setup, 'channel_numbers', None)
            if chs is None:
                chs = [getattr(setup, 'channel', 0)]
            try:
                seq = list(chs)
            except TypeError:
                seq = [chs]
            show_pol = len(seq) > 1
        except Exception:
            show_pol = False
        try:
            self._pol_widget.setVisible(show_pol)
        except Exception:
            pass

    def onParametersChanged(self):
        """Push TTTR parameters into cs.current_setup via CLI-style strings."""

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

        # Polarization / Jordi-style parameters
        try:
            pol = str(self.combo_polarization.currentText()).strip() or 'vm'
        except Exception:
            pol = 'vm'
        try:
            gfactor = float(self.spin_gfactor.value())
        except Exception:
            gfactor = 1.0
        try:
            vh_shift = int(self.spin_vh_shift.value())
        except Exception:
            vh_shift = 0

        # Toggle visibility of polarization row based on number of channels
        try:
            show_pol = len(channels) > 1
        except Exception:
            show_pol = False
        try:
            self._pol_widget.setVisible(show_pol)
        except Exception:
            pass

        try:
            cs.run(
                "\n".join(
                    [
                        f"cs.current_setup.reading_routine = '{routine}'",
                        f"cs.current_setup.channel_numbers = np.array([{channel_numbers_expr}], dtype=np.int8)",
                        f"cs.current_setup.channel = {first_channel}",
                        f"cs.current_setup.micro_time_coarsening = {coarsen}",
                        f"cs.current_setup.micro_time_shift = {shift}",
                        f"cs.current_setup.polarization = '{pol}'",
                        f"cs.current_setup.g_factor = {gfactor:f}",
                        f"cs.current_setup.vh_shift = {vh_shift}",
                    ]
                )
            )
        except Exception:
            pass

    def _on_gui_parameters_changed(self) -> None:
        """Slot for GUI edits: push parameters and refresh preview if possible."""

        try:
            self.onParametersChanged()
        except Exception:
            pass

        try:
            path = getattr(self, "_preview_filename", None)
        except Exception:
            path = None
        if not path:
            return

        try:
            import pathlib as _pathlib

            if isinstance(path, _pathlib.Path):
                p = path
            else:
                p = _pathlib.Path(str(path))
        except Exception:
            return

        try:
            self._load_preview_from_file(p)
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

        # Propagate updated detector selection into current_setup and refresh preview
        try:
            self._on_gui_parameters_changed()
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
            self._preview_channels = None
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

        cs.core.actions.dispatch(
            name="dataset.add",
            payload={"filename": s, "experiment_reader": None},
        )

    def _apply_preview_shift(self, y: np.ndarray, shift: int) -> np.ndarray:
        arr = np.asarray(y, dtype=float)
        if arr.size == 0 or shift == 0:
            return arr
        if shift > 0:
            return np.pad(arr, (shift, 0), mode="constant")[:-shift]
        step = abs(shift)
        return np.pad(arr, (0, step), mode="constant")[step:]

    def _load_preview_from_file(self, path: pathlib.Path) -> None:
        try:
            self.onParametersChanged()
        except Exception:
            pass

        try:
            reader_obj = cs.cs.current_setup
        except Exception:
            reader_obj = None
        if reader_obj is None:
            return

        # First try a direct TTTR-based microtime histogram per routing channel
        try:
            import tttrlib  # type: ignore[import]

            try:
                routine = getattr(reader_obj, "reading_routine", None)
            except Exception:
                routine = None

            try:
                chs_value = getattr(reader_obj, "channel_numbers", None)
            except Exception:
                chs_value = None
            if chs_value is None:
                try:
                    chs_value = [getattr(reader_obj, "channel", 0)]
                except Exception:
                    chs_value = [0]
            try:
                ch_list = sorted({int(c) for c in chs_value})
            except Exception:
                try:
                    ch_list = [int(getattr(reader_obj, "channel", 0) or 0)]
                except Exception:
                    ch_list = [0]

            try:
                coarsen = int(getattr(reader_obj, "micro_time_coarsening", 1) or 1)
            except Exception:
                coarsen = 1
            if coarsen <= 0:
                coarsen = 1

            try:
                shift = int(getattr(reader_obj, "micro_time_shift", 0) or 0)
            except Exception:
                shift = 0

            if routine:
                tttr_all = tttrlib.TTTR(path.as_posix(), routine)
            else:
                tttr_all = tttrlib.TTTR(path.as_posix())

            ts: list[np.ndarray] = []
            ys: list[np.ndarray] = []

            for ch in ch_list:
                try:
                    tttr_sel = tttr_all.get_tttr_by_channel([int(ch)])
                    y_raw, x_raw = tttr_sel.get_microtime_histogram(coarsen)
                except Exception:
                    continue
                y = np.asarray(y_raw, dtype=float)
                x = np.asarray(x_raw, dtype=float)
                if y.size == 0 or x.size == 0:
                    continue
                if shift != 0:
                    y = self._apply_preview_shift(y, shift)
                x = x * 1.0e9
                n = int(min(y.size, x.size))
                if n <= 0:
                    continue
                ts.append(x[:n])
                ys.append(y[:n])

            if ts and ys:
                self._preview_t = ts
                self._preview_y = ys
                try:
                    self._preview_channels = ch_list
                except Exception:
                    pass
                self._refresh_preview_plot()
                return
        except Exception:
            pass

        # Fallback: use the reader's standard behaviour (single curve)
        try:
            group = reader_obj.read(filename=path.as_posix())
        except Exception:
            return

        try:
            from chisurf.core.data import ExperimentDataCurveGroup as _Group

            if isinstance(group, _Group) and len(group) > 0:
                data_obj = group[0]
            else:
                data_obj = group
        except Exception:
            data_obj = group

        try:
            t = np.asarray(getattr(data_obj, "x", []), dtype=float)
            y = np.asarray(getattr(data_obj, "y", []), dtype=float)
        except Exception:
            return
        if t.size == 0 or y.size == 0:
            return

        self._preview_t = t
        self._preview_y = y
        try:
            self._preview_channels = None
        except Exception:
            pass
        self._refresh_preview_plot()

    def _refresh_preview_plot(self) -> None:
        t = getattr(self, "_preview_t", None)
        y = getattr(self, "_preview_y", None)
        if t is None or y is None:
            return
        try:
            self.preview_plot.clear()
        except Exception:
            return

        try:
            chs = getattr(self, "_preview_channels", None)
        except Exception:
            chs = None

        try:
            if isinstance(t, (list, tuple)) and isinstance(y, (list, tuple)):
                if not t or not y:
                    return
                try:
                    self.preview_plot.addLegend()
                except Exception:
                    pass
                colors = ["y", "c", "m", "g", "r", "b", "w"]
                for idx, (tt, yy) in enumerate(zip(t, y)):
                    if tt is None or yy is None:
                        continue
                    if np.size(tt) == 0 or np.size(yy) == 0:
                        continue
                    pen = colors[idx % len(colors)]
                    if isinstance(chs, (list, tuple)) and idx < len(chs):
                        name = f"ch {int(chs[idx])}"
                    else:
                        name = f"ch {idx}"
                    self.preview_plot.plot(tt, yy, pen=pen, name=name)
            else:
                if np.size(t) == 0 or np.size(y) == 0:
                    return
                self.preview_plot.plot(t, y, pen="y")

            try:
                self.preview_plot.setLogMode(y=True)
            except Exception:
                pass
            # Enforce a minimum visible y-value of 0.1 on the log-scaled axis
            try:
                ys = []
                if isinstance(y, (list, tuple)):
                    for yy in y:
                        try:
                            arr = np.asarray(yy, dtype=float)
                        except Exception:
                            continue
                        if arr.size:
                            ys.append(arr)
                else:
                    arr = np.asarray(y, dtype=float)
                    if arr.size:
                        ys.append(arr)
                if ys:
                    y_all = np.concatenate(ys)
                    y_pos = y_all[y_all > 0]
                    if y_pos.size:
                        y_max = float(y_pos.max())
                        y_min = 0.1
                        if y_max <= y_min:
                            y_max = y_min * 10.0
                        self.preview_plot.setYRange(y_min, y_max)
            except Exception:
                pass
        except Exception:
            pass
