from __future__ import annotations

import pathlib

import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import chisurf.gui.widgets
from chisurf.core.experiments.core import reader
from chisurf.gui.widgets.wizard.tttr_channeldefinition import load_detector_setups


class RICSController(reader.ExperimentReaderController, QtWidgets.QWidget):

    def get_filename(self) -> pathlib.Path:
        # Ensure current GUI values are pushed into cs.current_setup before
        # opening the file dialog (mirrors TCSPC TTTR controller behavior).
        try:
            self.onParametersChanged()
        except Exception:
            pass

        # Two-step workflow: if a file was previously dropped and inspected
        # in the preview widget, reuse that path instead of opening a dialog.
        if getattr(self, "_preview_filename", None) is not None:
            return pathlib.Path(self._preview_filename)

        fn = chisurf.gui.widgets.open_files(
            description='RICS TTTR/TIFF file',
            file_type='All files (*.*)',
            working_path=None
        )
        if isinstance(fn, (list, tuple)):
            return pathlib.Path(fn[0]) if fn else pathlib.Path("")
        return pathlib.Path(fn) if fn else pathlib.Path("")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label = QtWidgets.QLabel(
            "RICS loader (TTTR/TIFF → CLSM → ICS)\n"
            "Drop TTTR files (e.g. PTU/HT3) or TIFF stacks here. The reader will\n"
            "compute an ICS/RICS map via tttrlib.CLSMImage.compute_ics."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        # File parameter controls share a grid layout for clean alignment
        file_params_layout = QtWidgets.QGridLayout()
        file_params_layout.setContentsMargins(0, 0, 0, 0)
        file_params_layout.setSpacing(4)

        file_params_layout.addWidget(QtWidgets.QLabel("Setup:"), 0, 0)
        self.combo_setup = QtWidgets.QComboBox(self)
        file_params_layout.addWidget(self.combo_setup, 0, 1)

        file_params_layout.addWidget(QtWidgets.QLabel("Detector:"), 0, 2)
        self.combo_detector = QtWidgets.QComboBox(self)
        file_params_layout.addWidget(self.combo_detector, 0, 3)

        file_params_layout.addWidget(QtWidgets.QLabel("Routine:"), 1, 0)
        self.combo_routine = QtWidgets.QComboBox(self)
        try:
            # Populate with tttrlib-supported container names if available
            import tttrlib  # local import to avoid hard dependency at import time
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
        file_params_layout.addWidget(self.combo_routine, 1, 1)

        file_params_layout.addWidget(QtWidgets.QLabel("Routing chs:"), 1, 2)
        self.lineedit_channels = QtWidgets.QLineEdit(self)
        self.lineedit_channels.setReadOnly(True)
        self.lineedit_channels.setPlaceholderText("from detector setup")
        file_params_layout.addWidget(self.lineedit_channels, 1, 3)

        file_params_layout.setColumnStretch(1, 1)
        file_params_layout.setColumnStretch(3, 1)
        layout.addLayout(file_params_layout)

        opts_layout = QtWidgets.QGridLayout()
        opts_layout.setContentsMargins(0, 0, 0, 0)
        opts_layout.setSpacing(0)

        opts_layout.addWidget(QtWidgets.QLabel("X range:"), 0, 0)
        self.spin_x0 = QtWidgets.QSpinBox(self)
        self.spin_x0.setRange(0, 4096)
        self.spin_x0.setValue(0)
        self.spin_x1 = QtWidgets.QSpinBox(self)
        self.spin_x1.setRange(-1, 4096)
        self.spin_x1.setValue(-1)
        opts_layout.addWidget(self.spin_x0, 0, 1)
        opts_layout.addWidget(self.spin_x1, 0, 2)

        opts_layout.addWidget(QtWidgets.QLabel("Y range:"), 1, 0)
        self.spin_y0 = QtWidgets.QSpinBox(self)
        self.spin_y0.setRange(0, 4096)
        self.spin_y0.setValue(0)
        self.spin_y1 = QtWidgets.QSpinBox(self)
        self.spin_y1.setRange(-1, 4096)
        self.spin_y1.setValue(-1)
        opts_layout.addWidget(self.spin_y0, 1, 1)
        opts_layout.addWidget(self.spin_y1, 1, 2)

        opts_layout.addWidget(QtWidgets.QLabel("Subtract avg:"), 2, 0)
        self.combo_subtract = QtWidgets.QComboBox(self)
        self.combo_subtract.addItem("None", "")
        self.combo_subtract.addItem("Frame", "frame")
        self.combo_subtract.addItem("Stack", "stack")
        opts_layout.addWidget(self.combo_subtract, 2, 1, 1, 2)

        opts_layout.addWidget(QtWidgets.QLabel("Frame shift:"), 3, 0)
        self.spin_frame_shift = QtWidgets.QSpinBox(self)
        self.spin_frame_shift.setRange(-9999, 9999)
        self.spin_frame_shift.setValue(0)
        opts_layout.addWidget(self.spin_frame_shift, 3, 1)

        self.check_fftshift = QtWidgets.QCheckBox("Center zero lag", self)
        self.check_fftshift.setChecked(True)
        opts_layout.addWidget(self.check_fftshift, 3, 2)

        # Option to view RICS as a framewise stack instead of only the
        # mean ICS map. This does not change the fitting data (which still
        # uses the mean), but exposes the per-frame ICS stack in the
        # preview and metadata.
        self.check_framewise_rics = QtWidgets.QCheckBox("Framewise RICS", self)
        self.check_framewise_rics.setChecked(False)
        opts_layout.addWidget(self.check_framewise_rics, 4, 0, 1, 3)

        # Optional timing parameters for the CLSM scan. When left at 0.0 the
        # reader will attempt to infer these from the TTTR header (e.g. PTU
        # $TimePerPixel). Values are expressed in µs (pixel) and ms (line)
        # matching the analytic RICS models.
        opts_layout.addWidget(QtWidgets.QLabel("Pixel dur [µs]"), 5, 0)
        self.spin_pixel_dur = QtWidgets.QDoubleSpinBox(self)
        self.spin_pixel_dur.setDecimals(3)
        self.spin_pixel_dur.setRange(0.0, 1.0e9)
        self.spin_pixel_dur.setValue(0.0)
        opts_layout.addWidget(self.spin_pixel_dur, 5, 1, 1, 2)

        opts_layout.addWidget(QtWidgets.QLabel("Line dur [ms]"), 6, 0)
        self.spin_line_dur = QtWidgets.QDoubleSpinBox(self)
        self.spin_line_dur.setDecimals(3)
        self.spin_line_dur.setRange(0.0, 1.0e9)
        self.spin_line_dur.setValue(0.0)
        opts_layout.addWidget(self.spin_line_dur, 6, 1, 1, 2)

        layout.addLayout(opts_layout)

        # ------------------------------------------------------------------
        # Inline image preview (intensity vs RICS) prior to dataset creation
        # ------------------------------------------------------------------

        preview_group = QtWidgets.QGroupBox("Preview (drop TTTR/TIFF here)")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(0)
        self.radio_preview_intensity = QtWidgets.QRadioButton("Intensity", preview_group)
        self.radio_preview_rics = QtWidgets.QRadioButton("RICS", preview_group)
        try:
            self.radio_preview_intensity.setChecked(True)
        except Exception:
            pass
        mode_row.addWidget(self.radio_preview_intensity)
        mode_row.addWidget(self.radio_preview_rics)

        # Spacer between mode selector and action buttons so that the
        # Add/Clear buttons are right-aligned.
        mode_row.addStretch(1)

        # Toolbuttons for adding a RICS dataset from the current preview
        # file and for clearing the preview/ROI entirely.
        self.toolbtn_add_rics = QtWidgets.QToolButton(preview_group)
        self.toolbtn_add_rics.setText("Add")
        self.toolbtn_add_rics.setToolTip("Add RICS dataset for the current file")
        mode_row.addWidget(self.toolbtn_add_rics)

        self.toolbtn_clear_preview = QtWidgets.QToolButton(preview_group)
        self.toolbtn_clear_preview.setText("Clear")
        self.toolbtn_clear_preview.setToolTip("Clear preview and ROI")
        mode_row.addWidget(self.toolbtn_clear_preview)
        preview_layout.addLayout(mode_row)

        # Simple pyqtgraph ImageView for stack/2D visualization
        self.preview_view = pg.ImageView()
        try:
            # Keep the histogram widget reasonably small
            self.preview_view.ui.histogram.setMaximumWidth(120)
        except Exception:
            pass
        try:
            # Disable interactive zoom/pan for the preview to keep the
            # coordinate system fixed and ROI mapping predictable.
            view = self.preview_view.getView()
            view.setMouseEnabled(x=False, y=False)
            view.setMenuEnabled(False)
        except Exception:
            pass
        preview_layout.addWidget(self.preview_view, 1)

        # Optional rectangular ROI overlay used to define the spatial region
        # for RICS/ICS computation. This ROI is synchronized with the
        # numeric X/Y range spin boxes and therefore automatically persisted
        # through the existing reader/setup state + data.meta_data.
        self._preview_roi = None
        self._preview_img_shape = None  # (ny, nx) of currently shown image
        self._roi_sync_in_progress = False

        layout.addWidget(preview_group, 1)

        # Internal state for previewing dropped files
        self._preview_filename: pathlib.Path | None = None
        self._preview_rics_dirty: bool = False
        self._preview_rics_stack = None

        # Cache of detector setups loaded from the central JSON
        self._detector_setups = {}
        self._micro_time_ranges = None

        # Accept drops anywhere in the controller for quick inspection
        self.setAcceptDrops(True)

        # Connect edits to parameter propagation into cs.current_setup
        try:
            self.combo_routine.currentTextChanged.connect(self.onParametersChanged)
        except Exception:
            pass
        for w in (self.spin_x0, self.spin_x1, self.spin_y0, self.spin_y1, self.spin_frame_shift, self.spin_pixel_dur, self.spin_line_dur):
            try:
                w.valueChanged.connect(self.onParametersChanged)
            except Exception:
                pass
        try:
            self.combo_subtract.currentIndexChanged.connect(self.onParametersChanged)
        except Exception:
            pass
        try:
            self.check_fftshift.toggled.connect(self.onParametersChanged)
        except Exception:
            pass
        try:
            self.check_framewise_rics.toggled.connect(self.onParametersChanged)
        except Exception:
            pass

        # Preview mode toggles
        try:
            self.radio_preview_intensity.toggled.connect(self._on_preview_mode_changed)
            self.radio_preview_rics.toggled.connect(self._on_preview_mode_changed)
        except Exception:
            pass

        # Add / Clear preview toolbuttons
        try:
            self.toolbtn_clear_preview.clicked.connect(self._on_clear_preview_clicked)
        except Exception:
            pass
        try:
            self.toolbtn_add_rics.clicked.connect(self._on_add_rics_clicked)
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

        # After the controller has been constructed and the main window had a
        # chance to call updateUI, re-synchronize routine + routing channels
        # from the currently selected setup so the initial state matches the
        # detector wizard configuration without requiring the user to change
        # the setup once.
        try:
            QtCore.QTimer.singleShot(
                0,
                lambda: self._on_setup_combo_changed(self.combo_setup.currentIndex())
            )
        except Exception:
            pass

    @property
    def filename(self) -> str:
        # ExperimentReaderController requires a filename attribute; we
        # delegate to get_filename() for consistency with other controllers.
        fn = self.get_filename()
        return str(fn) if fn is not None else ""

    def updateUI(self):
        """Update UI elements based on current_setup properties."""
        import chisurf
        try:
            setup = chisurf.cs.current_setup
        except Exception:
            return

        # Sync reading routine and channel index from the current setup if
        # present. This keeps widget and CLI state in sync.
        try:
            rr = getattr(setup, 'reading_routine', None)
            if isinstance(rr, str) and rr:
                idx = self.combo_routine.findText(rr)
                if idx >= 0:
                    self.combo_routine.setCurrentIndex(idx)
        except Exception:
            pass

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

        try:
            x_range = getattr(setup, 'x_range', None)
            if isinstance(x_range, (list, tuple)) and len(x_range) >= 2:
                self.spin_x0.setValue(int(x_range[0]))
                self.spin_x1.setValue(int(x_range[1]))
        except Exception:
            pass

        try:
            y_range = getattr(setup, 'y_range', None)
            if isinstance(y_range, (list, tuple)) and len(y_range) >= 2:
                self.spin_y0.setValue(int(y_range[0]))
                self.spin_y1.setValue(int(y_range[1]))
        except Exception:
            pass

        try:
            mtr = getattr(setup, 'micro_time_ranges', None)
            if isinstance(mtr, (list, tuple)):
                self._micro_time_ranges = mtr
        except Exception:
            pass

        try:
            sa = getattr(setup, 'subtract_average', None) or ""
            idx = self.combo_subtract.findData(sa)
            if idx >= 0:
                self.combo_subtract.setCurrentIndex(idx)
        except Exception:
            pass

        try:
            fs = int(getattr(setup, 'frame_shift', 0) or 0)
            if self.spin_frame_shift.minimum() <= fs <= self.spin_frame_shift.maximum():
                self.spin_frame_shift.setValue(fs)
        except Exception:
            pass

        try:
            ff = getattr(setup, 'fftshift', True)
            self.check_fftshift.setChecked(bool(ff))
        except Exception:
            pass

        try:
            fw = getattr(setup, 'framewise_rics', False)
            self.check_framewise_rics.setChecked(bool(fw))
        except Exception:
            pass

        # Pixel and line durations (optional, 0.0 → auto from header)
        try:
            pd = getattr(setup, 'pixel_duration', None)
            if isinstance(pd, (int, float)) and pd >= 0.0:
                self.spin_pixel_dur.setValue(float(pd))
        except Exception:
            pass
        try:
            ld = getattr(setup, 'line_duration', None)
            if isinstance(ld, (int, float)) and ld >= 0.0:
                self.spin_line_dur.setValue(float(ld))
        except Exception:
            pass

    def onParametersChanged(self):
        """Push current RICS parameters into cs.current_setup via CLI.

        This mirrors TCSPC controllers which drive the underlying reader
        through CLI-style assignments, making the state visible in the
        IPython/console layer.
        """
        import chisurf
        routine = self.combo_routine.currentText()

        # Parse routing channel numbers from the line edit (comma/semicolon separated)
        try:
            ch_text = self.lineedit_channels.text().strip()
        except Exception:
            ch_text = ""
        channels = []
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
        channel_numbers_expr = ", ".join(str(int(c)) for c in channels)
        first_channel = int(channels[0])

        x0 = int(self.spin_x0.value())
        x1 = int(self.spin_x1.value())
        y0 = int(self.spin_y0.value())
        y1 = int(self.spin_y1.value())
        subtract_token = self.combo_subtract.currentData()
        if subtract_token is None:
            subtract_token = ""
        frame_shift = int(self.spin_frame_shift.value())
        fftshift_flag = bool(self.check_fftshift.isChecked())
        fftshift_str = "True" if fftshift_flag else "False"

        try:
            framewise_flag = bool(self.check_framewise_rics.isChecked())
        except Exception:
            framewise_flag = False
        framewise_str = "True" if framewise_flag else "False"

        # Pixel and line durations: interpret <= 0 as "auto from header".
        try:
            pixel_dur_val = float(self.spin_pixel_dur.value())
        except Exception:
            pixel_dur_val = 0.0
        try:
            line_dur_val = float(self.spin_line_dur.value())
        except Exception:
            line_dur_val = 0.0
        pixel_dur_expr = repr(float(pixel_dur_val)) if pixel_dur_val > 0.0 else "None"
        line_dur_expr = repr(float(line_dur_val)) if line_dur_val > 0.0 else "None"

        mtr_value = getattr(self, "_micro_time_ranges", None)
        mtr_list = []
        if isinstance(mtr_value, (list, tuple)):
            for r in mtr_value:
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    try:
                        a = int(r[0])
                        b = int(r[1])
                    except Exception:
                        continue
                    mtr_list.append([a, b])
        micro_time_ranges_expr = repr(mtr_list)

        # Optional: expose detector setup / detector names on current_setup
        try:
            setup_name = self.combo_setup.currentText().strip()
        except Exception:
            setup_name = ""
        try:
            detector_name = self.combo_detector.currentText().strip()
        except Exception:
            detector_name = ""
        setup_name_esc = setup_name.replace("'", "\\'")
        detector_name_esc = detector_name.replace("'", "\\'")
        try:
            chisurf.run(
                "\n".join(
                    [
                        f"cs.current_setup.reading_routine = '{routine}'",
                        f"cs.current_setup.channel_numbers = np.array([{channel_numbers_expr}], dtype=np.int8)",
                        f"cs.current_setup.channel = {first_channel}",
                        f"cs.current_setup.detector_setup = '{setup_name_esc}'",
                        f"cs.current_setup.detector_name = '{detector_name_esc}'",
                        f"cs.current_setup.x_range = ({x0}, {x1})",
                        f"cs.current_setup.y_range = ({y0}, {y1})",
                        f"cs.current_setup.micro_time_ranges = {micro_time_ranges_expr}",
                        f"cs.current_setup.subtract_average = '{subtract_token}'",
                        f"cs.current_setup.frame_shift = {frame_shift}",
                        f"cs.current_setup.fftshift = {fftshift_str}",
                        f"cs.current_setup.framewise_rics = {framewise_str}",
                        f"cs.current_setup.pixel_duration = {pixel_dur_expr}",
                        f"cs.current_setup.line_duration = {line_dur_expr}",
                        # Invalidate cached ICS so that subsequent reads
                        # honor updated ROI/micro-time ranges.
                        "cs.current_setup._cache_ics_stack = None",
                        "cs.current_setup._cache_filename = None",
                    ]
                )
            )
        except Exception:
            pass

        # Keep the interactive ROI overlay aligned with the numeric ranges
        # whenever parameters change, unless we are already in the middle of
        # a ROI-driven update to the spin boxes.
        try:
            if not getattr(self, "_roi_sync_in_progress", False):
                self._sync_preview_roi_to_ranges()
        except Exception:
            pass

        # Mark preview RICS as dirty so that switching back to RICS mode
        # recomputes the map for the updated parameters.
        try:
            if getattr(self, "_preview_filename", None) is not None:
                self._preview_rics_dirty = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Drag-and-drop support and preview handling
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
                self._preview_filename = path
                try:
                    self._load_preview_from_file(path)
                except Exception:
                    pass
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def _on_preview_mode_changed(self, _checked: bool) -> None:
        # Only update when one of the mode buttons is actually checked.
        try:
            show_intensity = bool(self.radio_preview_intensity.isChecked())
            show_rics = bool(self.radio_preview_rics.isChecked())
            if not (show_intensity or show_rics):
                return
        except Exception:
            self._refresh_preview_image()
            return

        # When switching into RICS mode and the ROI/parameters changed
        # since the last preview computation, re-run the reader so that
        # the ICS map reflects the latest ROI.
        if show_rics and getattr(self, "_preview_filename", None) is not None:
            if getattr(self, "_preview_rics_dirty", False):
                try:
                    self._load_preview_from_file(self._preview_filename)  # type: ignore[arg-type]
                    return
                except Exception:
                    # Fall back to simply refreshing the image.
                    pass

        self._refresh_preview_image()

    def _on_clear_preview_clicked(self) -> None:
        """Clear the current preview image and dropped filename."""

        # Forget previously dropped file so get_filename() opens a dialog again
        try:
            self._preview_filename = None
        except Exception:
            pass

        # Drop cached preview arrays
        for attr in ("_preview_intensity_stack", "_preview_intensity_mean", "_preview_rics_mean", "_preview_rics_stack"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass

        # Reset preview geometry / ROI-related cached state
        try:
            self._preview_img_shape = None
        except Exception:
            pass
        try:
            self._preview_rics_dirty = False
        except Exception:
            pass

        # Remove ROI overlay if present and drop the handle so it will be
        # recreated lazily on the next non-empty preview.
        try:
            roi = getattr(self, "_preview_roi", None)
        except Exception:
            roi = None
        if roi is not None:
            try:
                view = self.preview_view.getView()
                view.removeItem(roi)
            except Exception:
                pass
        try:
            self._preview_roi = None
        except Exception:
            pass
        try:
            self._roi_sync_in_progress = False
        except Exception:
            pass

        # Clear the ImageView content
        try:
            self.preview_view.clear()
        except Exception:
            try:
                # Fallback: show an empty image
                import numpy as np
                self.preview_view.setImage(np.zeros((1, 1), dtype=float))
            except Exception:
                pass

        # Reset numeric ROI ranges to their default full-image semantics
        # (0, -1) so that the next load uses the entire field of view.
        try:
            for sb, v in (
                (self.spin_x0, 0),
                (self.spin_x1, -1),
                (self.spin_y0, 0),
                (self.spin_y1, -1),
            ):
                try:
                    sb.blockSignals(True)
                    sb.setValue(int(v))
                finally:
                    sb.blockSignals(False)
        except Exception:
            pass

        # Propagate cleared parameters into cs.current_setup so that any
        # subsequent reads see the reset ROI/range configuration.
        try:
            self.onParametersChanged()
        except Exception:
            pass

    def _on_add_rics_clicked(self) -> None:
        """Add a RICS dataset for the currently previewed file.

        This mirrors the main window's "Add data" action but uses the
        preview filename when available so that the dataset reflects the
        ROI/micro-time/parameter configuration shown in the preview.
        """

        import chisurf
        import pathlib as _pathlib

        # Prefer the last dropped/previewed file. If none is available,
        # fall back to the standard file dialog via get_filename().
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

        # Ensure current GUI parameters (ROI, micro time ranges, etc.) are
        # pushed into cs.current_setup before adding the dataset.
        try:
            self.onParametersChanged()
        except Exception:
            pass

        # Normalize path like the main window's onAddDataset implementation
        s = p.as_posix().replace("\\", "/")
        chisurf.core.actions.dispatch(
            name="dataset.add",
            payload={"filename": s, "experiment_reader": None},
        )

    def _load_preview_from_file(self, path: pathlib.Path) -> None:
        """Load RICS/intensity preview for a dropped TTTR/TIFF file."""

        import chisurf
        import numpy as np

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
            from chisurf.core.data import ExperimentDataCurveGroup
            if isinstance(group, ExperimentDataCurveGroup) and len(group) > 0:
                data_obj = group[0]
            else:
                data_obj = group
        except Exception:
            data_obj = group

        meta = getattr(data_obj, 'meta_data', {}) or {}
        rics_meta = meta.get('rics', {}) or {}

        self._preview_intensity_stack = rics_meta.get('intensity_stack', None)
        self._preview_intensity_mean = rics_meta.get('intensity_mean', None)
        self._preview_rics_mean = rics_meta.get('ics_mean', None)
        self._preview_rics_stack = rics_meta.get('ics_stack', None)

        # Update timing controls from metadata if available so that the UI
        # reflects values derived from the TTTR header for PTU/HT3 files.
        try:
            pd = rics_meta.get('pixel_duration_us', None)
        except Exception:
            pd = None
        try:
            ld = rics_meta.get('line_duration_ms', None)
        except Exception:
            ld = None
        try:
            if isinstance(pd, (int, float)) and pd > 0.0:
                self.spin_pixel_dur.blockSignals(True)
                self.spin_pixel_dur.setValue(float(pd))
                self.spin_pixel_dur.blockSignals(False)
        except Exception:
            pass
        try:
            if isinstance(ld, (int, float)) and ld > 0.0:
                self.spin_line_dur.blockSignals(True)
                self.spin_line_dur.setValue(float(ld))
                self.spin_line_dur.blockSignals(False)
        except Exception:
            pass

        # If no ICS was computed, fall back to a simple sum-over-frames
        # intensity image as a last resort.
        if self._preview_rics_mean is None and self._preview_intensity_stack is not None:
            try:
                arr = np.asarray(self._preview_intensity_stack, dtype=float)
                if arr.ndim == 3:
                    self._preview_rics_mean = arr.mean(axis=0)
            except Exception:
                pass

        # Preview now reflects the latest parameters / ROI
        try:
            self._preview_rics_dirty = False
        except Exception:
            pass

        self._refresh_preview_image()

    def _refresh_preview_image(self) -> None:
        """Update the preview widget according to current mode and data."""

        import numpy as np

        try:
            show_intensity = bool(self.radio_preview_intensity.isChecked())
        except Exception:
            show_intensity = True
        try:
            framewise = bool(self.check_framewise_rics.isChecked())
        except Exception:
            framewise = False

        arr = None
        if show_intensity:
            stack = getattr(self, '_preview_intensity_stack', None)
            mean_img = getattr(self, '_preview_intensity_mean', None)
            if stack is not None:
                try:
                    s = np.asarray(stack, dtype=float)
                    if s.ndim == 3 and s.size > 0:
                        arr = s
                except Exception:
                    arr = None
            if arr is None and mean_img is not None:
                try:
                    m = np.asarray(mean_img, dtype=float)
                    if m.ndim == 2:
                        arr = m
                except Exception:
                    arr = None
        else:
            stack_rics = getattr(self, '_preview_rics_stack', None)
            if framewise and stack_rics is not None:
                try:
                    s = np.asarray(stack_rics, dtype=float)
                    if s.ndim == 3 and s.size > 0:
                        arr = s
                except Exception:
                    arr = None
            if arr is None:
                rics = getattr(self, '_preview_rics_mean', None)
                if rics is not None:
                    try:
                        m = np.asarray(rics, dtype=float)
                        if m.ndim == 2:
                            arr = m
                    except Exception:
                        arr = None

        if arr is None:
            return

        # Determine current 2D shape and remember it for ROI mapping.
        try:
            if arr.ndim == 3 and arr.size > 0:
                ny, nx = int(arr.shape[1]), int(arr.shape[2])
            elif arr.ndim == 2:
                ny, nx = int(arr.shape[0]), int(arr.shape[1])
            else:
                ny = nx = 0
        except Exception:
            ny = nx = 0
        if ny > 0 and nx > 0:
            self._preview_img_shape = (ny, nx)
            # Update numeric X/Y range limits to reflect the current image
            # dimensions so that ROI ranges cannot exceed the field of view.
            try:
                self._update_range_spin_limits_from_image(nx, ny)
            except Exception:
                pass

        try:
            if arr.ndim == 3:
                # Stack along frames: use time axis for browsing
                self.preview_view.setImage(arr, axes={"t": 0, "y": 1, "x": 2})
            elif arr.ndim == 2:
                self.preview_view.setImage(arr)
        except Exception:
            pass

        # Align ROI with the current numeric X/Y ranges and toggle visibility
        # depending on preview mode: visible for intensity, hidden for RICS.
        try:
            self._sync_preview_roi_to_ranges()
            roi = getattr(self, "_preview_roi", None)
            if roi is not None:
                show_intensity = bool(self.radio_preview_intensity.isChecked())
                roi.setVisible(show_intensity)
        except Exception:
            pass

    def _update_range_spin_limits_from_image(self, nx: int, ny: int) -> None:
        """Adjust X/Y range spinbox limits based on image dimensions.

        This keeps ROI indices within the valid [0, nx-1] / [0, ny-1] range
        (0-based indexing) while still allowing the special value -1 on the
        upper bounds to indicate "full image" for x1/y1.
        """

        max_x = max(0, nx - 1)
        max_y = max(0, ny - 1)

        # Update allowed ranges
        try:
            for sb, lo, hi in (
                (self.spin_x0, 0, max_x),
                (self.spin_x1, -1, max_x),
                (self.spin_y0, 0, max_y),
                (self.spin_y1, -1, max_y),
            ):
                try:
                    sb.blockSignals(True)
                    sb.setRange(int(lo), int(hi))
                finally:
                    sb.blockSignals(False)
        except Exception:
            pass

        # Clamp existing values into the new ranges; keep -1 as "full".
        try:
            if self.spin_x0.value() < 0:
                self.spin_x0.setValue(0)
            if self.spin_y0.value() < 0:
                self.spin_y0.setValue(0)

            x1 = self.spin_x1.value()
            if x1 < 0 or x1 > max_x:
                # Default to full width (last pixel index) on first load
                self.spin_x1.setValue(max_x)

            y1 = self.spin_y1.value()
            if y1 < 0 or y1 > max_y:
                # Default to full height (last pixel index)
                self.spin_y1.setValue(max_y)
        except Exception:
            pass

    def _ensure_preview_roi(self):
        """Create the rectangular ROI on first use and connect callbacks."""

        if self._preview_roi is not None:
            return self._preview_roi
        try:
            roi = pg.RectROI(
                [0, 0],
                [10, 10],
                pen={"color": 'y', "width": 1},
                rotatable=False,
            )
        except Exception:
            return None

        try:
            view = self.preview_view.getView()
            view.addItem(roi)
            roi.setZValue(10)
        except Exception:
            pass

        self._preview_roi = roi
        try:
            roi.sigRegionChanged.connect(self._on_preview_roi_changed)
        except Exception:
            pass
        return self._preview_roi

    def _sync_preview_roi_to_ranges(self) -> None:
        """Update the ROI geometry from the current X/Y range spin boxes."""

        if not self._preview_img_shape:
            return
        ny, nx = self._preview_img_shape

        try:
            x0 = int(self.spin_x0.value())
            x1 = int(self.spin_x1.value())
            y0 = int(self.spin_y0.value())
            y1 = int(self.spin_y1.value())
        except Exception:
            return

        # Normalize ranges: -1 or invalid values mean full size.
        if x1 < 0 or x1 > nx:
            x1 = nx
        if y1 < 0 or y1 > ny:
            y1 = ny
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 <= x0 or nx <= 0:
            x0, x1 = 0, nx
        if y1 <= y0 or ny <= 0:
            y0, y1 = 0, ny

        roi = self._ensure_preview_roi()
        if roi is None:
            return

        try:
            self._roi_sync_in_progress = True
            roi.setPos((float(x0), float(y0)))
            roi.setSize((float(x1 - x0), float(y1 - y0)))
        finally:
            self._roi_sync_in_progress = False

    def _on_preview_roi_changed(self) -> None:
        """When the user moves/resizes the ROI, update numeric ranges.

        This makes the ROI the primary way of defining the CLSM/ICS ROI while
        still persisting ranges via the existing x_range/y_range mechanism.
        """

        if self._roi_sync_in_progress:
            return
        if not self._preview_img_shape or self._preview_roi is None:
            return
        ny, nx = self._preview_img_shape

        try:
            pos = self._preview_roi.pos()
            size = self._preview_roi.size()
        except Exception:
            return

        try:
            x0 = int(round(float(pos.x())))
            y0 = int(round(float(pos.y())))
            w = int(round(float(size.x())))
            h = int(round(float(size.y())))
        except Exception:
            return

        x0 = max(0, min(x0, nx))
        y0 = max(0, min(y0, ny))
        x1 = max(x0 + 1, min(x0 + w, nx))
        y1 = max(y0 + 1, min(y0 + h, ny))

        # Push ROI limits into the spin boxes (blocking their signals to
        # avoid redundant cascading updates), then propagate via
        # onParametersChanged which ultimately updates cs.current_setup and
        # drives RICSReader.x_range/y_range.
        try:
            self._roi_sync_in_progress = True
            for sb, v in (
                (self.spin_x0, x0),
                (self.spin_x1, x1),
                (self.spin_y0, y0),
                (self.spin_y1, y1),
            ):
                try:
                    sb.blockSignals(True)
                    sb.setValue(int(v))
                finally:
                    sb.blockSignals(False)
        finally:
            self._roi_sync_in_progress = False

        try:
            self.onParametersChanged()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Detector setup / detector selection based on detector_setups.json
    # ------------------------------------------------------------------

    def _reload_detector_setups(self) -> None:
        """Load detector setups from the central JSON and populate combo_setup.

        This mirrors the behaviour of the FCS channel preset dialog but keeps
        the UI minimal: only setup name and detector name are exposed.
        """

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
        """Populate detector combo when the setup changes.

        In addition to updating the detector list, try to synchronize the
        TTTR reading routine combo with the setup's stored tttr_reading
        configuration (file_type), falling back gracefully when no such
        information is available.
        """

        try:
            setup_name = self.combo_setup.currentText().strip()
        except Exception:
            setup_name = ""
        sd = self._detector_setups.get(setup_name) if setup_name else None
        dets = sd.get("detectors", {}) if isinstance(sd, dict) else {}

        # Update detector combo
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
        self._micro_time_ranges = info.get("micro_time_ranges", None)

        # Update the read-only line edit with the full routing channel list
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
