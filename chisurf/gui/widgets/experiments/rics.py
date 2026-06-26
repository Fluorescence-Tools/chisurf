from __future__ import annotations

import pathlib

import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import chisurf as cs
import chisurf.gui.widgets
import chisurf.core.actions
from chisurf.core.experiments.core import reader
from chisurf.gui.widgets.wizard.tttr_channeldefinition import load_detector_setups


class _RicsDetectorWidget(QtWidgets.QWidget):
    """Detector cascade: Setup → Detector → Routine + read-only routing channels."""

    changed = QtCore.Signal()

    def __init__(self, model, target=None, parent=None, **kwargs):
        super().__init__(parent)
        self._model = model
        self._detector_setups: dict = {}
        self._micro_time_ranges = None

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setSpacing(4)

        grid.addWidget(QtWidgets.QLabel("Setup:"), 0, 0)
        self.combo_setup = QtWidgets.QComboBox()
        grid.addWidget(self.combo_setup, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Detector:"), 0, 2)
        self.combo_detector = QtWidgets.QComboBox()
        grid.addWidget(self.combo_detector, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Routine:"), 1, 0)
        self.combo_routine = QtWidgets.QComboBox()
        try:
            import tttrlib
            supported = list(tttrlib.TTTR.get_supported_container_names())
            if not supported:
                supported = ["PTU", "HT3", "SPC"]
            self.combo_routine.addItems([str(s) for s in supported])
        except Exception:
            self.combo_routine.addItems(["PTU", "HT3", "SPC"])
        grid.addWidget(self.combo_routine, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Routing chs:"), 1, 2)
        self.lineedit_channels = QtWidgets.QLineEdit()
        self.lineedit_channels.setReadOnly(True)
        self.lineedit_channels.setPlaceholderText("from detector setup")
        grid.addWidget(self.lineedit_channels, 1, 3)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.combo_setup.currentIndexChanged.connect(self._on_setup_changed)
        self.combo_detector.currentIndexChanged.connect(self._on_detector_changed)
        self.combo_routine.currentTextChanged.connect(self._on_routine_changed)

        self._reload_setups()

    def _reload_setups(self) -> None:
        try:
            data = load_detector_setups()
            setups = data.get("setups", {}) if isinstance(data, dict) else {}
        except Exception:
            setups = {}
        self._detector_setups = setups if isinstance(setups, dict) else {}

        self.combo_setup.blockSignals(True)
        self.combo_setup.clear()
        for name in sorted(self._detector_setups.keys()):
            self.combo_setup.addItem(str(name))
        self.combo_setup.blockSignals(False)

        if self.combo_setup.count() > 0:
            self._on_setup_changed(0)

    def _on_setup_changed(self, _idx: int) -> None:
        setup_name = self.combo_setup.currentText().strip()
        sd = self._detector_setups.get(setup_name) if setup_name else None
        dets = sd.get("detectors", {}) if isinstance(sd, dict) else {}

        self.combo_detector.blockSignals(True)
        self.combo_detector.clear()
        for det_name in sorted(dets.keys()):
            self.combo_detector.addItem(str(det_name))
        self.combo_detector.blockSignals(False)

        try:
            reading = sd.get("tttr_reading", {}) if isinstance(sd, dict) else {}
            routine_name = reading.get("file_type") if isinstance(reading, dict) else None
        except Exception:
            routine_name = None
        if isinstance(routine_name, str) and routine_name:
            self.combo_routine.blockSignals(True)
            idx = self.combo_routine.findText(routine_name)
            if idx >= 0:
                self.combo_routine.setCurrentIndex(idx)
            self.combo_routine.blockSignals(False)

        if self.combo_detector.count() > 0:
            self._on_detector_changed(0)
        else:
            self.changed.emit()

    def _on_detector_changed(self, _idx: int) -> None:
        setup_name = self.combo_setup.currentText().strip()
        det_name = self.combo_detector.currentText().strip()
        sd = self._detector_setups.get(setup_name) if setup_name else None
        dets = sd.get("detectors", {}) if isinstance(sd, dict) else {}
        info = dets.get(det_name) if det_name else None
        if isinstance(info, dict):
            chs = info.get("chs", [])
            self._micro_time_ranges = info.get("micro_time_ranges", None)
            try:
                text = ", ".join(str(int(c)) for c in chs)
            except Exception:
                text = ""
            self.lineedit_channels.setText(text)
            if self._model is not None:
                try:
                    self._model.channel_numbers = np.array(chs, dtype=np.int8)
                    self._model.channel = int(chs[0]) if chs else 0
                    self._model.micro_time_ranges = self._micro_time_ranges
                except Exception:
                    pass
        self.changed.emit()

    def _on_routine_changed(self, text: str) -> None:
        if self._model is not None:
            try:
                self._model.reading_routine = text
            except Exception:
                pass
        self.changed.emit()

    def get_channels(self) -> list[int]:
        ch_text = self.lineedit_channels.text().strip()
        channels: list[int] = []
        for part in ch_text.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                channels.append(int(part))
            except Exception:
                continue
        return channels if channels else [0]

    def sync_from_reader(self, setup) -> None:
        """Sync combo state from a reader/setup object (used by updateUI)."""
        rr = getattr(setup, 'reading_routine', None)
        if isinstance(rr, str) and rr:
            idx = self.combo_routine.findText(rr)
            if idx >= 0:
                self.combo_routine.blockSignals(True)
                self.combo_routine.setCurrentIndex(idx)
                self.combo_routine.blockSignals(False)

        chs = getattr(setup, 'channel_numbers', None)
        if chs is None:
            chs = [getattr(setup, 'channel', 0)]
        try:
            seq = list(chs)
        except TypeError:
            seq = [chs]
        self.lineedit_channels.setText(", ".join(str(int(c)) for c in seq))


class _RicsRoiWidget(QtWidgets.QWidget):
    """X/Y ROI range spinboxes in a compact 2×2 grid."""

    changed = QtCore.Signal()

    def __init__(self, model, target=None, parent=None, **kwargs):
        super().__init__(parent)
        self._model = model

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setSpacing(4)

        grid.addWidget(QtWidgets.QLabel("X range:"), 0, 0)
        self.spin_x0 = QtWidgets.QSpinBox()
        self.spin_x0.setRange(0, 4096)
        self.spin_x0.setValue(0)
        self.spin_x1 = QtWidgets.QSpinBox()
        self.spin_x1.setRange(-1, 4096)
        self.spin_x1.setValue(-1)
        grid.addWidget(self.spin_x0, 0, 1)
        grid.addWidget(self.spin_x1, 0, 2)

        grid.addWidget(QtWidgets.QLabel("Y range:"), 1, 0)
        self.spin_y0 = QtWidgets.QSpinBox()
        self.spin_y0.setRange(0, 4096)
        self.spin_y0.setValue(0)
        self.spin_y1 = QtWidgets.QSpinBox()
        self.spin_y1.setRange(-1, 4096)
        self.spin_y1.setValue(-1)
        grid.addWidget(self.spin_y0, 1, 1)
        grid.addWidget(self.spin_y1, 1, 2)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        for sb in (self.spin_x0, self.spin_x1, self.spin_y0, self.spin_y1):
            sb.valueChanged.connect(self._on_range_changed)

    def _on_range_changed(self) -> None:
        if self._model is not None:
            x0, x1, y0, y1 = self.get_ranges()
            try:
                self._model.x_range = (x0, x1)
                self._model.y_range = (y0, y1)
            except Exception:
                pass
        self.changed.emit()

    def get_ranges(self) -> tuple[int, int, int, int]:
        return (
            self.spin_x0.value(), self.spin_x1.value(),
            self.spin_y0.value(), self.spin_y1.value(),
        )

    def set_ranges(self, x0: int, x1: int, y0: int, y1: int) -> None:
        for sb, v in (
            (self.spin_x0, x0), (self.spin_x1, x1),
            (self.spin_y0, y0), (self.spin_y1, y1),
        ):
            sb.blockSignals(True)
            try:
                sb.setValue(int(v))
            finally:
                sb.blockSignals(False)

    def update_limits(self, nx: int, ny: int) -> None:
        max_x = max(0, nx - 1)
        max_y = max(0, ny - 1)
        for sb, lo, hi in (
            (self.spin_x0, 0, max_x),
            (self.spin_x1, -1, max_x),
            (self.spin_y0, 0, max_y),
            (self.spin_y1, -1, max_y),
        ):
            sb.blockSignals(True)
            try:
                sb.setRange(int(lo), int(hi))
            finally:
                sb.blockSignals(False)

        if self.spin_x0.value() < 0:
            self.spin_x0.setValue(0)
        if self.spin_y0.value() < 0:
            self.spin_y0.setValue(0)
        x1_val = self.spin_x1.value()
        if x1_val < 0 or x1_val > max_x:
            self.spin_x1.setValue(max_x)
        y1_val = self.spin_y1.value()
        if y1_val < 0 or y1_val > max_y:
            self.spin_y1.setValue(max_y)


def _register_rics_sections() -> None:
    from chisurf.gui.autoform.sections.registry import register_section
    register_section("rics_detector")(_RicsDetectorWidget)
    register_section("rics_roi")(_RicsRoiWidget)


class RICSController(reader.ExperimentReaderController, QtWidgets.QWidget):

    def get_filename(self) -> pathlib.Path:
        try:
            self.onParametersChanged()
        except Exception:
            pass

        if getattr(self, "_preview_filename", None) is not None:
            return pathlib.Path(self._preview_filename)

        fn = cs.gui.widgets.open_files(
            description='RICS TTTR/TIFF file',
            file_type='All files (*.*)',
            working_path=None,
        )
        if isinstance(fn, (list, tuple)):
            return pathlib.Path(fn[0]) if fn else pathlib.Path("")
        return pathlib.Path(fn) if fn else pathlib.Path("")

    @property
    def filename(self) -> str:
        fn = self.get_filename()
        return str(fn) if fn is not None else ""

    def __init__(self, *args, **kwargs):
        _register_rics_sections()
        super().__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label = QtWidgets.QLabel("RICS: Drop TTTR (PTU/HT3) or TIFF here.")
        label.setWordWrap(True)
        layout.addWidget(label)

        # AutoForm for Detector and Acquisition settings
        reader_obj = getattr(self, "experiment_reader", None)
        self._settings_form = None
        self._detector_widget: _RicsDetectorWidget | None = None
        self._roi_widget: _RicsRoiWidget | None = None

        # Stub spinboxes so ROI-sync methods never crash if AutoForm is absent
        self.spin_x0 = QtWidgets.QSpinBox()
        self.spin_x0.setRange(0, 4096)
        self.spin_x1 = QtWidgets.QSpinBox()
        self.spin_x1.setRange(-1, 4096)
        self.spin_x1.setValue(-1)
        self.spin_y0 = QtWidgets.QSpinBox()
        self.spin_y0.setRange(0, 4096)
        self.spin_y1 = QtWidgets.QSpinBox()
        self.spin_y1.setRange(-1, 4096)
        self.spin_y1.setValue(-1)

        if reader_obj is not None and hasattr(reader_obj, "view_spec"):
            from chisurf.gui.autoform import AutoForm
            self._settings_form = AutoForm(reader_obj, parent=self)
            layout.addWidget(self._settings_form)

            det_widgets = self._settings_form.findChildren(_RicsDetectorWidget)
            if det_widgets:
                self._detector_widget = det_widgets[0]
                self._detector_widget.changed.connect(self.onParametersChanged)

            roi_widgets = self._settings_form.findChildren(_RicsRoiWidget)
            if roi_widgets:
                self._roi_widget = roi_widgets[0]
                self.spin_x0 = self._roi_widget.spin_x0
                self.spin_x1 = self._roi_widget.spin_x1
                self.spin_y0 = self._roi_widget.spin_y0
                self.spin_y1 = self._roi_widget.spin_y1
                self._roi_widget.changed.connect(self.onParametersChanged)

        # Backward-compat refs so that any code outside this class that uses
        # self.combo_routine / self.lineedit_channels still works.
        if self._detector_widget is not None:
            self.combo_setup = self._detector_widget.combo_setup
            self.combo_detector = self._detector_widget.combo_detector
            self.combo_routine = self._detector_widget.combo_routine
            self.lineedit_channels = self._detector_widget.lineedit_channels

        # ------------------------------------------------------------------
        # Inline image preview (bespoke, intentionally outside AutoForm)
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
        mode_row.addStretch(1)

        self.toolbtn_add_rics = QtWidgets.QToolButton(preview_group)
        self.toolbtn_add_rics.setText("Add")
        self.toolbtn_add_rics.setToolTip("Add RICS dataset for the current file")
        mode_row.addWidget(self.toolbtn_add_rics)

        self.toolbtn_clear_preview = QtWidgets.QToolButton(preview_group)
        self.toolbtn_clear_preview.setText("Clear")
        self.toolbtn_clear_preview.setToolTip("Clear preview and ROI")
        mode_row.addWidget(self.toolbtn_clear_preview)
        preview_layout.addLayout(mode_row)

        self.preview_view = pg.ImageView()
        try:
            self.preview_view.ui.histogram.setMaximumWidth(120)
        except Exception:
            pass
        try:
            view = self.preview_view.getView()
            view.setMouseEnabled(x=False, y=False)
            view.setMenuEnabled(False)
        except Exception:
            pass
        preview_layout.addWidget(self.preview_view, 1)

        self._preview_roi = None
        self._preview_img_shape = None
        self._roi_sync_in_progress = False

        layout.addWidget(preview_group, 1)

        self._preview_filename: pathlib.Path | None = None
        self._preview_rics_dirty: bool = False
        self._preview_rics_stack = None

        self._micro_time_ranges = None

        self.setAcceptDrops(True)

        try:
            self.radio_preview_intensity.toggled.connect(self._on_preview_mode_changed)
            self.radio_preview_rics.toggled.connect(self._on_preview_mode_changed)
        except Exception:
            pass
        try:
            self.toolbtn_clear_preview.clicked.connect(self._on_clear_preview_clicked)
        except Exception:
            pass
        try:
            self.toolbtn_add_rics.clicked.connect(self._on_add_rics_clicked)
        except Exception:
            pass

    def updateUI(self) -> None:
        """Sync UI from current_setup state."""
        setup = getattr(self, "experiment_reader", None)
        if setup is None:
            return

        if self._detector_widget is not None:
            try:
                self._detector_widget.sync_from_reader(setup)
            except Exception:
                pass

        if self._roi_widget is not None:
            try:
                x_range = getattr(setup, 'x_range', None)
                y_range = getattr(setup, 'y_range', None)
                if isinstance(x_range, (list, tuple)) and len(x_range) >= 2:
                    y0 = int(y_range[0]) if isinstance(y_range, (list, tuple)) and len(y_range) >= 2 else 0
                    y1 = int(y_range[1]) if isinstance(y_range, (list, tuple)) and len(y_range) >= 2 else -1
                    self._roi_widget.set_ranges(int(x_range[0]), int(x_range[1]), y0, y1)
            except Exception:
                pass

        try:
            mtr = getattr(setup, 'micro_time_ranges', None)
            if isinstance(mtr, (list, tuple)):
                self._micro_time_ranges = mtr
                if self._detector_widget is not None:
                    self._detector_widget._micro_time_ranges = mtr
        except Exception:
            pass

        if self._settings_form is not None:
            try:
                self._settings_form.rebuild()
            except Exception:
                pass

    def onParametersChanged(self) -> None:
        """Push current parameters onto the bound reader."""
        setup = getattr(self, "experiment_reader", None)
        if setup is None:
            return

        if self._detector_widget is not None:
            channels = self._detector_widget.get_channels()
            routine = self._detector_widget.combo_routine.currentText()
            setup_name = self._detector_widget.combo_setup.currentText().strip()
            detector_name = self._detector_widget.combo_detector.currentText().strip()
            self._micro_time_ranges = self._detector_widget._micro_time_ranges
        else:
            channels = [0]
            routine = getattr(setup, 'reading_routine', 'PTU') or 'PTU'
            setup_name = ''
            detector_name = ''

        channel_numbers_expr = ", ".join(str(int(c)) for c in channels)
        first_channel = int(channels[0])
        setup_name_esc = setup_name.replace("'", "\\'")
        detector_name_esc = detector_name.replace("'", "\\'")

        x0 = int(self.spin_x0.value())
        x1 = int(self.spin_x1.value())
        y0 = int(self.spin_y0.value())
        y1 = int(self.spin_y1.value())

        # AutoForm keeps these reader attrs live; read back for cs.run()
        subtract_token = getattr(setup, 'subtract_average', '') or ''
        frame_shift = int(getattr(setup, 'frame_shift', 0) or 0)
        fftshift_flag = bool(getattr(setup, 'fftshift', True))
        framewise_flag = bool(getattr(setup, 'framewise_rics', False))
        pixel_dur_val = float(getattr(setup, 'pixel_duration', None) or 0.0)
        line_dur_val = float(getattr(setup, 'line_duration', None) or 0.0)

        fftshift_str = "True" if fftshift_flag else "False"
        framewise_str = "True" if framewise_flag else "False"
        pixel_dur_expr = repr(pixel_dur_val) if pixel_dur_val > 0.0 else "None"
        line_dur_expr = repr(line_dur_val) if line_dur_val > 0.0 else "None"

        mtr_list: list = []
        mtr_value = self._micro_time_ranges
        if isinstance(mtr_value, (list, tuple)):
            for r in mtr_value:
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    try:
                        mtr_list.append([int(r[0]), int(r[1])])
                    except Exception:
                        pass
        micro_time_ranges_expr = repr(mtr_list)

        try:
            cs.run(
                "\n".join([
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
                    "cs.current_setup._cache_ics_stack = None",
                    "cs.current_setup._cache_filename = None",
                ])
            )
        except Exception:
            pass

        try:
            if not getattr(self, "_roi_sync_in_progress", False):
                self._sync_preview_roi_to_ranges()
        except Exception:
            pass

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
        try:
            show_intensity = bool(self.radio_preview_intensity.isChecked())
            show_rics = bool(self.radio_preview_rics.isChecked())
            if not (show_intensity or show_rics):
                return
        except Exception:
            self._refresh_preview_image()
            return

        if show_rics and getattr(self, "_preview_filename", None) is not None:
            if getattr(self, "_preview_rics_dirty", False):
                try:
                    self._load_preview_from_file(self._preview_filename)  # type: ignore[arg-type]
                    return
                except Exception:
                    pass

        self._refresh_preview_image()

    def _on_clear_preview_clicked(self) -> None:
        try:
            self._preview_filename = None
        except Exception:
            pass

        for attr in ("_preview_intensity_stack", "_preview_intensity_mean",
                     "_preview_rics_mean", "_preview_rics_stack"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass

        try:
            self._preview_img_shape = None
        except Exception:
            pass
        try:
            self._preview_rics_dirty = False
        except Exception:
            pass

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

        try:
            self.preview_view.clear()
        except Exception:
            try:
                self.preview_view.setImage(np.zeros((1, 1), dtype=float))
            except Exception:
                pass

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

        # Also reset the roi_widget model attrs so AutoForm stays in sync
        if self._roi_widget is not None and self._roi_widget._model is not None:
            try:
                self._roi_widget._model.x_range = (0, -1)
                self._roi_widget._model.y_range = (0, -1)
            except Exception:
                pass

        try:
            self.onParametersChanged()
        except Exception:
            pass

    def _on_add_rics_clicked(self) -> None:
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
            p = pathlib.Path(path)
        except Exception:
            try:
                p = pathlib.Path(str(path))
            except Exception:
                return

        try:
            self.onParametersChanged()
        except Exception:
            pass

        s = p.as_posix().replace("\\", "/")
        cs.core.actions.dispatch(
            name="dataset.add",
            payload={"filename": s, "experiment_reader": None},
        )

    def _load_preview_from_file(self, path: pathlib.Path) -> None:
        try:
            self.onParametersChanged()
        except Exception:
            pass

        try:
            reader_obj = getattr(self, "experiment_reader", None)
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

        # Update timing attrs on reader; AutoForm rebuild() will reflect them
        pd = rics_meta.get('pixel_duration_us', None)
        ld = rics_meta.get('line_duration_ms', None)
        try:
            if isinstance(pd, (int, float)) and pd > 0.0:
                reader_obj.pixel_duration_us = float(pd)
        except Exception:
            pass
        try:
            if isinstance(ld, (int, float)) and ld > 0.0:
                reader_obj.line_duration_ms = float(ld)
        except Exception:
            pass
        if self._settings_form is not None:
            try:
                self._settings_form.rebuild()
            except Exception:
                pass

        if self._preview_rics_mean is None and self._preview_intensity_stack is not None:
            try:
                arr = np.asarray(self._preview_intensity_stack, dtype=float)
                if arr.ndim == 3:
                    self._preview_rics_mean = arr.mean(axis=0)
            except Exception:
                pass

        try:
            self._preview_rics_dirty = False
        except Exception:
            pass

        self._refresh_preview_image()

    def _refresh_preview_image(self) -> None:
        try:
            show_intensity = bool(self.radio_preview_intensity.isChecked())
        except Exception:
            show_intensity = True
        try:
            setup = getattr(self, "experiment_reader", None)
            framewise = bool(getattr(setup, 'framewise_rics', False))
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
            try:
                self._update_range_spin_limits_from_image(nx, ny)
            except Exception:
                pass

        try:
            if arr.ndim == 3:
                self.preview_view.setImage(arr, axes={"t": 0, "y": 1, "x": 2})
            elif arr.ndim == 2:
                self.preview_view.setImage(arr)
        except Exception:
            pass

        try:
            self._sync_preview_roi_to_ranges()
            roi = getattr(self, "_preview_roi", None)
            if roi is not None:
                roi.setVisible(show_intensity)
        except Exception:
            pass

    def _update_range_spin_limits_from_image(self, nx: int, ny: int) -> None:
        if self._roi_widget is not None:
            self._roi_widget.update_limits(nx, ny)
        else:
            # Fallback for stub spinboxes
            max_x = max(0, nx - 1)
            max_y = max(0, ny - 1)
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
            if self.spin_x0.value() < 0:
                self.spin_x0.setValue(0)
            if self.spin_y0.value() < 0:
                self.spin_y0.setValue(0)
            x1 = self.spin_x1.value()
            if x1 < 0 or x1 > max_x:
                self.spin_x1.setValue(max_x)
            y1 = self.spin_y1.value()
            if y1 < 0 or y1 > max_y:
                self.spin_y1.setValue(max_y)

    def _ensure_preview_roi(self):
        if self._preview_roi is not None:
            return self._preview_roi
        try:
            roi = pg.RectROI(
                [0, 0], [10, 10],
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

        # Update model ranges directly
        if self._roi_widget is not None and self._roi_widget._model is not None:
            try:
                self._roi_widget._model.x_range = (x0, x1)
                self._roi_widget._model.y_range = (y0, y1)
            except Exception:
                pass

        try:
            self.onParametersChanged()
        except Exception:
            pass
