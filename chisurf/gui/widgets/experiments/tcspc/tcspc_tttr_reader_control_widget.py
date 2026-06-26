from __future__ import annotations

import pathlib

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui

import chisurf as cs
from chisurf.core.experiments.core import reader
from chisurf.gui import QtWidgets
from chisurf.gui.widgets.sample_picker import show_sample_picker_dialog
from chisurf.gui.widgets.wizard.tttr_channeldefinition import load_detector_setups


class _TcspcTttrDetectorWidget(QtWidgets.QWidget):
    """Detector cascade for TCSPC-TTTR: Setup → Detector → Routine + channels.

    Bound directly to the reader (``model``). Selecting a detector writes the
    routing channels and reading routine onto the reader and remembers the
    chosen setup/detector names so the editor rebuilds idempotently.
    """

    changed = QtCore.Signal()

    def __init__(self, model, target=None, parent=None, **kwargs):
        super().__init__(parent)
        self._model = model
        self._detector_setups: dict = {}

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setSpacing(3)

        grid.addWidget(QtWidgets.QLabel("Setup:"), 0, 0)
        self.combo_setup = QtWidgets.QComboBox()
        grid.addWidget(self.combo_setup, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Detector:"), 0, 2)
        self.combo_detector = QtWidgets.QComboBox()
        grid.addWidget(self.combo_detector, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Routine:"), 1, 0)
        self.combo_routine = QtWidgets.QComboBox()
        try:
            import tttrlib  # type: ignore[import]
            supported = list(tttrlib.TTTR.get_supported_container_names()) or ["PTU", "HT3", "SPC"]
            self.combo_routine.addItems([str(s) for s in supported])
        except Exception:
            self.combo_routine.addItems(["PTU", "HT3", "SPC"])
        grid.addWidget(self.combo_routine, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Routing chs:"), 1, 2)
        self.lineedit_channels = QtWidgets.QLineEdit()
        self.lineedit_channels.setPlaceholderText("e.g. 0,3 or from detector setup")
        self.lineedit_channels.setToolTip(
            "Routing channels used for the decay histogram; populated from the "
            "detector setup or entered manually (comma-separated)."
        )
        grid.addWidget(self.lineedit_channels, 1, 3)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.combo_setup.currentIndexChanged.connect(self._on_setup_changed)
        self.combo_detector.currentIndexChanged.connect(self._on_detector_changed)
        self.combo_routine.currentTextChanged.connect(self._on_routine_changed)
        self.lineedit_channels.editingFinished.connect(self._on_channels_edited)

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
        prev = str(getattr(self._model, "detector_setup", "") or "")
        if prev:
            idx = self.combo_setup.findText(prev)
            if idx >= 0:
                self.combo_setup.setCurrentIndex(idx)
        self.combo_setup.blockSignals(False)

        if self.combo_setup.count() > 0:
            self._on_setup_changed(self.combo_setup.currentIndex(), restore=True)

    def _on_setup_changed(self, _idx: int, restore: bool = False) -> None:
        setup_name = self.combo_setup.currentText().strip()
        sd = self._detector_setups.get(setup_name) if setup_name else None
        dets = sd.get("detectors", {}) if isinstance(sd, dict) else {}

        self.combo_detector.blockSignals(True)
        self.combo_detector.clear()
        for det_name in sorted(dets.keys()):
            self.combo_detector.addItem(str(det_name))
        if restore:
            prev = str(getattr(self._model, "detector_name", "") or "")
            if prev:
                idx = self.combo_detector.findText(prev)
                if idx >= 0:
                    self.combo_detector.setCurrentIndex(idx)
        self.combo_detector.blockSignals(False)

        try:
            reading = sd.get("tttr_reading", {}) if isinstance(sd, dict) else {}
            routine_name = reading.get("file_type") if isinstance(reading, dict) else None
        except Exception:
            routine_name = None
        if isinstance(routine_name, str) and routine_name:
            self.combo_routine.blockSignals(True)
            ri = self.combo_routine.findText(routine_name)
            if ri >= 0:
                self.combo_routine.setCurrentIndex(ri)
            self.combo_routine.blockSignals(False)

        if self.combo_detector.count() > 0:
            self._apply_detector(emit=not restore)
        elif not restore:
            self.changed.emit()

    def _on_detector_changed(self, _idx: int) -> None:
        self._apply_detector(emit=True)

    def _apply_detector(self, emit: bool = True) -> None:
        setup_name = self.combo_setup.currentText().strip()
        det_name = self.combo_detector.currentText().strip()
        sd = self._detector_setups.get(setup_name) if setup_name else None
        dets = sd.get("detectors", {}) if isinstance(sd, dict) else {}
        info = dets.get(det_name) if det_name else None
        if isinstance(info, dict):
            chs = info.get("chs", [])
            try:
                text = ", ".join(str(int(c)) for c in chs)
            except Exception:
                text = ""
            self.lineedit_channels.blockSignals(True)
            self.lineedit_channels.setText(text)
            self.lineedit_channels.blockSignals(False)
        self._write_channels_to_model()
        try:
            self._model.detector_setup = setup_name
            self._model.detector_name = det_name
        except Exception:
            pass
        if emit:
            self.changed.emit()

    def _on_routine_changed(self, text: str) -> None:
        try:
            self._model.reading_routine = text
        except Exception:
            pass
        self.changed.emit()

    def _on_channels_edited(self) -> None:
        self._write_channels_to_model()
        self.changed.emit()

    def _write_channels_to_model(self) -> None:
        try:
            self._model.channel_numbers_str = self.lineedit_channels.text()
        except Exception:
            pass

    def get_channels(self) -> list[int]:
        text = self.lineedit_channels.text().strip()
        channels: list[int] = []
        for part in text.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                channels.append(int(part))
            except Exception:
                continue
        return channels if channels else [0]

    def sync_from_reader(self, setup) -> None:
        rr = getattr(setup, "reading_routine", None)
        if isinstance(rr, str) and rr:
            idx = self.combo_routine.findText(rr)
            if idx >= 0:
                self.combo_routine.blockSignals(True)
                self.combo_routine.setCurrentIndex(idx)
                self.combo_routine.blockSignals(False)
        chs = getattr(setup, "channel_numbers", None)
        if chs is None:
            chs = [getattr(setup, "channel", 0)]
        try:
            seq = list(chs)
        except TypeError:
            seq = [chs]
        self.lineedit_channels.blockSignals(True)
        self.lineedit_channels.setText(", ".join(str(int(c)) for c in seq))
        self.lineedit_channels.blockSignals(False)


def _register_tcspc_tttr_sections() -> None:
    from chisurf.gui.autoform.sections.registry import register_section
    register_section("tcspc_tttr_detector")(_TcspcTttrDetectorWidget)


class TCSPCTTTRReaderControlWidget(
    reader.ExperimentReaderController,
    QtWidgets.QWidget,
):
    """TCSPC-TTTR controller: declarative settings (AutoForm) + decay preview."""

    def __init__(self, *args, **kwargs):
        _register_tcspc_tttr_sections()
        super().__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(QtWidgets.QLabel("TCSPC: Drop TTTR (PTU/HT3/SPC) here."))

        # Declarative settings: Detector cascade + Acquisition + Anisotropy.
        reader_obj = getattr(self, "experiment_reader", None)
        self._settings_form = None
        self._detector_widget: _TcspcTttrDetectorWidget | None = None
        if reader_obj is not None and hasattr(reader_obj, "view_spec"):
            from chisurf.gui.autoform import AutoForm
            self._settings_form = AutoForm(reader_obj, parent=self)
            layout.addWidget(self._settings_form)
            self._bind_detector_widget()

        # Preview group: decay plot with Compute/Add/Clear buttons (bespoke).
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

        # Internal preview state
        self._preview_filename: pathlib.Path | None = None
        self._preview_t = None
        self._preview_y = None
        self._preview_channels = None

        self.setAcceptDrops(True)

        self.toolbtn_clear.clicked.connect(self._on_clear_preview_clicked)
        self.toolbtn_add.clicked.connect(self._on_add_clicked)
        self.toolbtn_compute.clicked.connect(self._on_compute_clicked)

    # ------------------------------------------------------------------
    # AutoForm wiring
    # ------------------------------------------------------------------

    def _bind_detector_widget(self) -> None:
        if self._settings_form is None:
            return
        widgets = self._settings_form.findChildren(_TcspcTttrDetectorWidget)
        if widgets:
            self._detector_widget = widgets[0]
            self.combo_routine = self._detector_widget.combo_routine
            self.lineedit_channels = self._detector_widget.lineedit_channels
            self._detector_widget.changed.connect(self._on_gui_parameters_changed)

    def _db(self):
        try:
            return self.db
        except Exception:
            return None

    def _select_sample_id(self) -> str | None:
        return show_sample_picker_dialog(db=self._db(), parent=self)

    def _set_reader_sample_id(self, sample_id: str | None) -> None:
        reader_obj = self._reader_for_current_setup()
        if reader_obj is None:
            return
        try:
            reader_obj.sample_id = sample_id
        except Exception:
            pass

    def _reader_for_current_setup(self):
        """Return the reader this controller is bound to."""
        return getattr(self, "experiment_reader", None)

    def get_filename(self) -> pathlib.Path:
        try:
            self.onParametersChanged()
        except Exception:
            pass
        fn_prev = getattr(self, "_preview_filename", None)
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
        reader_obj = getattr(self, "experiment_reader", None)
        if reader_obj is None:
            return
        if self._detector_widget is not None:
            try:
                self._detector_widget.sync_from_reader(reader_obj)
            except Exception:
                pass
        if self._settings_form is not None:
            try:
                self._settings_form.rebuild()
                self._bind_detector_widget()
            except Exception:
                pass

    def onParametersChanged(self):
        """Ensure the detector selection (routine + channels) is on the reader.

        The Acquisition / Anisotropy fields are live-bound to the reader via
        AutoForm, so only the detector cascade values are pushed here.
        """
        reader_obj = getattr(self, "experiment_reader", None)
        if reader_obj is None or self._detector_widget is None:
            return
        try:
            reader_obj.reading_routine = self._detector_widget.combo_routine.currentText()
        except Exception:
            pass
        try:
            reader_obj.channel_numbers_str = self._detector_widget.lineedit_channels.text()
        except Exception:
            pass

    def _on_gui_parameters_changed(self) -> None:
        try:
            self.onParametersChanged()
        except Exception:
            pass
        path = getattr(self, "_preview_filename", None)
        if not path:
            return
        try:
            p = path if isinstance(path, pathlib.Path) else pathlib.Path(str(path))
            self._load_preview_from_file(p)
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
                self._preview_filename = path
                try:
                    self._load_preview_from_file(path)
                except Exception:
                    pass
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def _on_clear_preview_clicked(self) -> None:
        self._preview_filename = None
        self._preview_t = None
        self._preview_y = None
        self._preview_channels = None
        try:
            self.preview_plot.clear()
        except Exception:
            pass

    def _on_compute_clicked(self) -> None:
        try:
            self.onParametersChanged()
        except Exception:
            pass
        path = getattr(self, "_preview_filename", None)
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
        self._preview_filename = p
        try:
            self._load_preview_from_file(p)
        except Exception:
            pass

    def _on_add_clicked(self) -> None:
        path = getattr(self, "_preview_filename", None)
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
        sample_id = self._select_sample_id()
        reader_obj = self._reader_for_current_setup()
        self._set_reader_sample_id(sample_id)
        try:
            s = p.as_posix().replace("\\", "/")
        except Exception:
            return
        cs.core.actions.dispatch(
            name="dataset.add",
            payload={"filename": s, "experiment_reader": reader_obj},
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

        reader_obj = getattr(self, "experiment_reader", None)
        if reader_obj is None:
            return

        # Direct TTTR-based microtime histogram per routing channel
        try:
            import tttrlib  # type: ignore[import]

            routine = getattr(reader_obj, "reading_routine", None)
            chs_value = getattr(reader_obj, "channel_numbers", None)
            if chs_value is None:
                chs_value = [getattr(reader_obj, "channel", 0)]
            try:
                ch_list = sorted({int(c) for c in chs_value})
            except Exception:
                ch_list = [int(getattr(reader_obj, "channel", 0) or 0)]

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
                self._preview_channels = ch_list
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
        self._preview_channels = None
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
        chs = getattr(self, "_preview_channels", None)
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
