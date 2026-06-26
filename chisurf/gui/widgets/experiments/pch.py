from __future__ import annotations

import pathlib

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

import chisurf as cs
import chisurf.gui.widgets
from chisurf.core.experiments.core import reader
from chisurf.gui.widgets.sample_picker import show_sample_picker_dialog
from chisurf.gui.widgets.wizard.tttr_channeldefinition import load_detector_setups


class _PchDetectorWidget(QtWidgets.QWidget):
    """Detector cascade for PCH: Setup → Detector → Routine + editable channels.

    Binds directly to the reader (``model``). Selecting a detector writes the
    routing channels, micro-time window, reading routine and the chosen
    setup/detector names back onto the reader so the editor can be rebuilt
    idempotently.
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
        self.lineedit_channels.setPlaceholderText("e.g. 0,2")
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
        # Restore previously selected setup from the model when present
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

        # Sync reading routine from the setup's tttr_reading.file_type
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

            # Micro-time window from the first defined range
            mtr = info.get("micro_time_ranges", None)
            if isinstance(mtr, (list, tuple)) and mtr:
                first = mtr[0]
                if isinstance(first, dict):
                    first = first.get("range", first.get("values", first))
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    try:
                        self._model.micro_time_range = (int(first[0]), int(first[1]))
                    except Exception:
                        pass

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
        channels = self.get_channels()
        try:
            self._model.channel_numbers = np.array(channels, dtype=np.int8)
            self._model.channel = int(channels[0])
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
        """Refresh routine/channels display from a reader/setup object."""
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
        self.lineedit_channels.blockSignals(True)
        self.lineedit_channels.setText(", ".join(str(int(c)) for c in seq))
        self.lineedit_channels.blockSignals(False)


def _register_pch_sections() -> None:
    from chisurf.gui.autoform.sections.registry import register_section
    register_section("pch_detector")(_PchDetectorWidget)


class PCHController(reader.ExperimentReaderController, QtWidgets.QWidget):

    def get_filename(self) -> pathlib.Path:
        # Ensure current GUI values are pushed before opening the dialog.
        try:
            self.onParametersChanged()
        except Exception:
            pass

        # Reuse last dropped/previewed file when available.
        try:
            fn_prev = getattr(self, "_preview_filename", None)
        except Exception:
            fn_prev = None
        if fn_prev:
            return pathlib.Path(fn_prev)

        fn = cs.gui.widgets.open_files(
            description='PCH TTTR file',
            file_type='TTTR files (*.ptu *.ht3 *.t2r *.t3r);;All files (*.*)',
            working_path=None,
        )
        if isinstance(fn, (list, tuple)):
            return pathlib.Path(fn[0]) if fn else pathlib.Path("")
        return pathlib.Path(fn) if fn else pathlib.Path("")

    def __init__(self, *args, **kwargs):
        _register_pch_sections()
        super().__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        layout.addWidget(QtWidgets.QLabel("PCH: Drop TTTR (PTU/HT3) here."))

        # AutoForm covers the Detector and Acquisition panels (declarative spec)
        reader_obj = getattr(self, "experiment_reader", None)
        self._settings_form = None
        self._detector_widget: _PchDetectorWidget | None = None
        self._refresh_pending = False
        if reader_obj is not None and hasattr(reader_obj, "view_spec"):
            from chisurf.gui.autoform import AutoForm
            self._settings_form = AutoForm(reader_obj, parent=self)
            layout.addWidget(self._settings_form)
            self._bind_detector_widget()

        # P(k) preview
        preview_group = QtWidgets.QGroupBox("Preview (drop TTTR here)")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(0)
        header.addStretch(1)

        self.toolbtn_compute_pch = QtWidgets.QToolButton(preview_group)
        self.toolbtn_compute_pch.setText("🔬 Compute")
        self.toolbtn_compute_pch.setToolTip("Compute P(k) preview for the current file")
        header.addWidget(self.toolbtn_compute_pch)

        self.toolbtn_add_pch = QtWidgets.QToolButton(preview_group)
        self.toolbtn_add_pch.setText("➕ Add")
        self.toolbtn_add_pch.setToolTip("Add PCH dataset for the current file")
        header.addWidget(self.toolbtn_add_pch)

        self.toolbtn_clear_preview = QtWidgets.QToolButton(preview_group)
        self.toolbtn_clear_preview.setText("🧹 Clear")
        self.toolbtn_clear_preview.setToolTip("Clear preview")
        header.addWidget(self.toolbtn_clear_preview)
        preview_layout.addLayout(header)

        self.preview_plot = pg.PlotWidget(title="Photon Counting Histogram")
        try:
            self.preview_plot.setLogMode(y=True)
        except Exception:
            pass
        self.preview_plot.setLabel("bottom", "k")
        self.preview_plot.setLabel("left", "P(k)")
        try:
            self.preview_plot.setMaximumHeight(200)
        except Exception:
            pass
        preview_layout.addWidget(self.preview_plot)

        layout.addWidget(preview_group)

        # Internal preview state
        self._preview_filename: pathlib.Path | None = None
        self._preview_k = None
        self._preview_p = None
        self._preview_file_list = None

        # Enable drag-and-drop of TTTR files at the controller level.
        self.setAcceptDrops(True)

        # Preview buttons
        self.toolbtn_clear_preview.clicked.connect(self._on_clear_preview_clicked)
        self.toolbtn_add_pch.clicked.connect(self._on_add_pch_clicked)
        self.toolbtn_compute_pch.clicked.connect(self._on_compute_pch_clicked)

    def _bind_detector_widget(self) -> None:
        """Grab the AutoForm-built detector widget and expose its child refs.

        Keeps ``self.combo_routine`` / ``self.lineedit_channels`` available so
        the load/preview paths can read the current routine and channels.
        """
        if self._settings_form is None:
            return
        widgets = self._settings_form.findChildren(_PchDetectorWidget)
        if not widgets:
            return
        self._detector_widget = widgets[0]
        self.combo_routine = self._detector_widget.combo_routine
        self.lineedit_channels = self._detector_widget.lineedit_channels
        self._detector_widget.changed.connect(self._on_detector_changed)

    def _on_detector_changed(self) -> None:
        """Detector selection changed: push to setup and refresh value fields.

        The detector may have rewritten ``micro_time_range`` on the reader; a
        deferred rebuild keeps the Acquisition value fields in sync without
        deleting the widget that emitted the signal mid-callback.
        """
        self.onParametersChanged()
        if not self._refresh_pending and self._settings_form is not None:
            self._refresh_pending = True
            QtCore.QTimer.singleShot(0, self._rebuild_settings_form)

    def _rebuild_settings_form(self) -> None:
        self._refresh_pending = False
        if self._settings_form is None:
            return
        try:
            self._settings_form.rebuild()
        finally:
            self._bind_detector_widget()

    @property
    def filename(self) -> str:
        fn = self.get_filename()
        return str(fn) if fn is not None else ""

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
        """Ensure the bound reader reflects the current detector selection.

        The Acquisition fields (bin time, micro-time window) are already
        live-bound to the reader through AutoForm; this pushes the detector
        widget's routine and routing channels so a load reads consistent state.
        """
        reader_obj = getattr(self, "experiment_reader", None)
        if reader_obj is None:
            return
        if self._detector_widget is not None:
            try:
                reader_obj.reading_routine = self._detector_widget.combo_routine.currentText()
            except Exception:
                pass
            try:
                channels = self._detector_widget.get_channels()
                reader_obj.channel_numbers = np.array(channels, dtype=np.int8)
                reader_obj.channel = int(channels[0])
            except Exception:
                pass

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
            paths = [pathlib.Path(str(u.toLocalFile())) for u in urls]
            if paths:
                try:
                    self._preview_file_list = paths
                except Exception:
                    pass
                if len(paths) == 1:
                    path = paths[0]
                    self._preview_filename = path
                    try:
                        self._load_preview_from_file(path)
                    except Exception:
                        pass
                else:
                    self._preview_filename = paths[0]
                    try:
                        self._load_preview_from_paths(paths)
                    except Exception:
                        pass
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def _load_preview_from_file(self, path: pathlib.Path) -> None:
        try:
            self.onParametersChanged()
        except Exception:
            pass

        reader_obj = getattr(self, "experiment_reader", None)
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
        pch_meta = meta.get('pch', {}) or {}

        k_vals = pch_meta.get('k_vals', getattr(data_obj, 'x', None))
        p_exp = pch_meta.get('p_exp', getattr(data_obj, 'y', None))

        if k_vals is None or p_exp is None:
            return

        try:
            k_seq = list(k_vals)
            p_seq = list(p_exp)
        except Exception:
            return
        if not k_seq or not p_seq or len(k_seq) != len(p_seq):
            return

        self._preview_k = k_seq
        self._preview_p = p_seq
        self._refresh_preview_plot()

    def _load_preview_from_paths(self, paths) -> None:
        import numpy as np

        try:
            self.onParametersChanged()
        except Exception:
            pass

        reader_obj = getattr(self, "experiment_reader", None)
        if reader_obj is None:
            return

        combined_hist = None
        combined_total_bins = 0.0
        max_len = 0

        for path in paths:
            try:
                group = reader_obj.read(filename=path.as_posix())
            except Exception:
                continue

            try:
                from chisurf.core.data import ExperimentDataCurveGroup

                if isinstance(group, ExperimentDataCurveGroup) and len(group) > 0:
                    data_obj = group[0]
                else:
                    data_obj = group
            except Exception:
                data_obj = group

            meta = getattr(data_obj, 'meta_data', {}) or {}
            pch_meta = meta.get('pch', {}) or {}

            hist_counts = pch_meta.get('hist_counts', None)
            total_bins = pch_meta.get('total_bins', None)
            if hist_counts is None or total_bins is None:
                continue

            try:
                h = np.asarray(hist_counts, dtype=float)
            except Exception:
                continue
            if h.size == 0:
                continue

            try:
                tb = float(total_bins)
            except Exception:
                tb = 0.0
            if not (tb > 0.0):
                continue

            if combined_hist is None:
                combined_hist = h.copy()
                max_len = h.size
            else:
                if h.size > max_len:
                    pad = h.size - max_len
                    combined_hist = np.pad(combined_hist, (0, pad), mode="constant")
                    max_len = h.size
                elif h.size < max_len:
                    h = np.pad(h, (0, max_len - h.size), mode="constant")
                combined_hist = combined_hist + h

            combined_total_bins += tb

        if combined_hist is None or not (combined_total_bins > 0.0):
            return

        k_vals = np.arange(combined_hist.size, dtype=float)
        p_exp = combined_hist.astype(float) / float(combined_total_bins)

        try:
            self._preview_k = list(k_vals)
            self._preview_p = list(p_exp)
        except Exception:
            return

        self._refresh_preview_plot()

    def _refresh_preview_plot(self) -> None:
        if not self._preview_k or not self._preview_p:
            return
        try:
            self.preview_plot.clear()
            self.preview_plot.plot(self._preview_k, self._preview_p, pen=None, symbol='o')
            try:
                self.preview_plot.setLogMode(y=True)
            except Exception:
                pass
        except Exception:
            pass

    def _on_clear_preview_clicked(self) -> None:
        try:
            self._preview_filename = None
        except Exception:
            pass
        try:
            self._preview_k = None
            self._preview_p = None
        except Exception:
            pass
        try:
            self.preview_plot.clear()
        except Exception:
            pass
        try:
            self._preview_file_list = None
        except Exception:
            pass

    def _on_compute_pch_clicked(self) -> None:
        import pathlib as _pathlib

        # Recompute P(k) preview for the currently selected or newly chosen file.
        try:
            self.onParametersChanged()
        except Exception:
            pass

        try:
            paths = getattr(self, "_preview_file_list", None)
        except Exception:
            paths = None
        if paths:
            try:
                import pathlib as _pl
                path_objs = []
                for pp in paths:
                    if isinstance(pp, _pl.Path):
                        path_objs.append(pp)
                    else:
                        path_objs.append(_pl.Path(str(pp)))
            except Exception:
                path_objs = []
            if not path_objs:
                return
            try:
                self._preview_file_list = path_objs
                self._preview_filename = path_objs[0]
            except Exception:
                pass
            try:
                self._load_preview_from_paths(path_objs)
            except Exception:
                pass
            return

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

    def _db(self):
        """Return the MFDB connection from the current reader when available."""
        try:
            return self.db
        except Exception:
            return None

    def _reader_for_current_setup(self):
        """Return the reader this controller is bound to."""
        return getattr(self, "experiment_reader", None)

    def _set_reader_sample_id(self, sample_id: str | None) -> None:
        """Store the selected sample ID on the current setup reader."""
        reader_obj = self._reader_for_current_setup()
        if reader_obj is None:
            return
        try:
            reader_obj.sample_id = sample_id
        except Exception:
            pass
        experiment_reader = getattr(reader_obj, "experiment_reader", None)
        if experiment_reader is not None:
            try:
                experiment_reader.sample_id = sample_id
            except Exception:
                pass

    def _on_add_pch_clicked(self) -> None:
        import pathlib as _pathlib

        try:
            plist = getattr(self, "_preview_file_list", None)
        except Exception:
            plist = None

        path_list = []
        if plist:
            try:
                for pp in plist:
                    if isinstance(pp, _pathlib.Path):
                        path_list.append(pp)
                    else:
                        path_list.append(_pathlib.Path(str(pp)))
            except Exception:
                path_list = []

        if not path_list:
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
            path_list = [p]

        try:
            self.onParametersChanged()
        except Exception:
            pass

        # Show sample picker once for the entire batch, not per file
        sample_id = show_sample_picker_dialog(db=self._db(), parent=self)
        if not sample_id:
            return  # User cancelled
        reader_obj = self._reader_for_current_setup()
        self._set_reader_sample_id(sample_id)

        for p in path_list:
            try:
                s = p.as_posix().replace("\\", "/")
            except Exception:
                continue
            cs.core.actions.dispatch(
                name="dataset.add",
                payload={"filename": s, "experiment_reader": reader_obj},
            )


__all__ = ["PCHController"]
