from __future__ import annotations

import pathlib

from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import chisurf.gui.widgets
from chisurf.experiments.core import reader
from chisurf.gui.widgets.wizard.tttr_channeldefinition import load_detector_setups


class PCHController(reader.ExperimentReaderController, QtWidgets.QWidget):

    def get_filename(self) -> pathlib.Path:
        import chisurf

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

        fn = chisurf.gui.widgets.open_files(
            description='PCH TTTR file',
            file_type='TTTR files (*.ptu *.ht3 *.t2r *.t3r);;All files (*.*)',
            working_path=None,
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
            "PCH loader (TTTR → P(k))\n"
            "Drop TTTR files (e.g. PTU/HT3) here. The reader will compute a\n"
            "photon-count histogram P(k) suitable for PCH fitting."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        # Detector setup + logical detector selection (from channel wizard JSON)
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
        file_params_layout.addWidget(self.combo_routine, 1, 1)

        file_params_layout.addWidget(QtWidgets.QLabel("Routing chs:"), 1, 2)
        self.lineedit_channels = QtWidgets.QLineEdit(self)
        self.lineedit_channels.setPlaceholderText("e.g. 0,2")
        file_params_layout.addWidget(self.lineedit_channels, 1, 3)

        layout.addLayout(file_params_layout)

        # Timing / micro-time controls
        opts = QtWidgets.QGridLayout()
        opts.setContentsMargins(0, 0, 0, 0)
        opts.setSpacing(0)

        opts.addWidget(QtWidgets.QLabel("Bin time [µs]"), 0, 0)
        self.spin_bin = QtWidgets.QDoubleSpinBox(self)
        self.spin_bin.setDecimals(3)
        self.spin_bin.setRange(0.1, 1.0e6)
        self.spin_bin.setValue(100.0)
        opts.addWidget(self.spin_bin, 0, 1, 1, 2)

        opts.addWidget(QtWidgets.QLabel("Micro time min"), 1, 0)
        self.spin_mt_min = QtWidgets.QSpinBox(self)
        self.spin_mt_min.setRange(0, 65535)
        self.spin_mt_min.setValue(0)
        opts.addWidget(self.spin_mt_min, 1, 1)

        opts.addWidget(QtWidgets.QLabel("max"), 1, 2)
        self.spin_mt_max = QtWidgets.QSpinBox(self)
        self.spin_mt_max.setRange(0, 65535)
        self.spin_mt_max.setValue(65535)
        opts.addWidget(self.spin_mt_max, 1, 3)

        layout.addLayout(opts)

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
        self.toolbtn_compute_pch.setText("Compute")
        self.toolbtn_compute_pch.setToolTip("Compute P(k) preview for the current file")
        header.addWidget(self.toolbtn_compute_pch)

        self.toolbtn_add_pch = QtWidgets.QToolButton(preview_group)
        self.toolbtn_add_pch.setText("Add")
        self.toolbtn_add_pch.setToolTip("Add PCH dataset for the current file")
        header.addWidget(self.toolbtn_add_pch)

        self.toolbtn_clear_preview = QtWidgets.QToolButton(preview_group)
        self.toolbtn_clear_preview.setText("Clear")
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

        # Cache of detector setups loaded from the central JSON
        self._detector_setups = {}

        # Enable drag-and-drop of TTTR files at the controller level. We rely
        # on Qt's default propagation so drops anywhere inside the controller
        # (including on the plot) are handled by dragEnterEvent/dropEvent
        # implemented on this widget, mirroring the RICS controller.
        self.setAcceptDrops(True)

        # Wire parameter changes
        try:
            self.combo_routine.currentTextChanged.connect(self.onParametersChanged)
        except Exception:
            pass
        try:
            self.lineedit_channels.editingFinished.connect(self.onParametersChanged)
        except Exception:
            pass
        for w in (self.spin_bin, self.spin_mt_min, self.spin_mt_max):
            try:
                w.valueChanged.connect(self.onParametersChanged)
            except Exception:
                pass

        # Add / Clear / Compute preview
        try:
            self.toolbtn_clear_preview.clicked.connect(self._on_clear_preview_clicked)
        except Exception:
            pass
        try:
            self.toolbtn_add_pch.clicked.connect(self._on_add_pch_clicked)
        except Exception:
            pass
        try:
            self.toolbtn_compute_pch.clicked.connect(self._on_compute_pch_clicked)
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

    @property
    def filename(self) -> str:
        fn = self.get_filename()
        return str(fn) if fn is not None else ""

    def updateUI(self):
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

        # bin time
        try:
            bt = float(getattr(setup, 'bin_time_us', self.spin_bin.value()))
            if bt > 0.0:
                self.spin_bin.setValue(bt)
        except Exception:
            pass

        # micro time range
        try:
            mtr = getattr(setup, 'micro_time_range', None)
            if isinstance(mtr, (tuple, list)) and len(mtr) >= 2:
                self.spin_mt_min.setValue(int(mtr[0]))
                self.spin_mt_max.setValue(int(mtr[1]))
        except Exception:
            pass

    def onParametersChanged(self):
        import chisurf

        routine = self.combo_routine.currentText()

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

        bt = float(self.spin_bin.value())
        mt_min = int(self.spin_mt_min.value())
        mt_max = int(self.spin_mt_max.value())

        try:
            chisurf.run(
                "\n".join(
                    [
                        f"cs.current_setup.reading_routine = '{routine}'",
                        f"cs.current_setup.channel_numbers = np.array([{channel_numbers_expr}], dtype=np.int8)",
                        f"cs.current_setup.channel = {first_channel}",
                        f"cs.current_setup.bin_time_us = {bt}",
                        f"cs.current_setup.micro_time_range = ({mt_min}, {mt_max})",
                    ]
                )
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Detector setup / detector selection based on detector_setups.json
    # ------------------------------------------------------------------

    def _reload_detector_setups(self) -> None:
        """Load detector setups from the central JSON and populate combo_setup.

        This mirrors the behaviour of the RICS loader but only exposes setup
        name and detector name, which are used to derive routing channels and
        optional micro-time windows for PCH.
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

        Additionally, try to synchronize the TTTR reading routine combo with
        the setup's stored ``tttr_reading.file_type`` when available.
        """

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
        """Update routing channels and micro-time range from detector definition."""

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
        mtr = info.get("micro_time_ranges", None)

        # Update routing channels text box
        try:
            text = ", ".join(str(int(c)) for c in chs)
        except Exception:
            text = ""
        try:
            self.lineedit_channels.setText(text)
        except Exception:
            pass

        # Optionally update micro-time spin boxes from the first defined range
        try:
            ranges = mtr
            if isinstance(ranges, (list, tuple)) and ranges:
                first = ranges[0]
                if isinstance(first, dict):
                    # Support structures like {"range": [start, end]}
                    first = first.get("range", first.get("values", first))
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    mt_min = int(first[0])
                    mt_max = int(first[1])
                    self.spin_mt_min.setValue(mt_min)
                    self.spin_mt_max.setValue(mt_max)
        except Exception:
            pass

        # Propagate updated detector selection into current_setup
        try:
            self.onParametersChanged()
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
            from chisurf.data import ExperimentDataCurveGroup

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

        combined_hist = None
        combined_total_bins = 0.0
        max_len = 0

        for path in paths:
            try:
                group = reader_obj.read(filename=path.as_posix())
            except Exception:
                continue

            try:
                from chisurf.data import ExperimentDataCurveGroup

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

    def _on_add_pch_clicked(self) -> None:
        import chisurf
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

        commands = []
        for p in path_list:
            try:
                s = p.as_posix().replace("\\", "/")
            except Exception:
                continue
            commands.append(f"chisurf.macros.add_dataset(filename=r\"{s}\")")

        if not commands:
            return

        try:
            chisurf.run("\n".join(commands))
        except Exception:
            pass


__all__ = ["PCHController"]
