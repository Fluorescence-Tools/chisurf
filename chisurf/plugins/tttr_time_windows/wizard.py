from __future__ import annotations
import pathlib
from typing import List, Tuple

import numpy as np

# Use Qt abstraction layer
from qtpy import QtWidgets, QtCore

# Reuse the intensity plot widget for visualization
try:
    from chisurf.plugins.intensity_trace.__init__ import IntensityPlotWidget
except Exception:
    IntensityPlotWidget = None  # type: ignore

# Drag/drop list widget
from chisurf.gui.widgets.general import FileList, get_directory

# tttrlib for reading TTTR data
try:
    import tttrlib
except Exception:  # pragma: no cover
    tttrlib = None  # type: ignore

# Logging
from chisurf import logging

# Get supported TTTR extensions dynamically (via tttrlib through Trace Browser)
try:
    from chisurf.plugins.trace_browser.__init__ import get_tttr_supported_exts
except Exception:
    get_tttr_supported_exts = None  # type: ignore


def _supported_exts():
    exts = []
    try:
        if get_tttr_supported_exts is not None:
            exts = list(get_tttr_supported_exts())
        elif tttrlib is not None and hasattr(tttrlib, 'get_supported_filetypes'):
            exts = list(tttrlib.get_supported_filetypes())
    except Exception:
        exts = []
    norm = set()
    for e in exts:
        s = str(e).strip().lower()
        if not s:
            continue
        if not s.startswith('.'):
            s = '.' + s
        norm.add(s)
    if not norm:
        norm = {'.ptu', '.phu', '.ht2', '.ht3', '.pt3', '.t3r'}
    return norm


class TTTRFileList(FileList):
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            try:
                exts = _supported_exts()
            except Exception:
                exts = set()
            for url in event.mimeData().urls():
                s = str(url.toLocalFile()).lower()
                if any(s.endswith(e) or s.endswith(e + '.gz') or s.endswith(e + '.bz2') or s.endswith(e + '.zip') for e in exts):
                    event.acceptProposedAction()
                    return
        super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            try:
                exts = _supported_exts()
            except Exception:
                exts = set()
            for url in event.mimeData().urls():
                s = str(url.toLocalFile())
                sl = s.lower()
                if any(sl.endswith(e) or sl.endswith(e + '.gz') or sl.endswith(e + '.bz2') or sl.endswith(e + '.zip') for e in exts):
                    self.addItem(s)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)




def compute_bids_from_tttr(tttr: "tttrlib.TTTR", time_window_s: float) -> np.ndarray:
    """Compute start/stop photon indices for fixed time windows.

    Returns an array of shape (n_windows, 2) with [start_idx, stop_idx) per row.
    """
    if time_window_s <= 0:
        raise ValueError("time_window_s must be positive")
    mt = tttr.macro_times
    if mt is None or len(mt) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    try:
        res = float(tttr.header.macro_time_resolution)
    except Exception:
        # Fallback: try attribute directly
        res = float(getattr(tttr, 'macro_time_resolution', 0.0))
    if res <= 0:
        raise RuntimeError("Macro time resolution unavailable from TTTR header")

    clocks_per_bin = int(np.floor(time_window_s / res))
    if clocks_per_bin < 1:
        clocks_per_bin = 1

    max_clock = int(mt.max())
    if max_clock < 0:
        return np.zeros((0, 2), dtype=np.int64)

    # Bin edges in macro-time clock counts
    edges = np.arange(0, max_clock + 1, clocks_per_bin, dtype=np.int64)
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    # Compute indices for half-open intervals [edge, edge + clocks_per_bin)
    starts = np.searchsorted(mt, edges, side='left')
    stops = np.searchsorted(mt, edges + clocks_per_bin, side='left')
    bids = np.stack([starts, stops], axis=1)
    return bids


class SetupPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Setup")
        self.setSubTitle("Choose the time-window length and output folder")

        layout = QtWidgets.QFormLayout(self)

        self.tws_spin = QtWidgets.QDoubleSpinBox(self)
        self.tws_spin.setDecimals(3)
        self.tws_spin.setRange(0.001, 3600 * 1000.0)  # ms
        self.tws_spin.setSingleStep(1.0)
        self.tws_spin.setValue(10.0)
        self.tws_spin.setSuffix(" ms")
        layout.addRow("Time window:", self.tws_spin)

        h = QtWidgets.QHBoxLayout()
        self.output_edit = QtWidgets.QLineEdit(self)
        self.btn_browse = QtWidgets.QPushButton("Browse…", self)
        self.btn_browse.clicked.connect(self._choose_dir)
        h.addWidget(self.output_edit)
        h.addWidget(self.btn_browse)
        layout.addRow("Output folder:", self._wrap(h))

        self.setLayout(layout)

    def _choose_dir(self):
        d, _ = get_directory(caption="Select output folder")
        if d is not None:
            self.output_edit.setText(str(d))

    @staticmethod
    def _wrap(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setLayout(layout)
        return w

    def get_values(self) -> Tuple[float, pathlib.Path | None]:
        tw_ms = float(self.tws_spin.value())
        out = self.output_edit.text().strip()
        return tw_ms, (pathlib.Path(out) if out else None)


class FilesPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Files")
        self.setSubTitle("Drag and drop TTTR files to process")

        layout = QtWidgets.QVBoxLayout(self)

        self.file_list = TTTRFileList(accept_drops=True, filename_ending='')
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.file_list.setMinimumHeight(200)

        btns = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add…", self)
        self.btn_clear = QtWidgets.QPushButton("Clear", self)
        btns.addWidget(self.btn_add)
        btns.addStretch(1)
        btns.addWidget(self.btn_clear)

        try:
            _exts_text = " ".join(sorted(_supported_exts()))
        except Exception:
            _exts_text = ".ptu .phu .ht2 .ht3 .pt3 .t3r"
        layout.addWidget(QtWidgets.QLabel(f"Drop TTTR files here ({_exts_text})", self))
        layout.addWidget(self.file_list)
        layout.addLayout(btns)

        self.btn_add.clicked.connect(self._on_add)
        self.btn_clear.clicked.connect(self.file_list.clear)

    def _on_add(self):
        dlg = QtWidgets.QFileDialog(self, "Select TTTR files")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        exts = sorted(_supported_exts())
        pattern = " ".join(f"*{e}" for e in exts)
        dlg.setNameFilters([
            f"TTTR files ({pattern})",
            "All files (*.*)",
        ])
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            extset = set(exts)
            for s in dlg.selectedFiles():
                p = pathlib.Path(s)
                if p.suffix.lower() in extset:
                    self.file_list.addItem(str(p))

    def get_files(self) -> List[pathlib.Path]:
        return [pathlib.Path(s) for s in self.file_list.filenames]


class ProcessPage(QtWidgets.QWizardPage):
    def __init__(self, setup_page: SetupPage, files_page: FilesPage, parent=None):
        super().__init__(parent)
        self.setup_page = setup_page
        self.files_page = files_page
        self.setTitle("Preview & Process")
        self.setSubTitle("Preview intensity for one file and save BIDs for all")

        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Preview file:", self))
        self.cmb_file = QtWidgets.QComboBox(self)
        self.cmb_file.currentIndexChanged.connect(self._on_select_file)
        top.addWidget(self.cmb_file, 1)
        layout.addLayout(top)

        self.plot = IntensityPlotWidget(self) if IntensityPlotWidget is not None else None
        if self.plot is not None:
            layout.addWidget(self.plot)
        else:
            layout.addWidget(QtWidgets.QLabel("Plot widget unavailable", self))

        btns = QtWidgets.QHBoxLayout()
        self.btn_process = QtWidgets.QPushButton("Process All", self)
        self.btn_process.clicked.connect(self._process_all)
        btns.addStretch(1)
        btns.addWidget(self.btn_process)
        layout.addLayout(btns)

        self.status = QtWidgets.QTextEdit(self)
        self.status.setReadOnly(True)
        self.status.setMinimumHeight(100)
        layout.addWidget(self.status)

        self.setLayout(layout)

    def initializePage(self):
        # Populate combo with files
        self.cmb_file.clear()
        for p in self.files_page.get_files():
            self.cmb_file.addItem(p.name, str(p))
        if self.cmb_file.count() > 0:
            self.cmb_file.setCurrentIndex(0)
            self._on_select_file(0)

    def _on_select_file(self, _idx: int):
        if self.plot is None:
            return
        path = self.cmb_file.currentData()
        if not path:
            return
        tw_ms, _out = self.setup_page.get_values()
        tw_s = float(tw_ms) / 1000.0
        try:
            tttr = tttrlib.TTTR(str(path))
            counts = tttr.get_intensity_trace(tw_s)
            if counts is None:
                return
            time_axis = np.arange(len(counts), dtype=float) * tw_s
            traces = np.asarray(counts, dtype=float).reshape(-1, 1)
            # Visualize with vertical lines at bin boundaries
            self.plot.plot_trace_and_histogram(time_axis, traces, channel_labels=["All"],
                                               bin_count=60, time_window_ms=tw_ms,
                                               show_window_lines=True)
        except Exception as exc:
            logging.error(f"Failed to preview {path}: {exc}")

    def _process_all(self):
        files = self.files_page.get_files()
        if not files:
            return
        tw_ms, out_dir = self.setup_page.get_values()
        if out_dir is None:
            # Ask for directory if not provided
            d, _ = get_directory(caption="Select output folder")
            if d is None:
                return
            out_dir = d
        out_dir.mkdir(parents=True, exist_ok=True)

        tw_s = float(tw_ms) / 1000.0
        ok = 0
        for p in files:
            try:
                tttr = tttrlib.TTTR(str(p))
                bids = compute_bids_from_tttr(tttr, tw_s)
                if bids.size == 0:
                    self._log(f"{p.name}: no data")
                    continue
                out = out_dir / f"{p.stem}.bid"
                np.savetxt(str(out), bids.astype(np.int64), fmt="%d\t%d")
                self._log(f"Saved {out.name} ({len(bids)} windows)")
                ok += 1
            except Exception as exc:
                self._log(f"Error processing {p.name}: {exc}")
        self._log(f"Done: {ok}/{len(files)} files processed. Output: {out_dir}")

    def _log(self, msg: str):
        try:
            logging.info(msg)
        except Exception:
            pass
        self.status.append(msg)
        self.status.ensureCursorVisible()


class TTTRTimeWindowWizard(QtWidgets.QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TTTR Time-Window Splitter (BIDs)")
        self.resize(1000, 700)

        self.page_setup = SetupPage(self)
        self.page_files = FilesPage(self)
        self.page_process = ProcessPage(self.page_setup, self.page_files, self)

        self.addPage(self.page_setup)
        self.addPage(self.page_files)
        self.addPage(self.page_process)


if __name__ == "plugin":
    app = QtWidgets.QApplication.instance()
    parent = None if app is None else app.activeWindow()
    w = TTTRTimeWindowWizard(parent)
    w.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = TTTRTimeWindowWizard()
    w.show()
    sys.exit(app.exec_())
