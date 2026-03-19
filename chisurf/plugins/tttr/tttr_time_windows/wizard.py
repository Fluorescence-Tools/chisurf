from __future__ import annotations
import pathlib
from typing import List, Tuple

import numpy as np

# Use Qt abstraction layer
from qtpy import QtWidgets, QtCore

# Reuse the intensity plot widget for visualization
try:
    from chisurf.plugins.tttr.intensity_trace.__init__ import IntensityPlotWidget
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
    from chisurf.plugins.tttr.trace_browser.__init__ import get_tttr_supported_exts
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


class TTTRTimeWindowTool(QtWidgets.QWidget):
    """Single-window TTTR→BID tool with preview.

    Consolidates setup, file selection, preview, and processing into one window.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TTTR Time-Window BIDs")
        try:
            self.resize(1000, 700)
        except Exception:
            pass

        # --- Top controls: time window + output folder ---
        top_form = QtWidgets.QGridLayout()

        self.tws_spin = QtWidgets.QDoubleSpinBox(self)
        self.tws_spin.setDecimals(3)
        self.tws_spin.setRange(0.001, 3600 * 1000.0)  # ms
        self.tws_spin.setSingleStep(1.0)
        self.tws_spin.setValue(10.0)
        self.tws_spin.setSuffix(" ms")
        self.tws_spin.valueChanged.connect(self._update_preview)

        self.output_edit = QtWidgets.QLineEdit(self)
        self.btn_browse = QtWidgets.QPushButton("Browse…", self)
        self.btn_browse.clicked.connect(self._choose_dir)

        top_form.addWidget(QtWidgets.QLabel("Time window:"), 0, 0)
        top_form.addWidget(self.tws_spin, 0, 1)
        top_form.addWidget(QtWidgets.QLabel("Output folder:"), 0, 2)
        top_form.addWidget(self.output_edit, 0, 3)
        top_form.addWidget(self.btn_browse, 0, 4)

        # --- Files & actions row ---
        files_layout = QtWidgets.QVBoxLayout()
        try:
            _exts_text = " ".join(sorted(_supported_exts()))
        except Exception:
            _exts_text = ".ptu .phu .ht2 .ht3 .pt3 .t3r"
        files_layout.addWidget(QtWidgets.QLabel(f"Drop TTTR files here ({_exts_text})", self))

        self.file_list = TTTRFileList(accept_drops=True, filename_ending='')
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.file_list.setMinimumHeight(160)
        files_layout.addWidget(self.file_list)

        btns = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add…", self)
        self.btn_clear = QtWidgets.QPushButton("Clear", self)
        btns.addWidget(self.btn_add)
        btns.addStretch(1)
        btns.addWidget(self.btn_clear)
        files_layout.addLayout(btns)

        self.btn_add.clicked.connect(self._on_add)
        self.btn_clear.clicked.connect(self.file_list.clear)

        # --- Preview selector + plot ---
        preview_row = QtWidgets.QHBoxLayout()
        preview_row.addWidget(QtWidgets.QLabel("Preview file:", self))
        self.cmb_file = QtWidgets.QComboBox(self)
        self.cmb_file.currentIndexChanged.connect(self._on_select_file)
        preview_row.addWidget(self.cmb_file, 1)

        self.plot = IntensityPlotWidget(self) if IntensityPlotWidget is not None else None

        # --- Bottom actions + status ---
        action_row = QtWidgets.QHBoxLayout()
        self.btn_process = QtWidgets.QPushButton("Process All", self)
        self.btn_process.clicked.connect(self._process_all)
        action_row.addStretch(1)
        action_row.addWidget(self.btn_process)

        self.status = QtWidgets.QTextEdit(self)
        self.status.setReadOnly(True)
        self.status.setMinimumHeight(100)

        # --- Compose main layout ---
        main = QtWidgets.QVBoxLayout(self)
        main.addLayout(top_form)
        main.addLayout(files_layout)
        main.addLayout(preview_row)
        if self.plot is not None:
            main.addWidget(self.plot)
        else:
            main.addWidget(QtWidgets.QLabel("Plot widget unavailable", self))
        main.addLayout(action_row)
        main.addWidget(self.status)

        # Initialize preview combo when list changes
        try:
            self.file_list.model().rowsInserted.connect(self._refresh_combo)
            self.file_list.model().rowsRemoved.connect(self._refresh_combo)
            self.file_list.model().modelReset.connect(self._refresh_combo)
        except Exception:
            pass

    # --- Helpers ---
    def _choose_dir(self):
        d, _ = get_directory(caption="Select output folder")
        if d is not None:
            self.output_edit.setText(str(d))

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
        self._refresh_combo()

    def _refresh_combo(self):
        try:
            self.cmb_file.blockSignals(True)
            self.cmb_file.clear()
            for s in getattr(self.file_list, 'filenames', []):
                p = pathlib.Path(s)
                self.cmb_file.addItem(p.name, str(p))
        finally:
            try:
                self.cmb_file.blockSignals(False)
            except Exception:
                pass
        if self.cmb_file.count() > 0:
            self.cmb_file.setCurrentIndex(0)
            self._on_select_file(0)

    def _update_preview(self):
        idx = self.cmb_file.currentIndex()
        self._on_select_file(idx)

    def _on_select_file(self, _idx: int):
        if self.plot is None:
            return
        path = self.cmb_file.currentData()
        if not path:
            return
        tw_ms = float(self.tws_spin.value())
        tw_s = tw_ms / 1000.0
        try:
            tttr = tttrlib.TTTR(str(path))
            counts = tttr.get_intensity_trace(tw_s)
            if counts is None:
                return
            time_axis = np.arange(len(counts), dtype=float) * tw_s
            traces = np.asarray(counts, dtype=float).reshape(-1, 1)
            self.plot.plot_trace_and_histogram(time_axis, traces, channel_labels=["All"],
                                               bin_count=60, time_window_ms=tw_ms,
                                               show_window_lines=True)
        except Exception as exc:
            logging.error(f"Failed to preview {path}: {exc}")

    def _process_all(self):
        files = [pathlib.Path(s) for s in getattr(self.file_list, 'filenames', [])]
        if not files:
            return

        tw_ms = float(self.tws_spin.value())
        out_dir_txt = self.output_edit.text().strip()
        out_dir = pathlib.Path(out_dir_txt) if out_dir_txt else None

        if out_dir is None:
            if len(files) == 1:
                single = files[0]
                folder_name = f"{single.stem}_TW_{tw_ms:.0f}ms"
                out_dir = single.parent / folder_name
            else:
                suggested_name = f"analysis_TW_{tw_ms:.0f}ms"
                d, _ = get_directory(caption="Select output folder", suggestion=suggested_name)
                if d is None:
                    return
                out_dir = d

        # Create the main output directory and a 'bst' subdirectory
        bst_dir = out_dir / 'bst'
        try:
            bst_dir.mkdir(parents=True, exist_ok=True)
            self.output_edit.setText(str(out_dir))
        except Exception as exc:
            self._log(f"Error creating directory {bst_dir}: {exc}")
            return

        tw_s = tw_ms / 1000.0
        ok = 0
        for p in files:
            try:
                tttr = tttrlib.TTTR(str(p))
                bids = compute_bids_from_tttr(tttr, tw_s)
                if bids.size == 0:
                    self._log(f"{p.name}: no data")
                    continue
                # Save as .bst file in the 'bst' subdirectory
                out = bst_dir / f"{p.stem}.bst"
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
        try:
            self.status.append(msg)
            self.status.ensureCursorVisible()
        except Exception:
            pass


# Backward compatibility: keep the old class name pointing to the new single-window tool
TTTRTimeWindowWizard = TTTRTimeWindowTool

if __name__ == "plugin":
    w = TTTRTimeWindowTool()
    w.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = TTTRTimeWindowTool()
    w.show()
    sys.exit(app.exec_())
