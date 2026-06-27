"""Dockable panel for computing TTTR microtime LUTs."""

from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from ..core import (
    build_linearization_table,
    histogram_micro,
    infer_n_bins,
    load_microtimes,
    save_lut,
    stochastic_rebin_ntac,
)


class TACLinearizationPanel(QtWidgets.QWidget):
    """Interactive panel for Felekyan-style TAC LUT computation."""

    def __init__(self) -> None:
        """Create the panel and its controls."""
        super().__init__()

        self.setWindowTitle("TAC Linearization")
        self.resize(1200, 800)

        self.sp_seed = QtWidgets.QSpinBox()
        self.sp_seed.setRange(0, 2**31 - 1)
        self.sp_seed.setValue(12345)

        self.chk_wrap = QtWidgets.QCheckBox("Mitigate wrap spike (floor + ε)")
        self.chk_wrap.setChecked(False)

        self.sp_eps = QtWidgets.QDoubleSpinBox()
        self.sp_eps.setDecimals(9)
        self.sp_eps.setSingleStep(1e-7)
        self.sp_eps.setRange(0.0, 1e-2)
        self.sp_eps.setValue(1e-6)

        self.sp_thresh = QtWidgets.QDoubleSpinBox()
        self.sp_thresh.setDecimals(6)
        self.sp_thresh.setSingleStep(0.01)
        self.sp_thresh.setRange(0.0, 1e12)
        self.sp_thresh.setValue(0.2)

        self.chk_brush = QtWidgets.QCheckBox("Enable zero brush")
        self.chk_brush.setChecked(False)
        self.sp_brush = QtWidgets.QSpinBox()
        self.sp_brush.setRange(1, 500)
        self.sp_brush.setValue(5)
        self.chk_erase = QtWidgets.QCheckBox("Eraser mode")

        self.chk_zl = QtWidgets.QCheckBox("Zero left")
        self.sp_zl = QtWidgets.QSpinBox()
        self.sp_zl.setRange(0, 100000)
        self.sp_zl.setValue(0)
        self.chk_zr = QtWidgets.QCheckBox("Zero right")
        self.sp_zr = QtWidgets.QSpinBox()
        self.sp_zr.setRange(0, 100000)
        self.sp_zr.setValue(0)

        self.lst_zero = QtWidgets.QListWidget()
        self.lst_zero.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.btn_zero_clear = QtWidgets.QPushButton("Clear all zeroed")

        self._create_ui()
        self._setup_connections()
        self.sp_eps.setEnabled(self.chk_wrap.isChecked())
        self._init_plots()

    class AdvancedDialog(QtWidgets.QDialog):
        """Dialog for advanced LUT parameters."""

        def __init__(self, parent: TACLinearizationPanel | None = None) -> None:
            """Create the advanced parameter dialog."""
            super().__init__(parent)
            self.setWindowTitle("Advanced Parameters")
            self.setModal(True)
            self.resize(400, 600)

            layout = QtWidgets.QVBoxLayout(self)

            seed_group = QtWidgets.QGroupBox("Random Number Generation")
            seed_layout = QtWidgets.QFormLayout(seed_group)
            seed_layout.addRow("RNG seed", parent.sp_seed)
            layout.addWidget(seed_group)

            spike_group = QtWidgets.QGroupBox("Spike Mitigation")
            spike_layout = QtWidgets.QVBoxLayout(spike_group)
            spike_layout.addWidget(parent.chk_wrap)
            eps_layout = QtWidgets.QFormLayout()
            eps_layout.addRow("ε (wrap)", parent.sp_eps)
            spike_layout.addLayout(eps_layout)
            layout.addWidget(spike_group)

            thresh_group = QtWidgets.QGroupBox("Data Thresholding")
            thresh_layout = QtWidgets.QFormLayout(thresh_group)
            thresh_layout.addRow("Low-count threshold", parent.sp_thresh)
            layout.addWidget(thresh_group)

            zero_group = QtWidgets.QGroupBox("Zero-out bins (cleanup)")
            zero_layout = QtWidgets.QVBoxLayout(zero_group)

            brush_layout = QtWidgets.QHBoxLayout()
            brush_layout.addWidget(parent.chk_brush)
            brush_layout.addWidget(QtWidgets.QLabel("Radius"))
            brush_layout.addWidget(parent.sp_brush)
            brush_layout.addStretch()
            brush_layout.addWidget(parent.chk_erase)

            cuts_layout = QtWidgets.QGridLayout()
            cuts_layout.addWidget(parent.chk_zl, 0, 0)
            cuts_layout.addWidget(parent.sp_zl, 0, 1)
            cuts_layout.addWidget(parent.chk_zr, 1, 0)
            cuts_layout.addWidget(parent.sp_zr, 1, 1)

            zero_layout.addLayout(brush_layout)
            zero_layout.addLayout(cuts_layout)
            zero_layout.addWidget(parent.lst_zero)
            zero_layout.addWidget(parent.btn_zero_clear)
            layout.addWidget(zero_group)

            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

    def _show_advanced_dialog(self) -> None:
        """Show advanced parameters and refresh the plots."""
        dialog = self.AdvancedDialog(self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self._apply_params()
            self._update_plots()

    def _create_ui(self) -> None:
        """Create the main panel layout."""
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        plots_widget = QtWidgets.QWidget()
        plots_layout = QtWidgets.QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)

        self.plt_raw = pg.PlotWidget()
        self.plt_raw.setTitle("Raw TAC histogram (drag the orange region)")
        self.plt_raw.setLabel("left", "Counts")
        self.plt_raw.setLabel("bottom", "TAC bin")

        self.plt_after = pg.PlotWidget()
        self.plt_after.setTitle("After linearization (actual corrected preview)")
        self.plt_after.setLabel("left", "Counts")
        self.plt_after.setLabel("bottom", "Equal-width bin within one SYNC")

        plots_layout.addWidget(self.plt_raw, 1)
        plots_layout.addWidget(self.plt_after, 1)

        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        files_group = QtWidgets.QGroupBox("Files")
        files_layout = QtWidgets.QVBoxLayout(files_group)

        self.files_list = QtWidgets.QListWidget()
        self.files_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.files_list.setAcceptDrops(True)
        self.files_list.installEventFilter(self)

        btn_load = QtWidgets.QPushButton("📂 Load TTTR Files...")
        btn_load.clicked.connect(self._load_files)

        btn_clear = QtWidgets.QPushButton("🧹 Clear list")
        btn_clear.clicked.connect(self._clear_files)

        files_layout.addWidget(self.files_list)
        file_buttons = QtWidgets.QHBoxLayout()
        file_buttons.addWidget(btn_load)
        file_buttons.addWidget(btn_clear)
        file_buttons.addStretch(1)
        files_layout.addLayout(file_buttons)

        params_group = QtWidgets.QGroupBox("Parameters")
        params_layout = QtWidgets.QFormLayout(params_group)

        self.lbl_nbins = QtWidgets.QLabel("N/A")
        params_layout.addRow("n_bins (hist)", self.lbl_nbins)

        self.sp_start = QtWidgets.QSpinBox()
        self.sp_start.setRange(0, 100000)
        self.sp_start.setValue(1000)
        params_layout.addRow("linear_start", self.sp_start)

        self.sp_stop = QtWidgets.QSpinBox()
        self.sp_stop.setRange(1, 100000)
        self.sp_stop.setValue(2000)
        params_layout.addRow("linear_stop", self.sp_stop)

        self.sp_ntac = QtWidgets.QSpinBox()
        self.sp_ntac.setRange(2, 1000000)
        self.sp_ntac.setValue(4096)
        params_layout.addRow("ntac_required", self.sp_ntac)

        self.sp_noff = QtWidgets.QSpinBox()
        self.sp_noff.setRange(0, 100000)
        self.sp_noff.setValue(500)
        params_layout.addRow("Noffset", self.sp_noff)

        self.sp_prev = QtWidgets.QSpinBox()
        self.sp_prev.setRange(1000, 10000000)
        self.sp_prev.setValue(500000)
        params_layout.addRow("preview photons", self.sp_prev)

        self.chk_norm = QtWidgets.QCheckBox("Normalize raw TAC by region mean")
        self.chk_norm.setChecked(False)
        params_layout.addRow("", self.chk_norm)

        self.btn_advanced = QtWidgets.QPushButton("⚙️ Advanced Parameters...")
        self.btn_advanced.clicked.connect(self._show_advanced_dialog)
        params_layout.addRow("", self.btn_advanced)

        buttons_layout = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("💾 Save LUT")
        self.btn_apply = QtWidgets.QPushButton("🔧 Apply params")
        self.btn_export = QtWidgets.QPushButton("📤 Export corrected…")

        buttons_layout.addWidget(self.btn_save)
        buttons_layout.addWidget(self.btn_apply)
        buttons_layout.addWidget(self.btn_export)
        buttons_layout.addStretch(1)

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)

        controls_layout.addWidget(files_group)
        controls_layout.addWidget(params_group)
        controls_layout.addLayout(buttons_layout)
        controls_layout.addWidget(self.info_label)
        controls_layout.addStretch()

        controls_widget.setMinimumWidth(260)
        controls_widget.setMaximumWidth(420)

        self.controls_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.controls_splitter.addWidget(plots_widget)
        self.controls_splitter.addWidget(controls_widget)
        self.controls_splitter.setStretchFactor(0, 1)
        self.controls_splitter.setStretchFactor(1, 0)
        self.controls_splitter.setCollapsible(1, False)
        self.controls_splitter.setSizes([800, 320])

        main_layout.addWidget(self.controls_splitter, 1)

    def _setup_connections(self) -> None:
        """Connect panel controls to update handlers."""
        self.sp_start.valueChanged.connect(self._on_params_changed)
        self.sp_stop.valueChanged.connect(self._on_params_changed)
        self.sp_ntac.valueChanged.connect(self._on_params_changed)
        self.sp_noff.valueChanged.connect(self._on_params_changed)
        self.sp_seed.valueChanged.connect(self._on_params_changed)
        self.sp_prev.valueChanged.connect(self._on_params_changed)
        self.chk_norm.stateChanged.connect(self._on_params_changed)
        self.chk_wrap.stateChanged.connect(self._on_wrap_changed)
        self.sp_eps.valueChanged.connect(self._on_params_changed)
        self.sp_thresh.valueChanged.connect(self._on_params_changed)

        self.chk_zl.stateChanged.connect(self._on_zero_controls_changed)
        self.chk_zr.stateChanged.connect(self._on_zero_controls_changed)
        self.sp_zl.valueChanged.connect(self._on_zero_controls_changed)
        self.sp_zr.valueChanged.connect(self._on_zero_controls_changed)
        self.btn_zero_clear.clicked.connect(self._clear_zeroed)

        self.btn_apply.clicked.connect(self._apply_params)
        self.btn_save.clicked.connect(self._save_lut)
        self.btn_export.clicked.connect(self._export_corrected)
        self.chk_brush.stateChanged.connect(self._update_brush_state)

    def _init_plots(self) -> None:
        """Initialize plot items."""
        self.region = pg.LinearRegionItem(
            values=[1000, 2000],
            brush=(255, 165, 0, 60),
            movable=True,
        )
        self.plt_raw.addItem(self.region)

        self.offset_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pos=500,
            pen=pg.mkPen((200, 0, 0), width=2),
        )
        self.plt_raw.addItem(self.offset_line)

        self.thresh_line = pg.InfiniteLine(
            angle=0,
            movable=True,
            pos=0.2,
            pen=pg.mkPen((0, 180, 0), width=2),
        )
        self.plt_raw.addItem(self.thresh_line)

        self.left_cut_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pos=0,
            pen=pg.mkPen((120, 120, 120), width=2, style=QtCore.Qt.DashLine),
        )
        self.right_cut_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pos=0,
            pen=pg.mkPen((120, 120, 120), width=2, style=QtCore.Qt.DashLine),
        )

        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.offset_line.sigPositionChanged.connect(self._on_offset_line_changed)
        self.thresh_line.sigPositionChanged.connect(self._on_thresh_line_changed)
        self.left_cut_line.sigPositionChanged.connect(self._on_left_cut_changed)
        self.right_cut_line.sigPositionChanged.connect(self._on_right_cut_changed)

        self.plt_raw.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.plt_raw.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._brush_active = False

    def _load_files(self) -> None:
        """Open a file dialog and load selected TTTR files."""
        dialog = QtWidgets.QFileDialog(self, "Load TTTR Files")
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dialog.setNameFilter("TTTR files (*.spc *.ht3 *.ptu *.t3r *.t2r);;All files (*.*)")

        if dialog.exec():
            self._process_files(dialog.selectedFiles())

    def _clear_files(self) -> None:
        """Clear loaded files and reset plots."""
        self.files_list.clear()
        self.files = []
        self.counts = None
        self.micro = None
        self.n_bins = None
        self.zero_mask = None
        self.current_table = None

        self.lbl_nbins.setText("N/A")
        self.sp_start.setRange(0, 100000)
        self.sp_stop.setRange(1, 100000)
        self.sp_prev.setRange(1000, 10000000)
        self.sp_start.setValue(1000)
        self.sp_stop.setValue(2000)
        self.sp_ntac.setValue(4096)
        self.sp_noff.setValue(500)
        self.sp_prev.setValue(500000)

        self.plt_raw.clear()
        self.plt_after.clear()
        self.plt_raw.setTitle("Raw TAC histogram (drag the orange region)")
        self.plt_raw.setLabel("left", "Counts")
        self.plt_raw.setLabel("bottom", "TAC bin")
        self.plt_after.setTitle("After linearization (actual corrected preview)")
        self.plt_after.setLabel("left", "Counts")
        self.plt_after.setLabel("bottom", "Equal-width bin within one SYNC")
        self.info_label.setText("")

        self.region.setRegion([1000, 2000])
        self.plt_raw.addItem(self.region)
        self.offset_line.setValue(500)
        self.plt_raw.addItem(self.offset_line)
        self.thresh_line.setValue(0.2)
        self.plt_raw.addItem(self.thresh_line)

    def _process_files(self, files: list[str]) -> None:
        """Load and process TTTR files."""
        try:
            self.files = files
            self.files_list.clear()
            self.files_list.addItems([os.path.basename(filename) for filename in files])

            self.micro = load_microtimes(files)
            self.n_bins = infer_n_bins(self.micro, None)
            self.counts = histogram_micro(self.micro, self.n_bins)
            self.zero_mask = np.zeros(self.n_bins, dtype=bool)

            self.lbl_nbins.setText(str(self.n_bins))
            self.sp_start.setRange(0, self.n_bins - 2)
            self.sp_stop.setRange(1, self.n_bins - 1)
            self.sp_prev.setRange(1000, max(1000, len(self.micro)))

            region = self._pick_initial_region()
            self.sp_start.setValue(region[0])
            self.sp_stop.setValue(region[1])
            self.region.setRegion(region)

            self._update_plots()
            self._apply_params()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load files: {exc}")

    def _pick_initial_region(self) -> tuple[int, int]:
        """Choose an initial linearization region."""
        if self.counts is None:
            return 1000, 2000

        n_bins = len(self.counts)
        nonzero = np.where(self.counts > 0)[0]
        if nonzero.size < 4:
            start = max(0, n_bins // 4)
            stop = min(n_bins, start + max(32, n_bins // 10))
            return start, stop

        first = nonzero[0]
        last = nonzero[-1] + 1
        width = max(32, (last - first) // 5)
        start = first + (last - first - width) // 2
        return start, start + width

    def _counts_effective(self) -> np.ndarray | None:
        """Return effective counts after zeroing and thresholding."""
        if self.counts is None:
            return None

        counts = self.counts.copy()
        if self.zero_mask is not None:
            counts[self.zero_mask] = 0

        if self.chk_zl.isChecked():
            counts[: self.sp_zl.value()] = 0

        if self.chk_zr.isChecked():
            counts[max(0, self.sp_zr.value()) :] = 0

        start = self.sp_start.value()
        stop = self.sp_stop.value()
        if self.chk_norm.isChecked():
            region = counts[start:stop]
            mean = float(region.mean()) if region.size > 0 else 1.0
            scale = mean if mean > 0 else 1.0
            display = counts / scale
        else:
            display = counts.astype(float)

        threshold = self.sp_thresh.value()
        counts[display < threshold] = 0
        return counts

    def _update_plots(self) -> None:
        """Update the raw TAC plot."""
        if self.counts is None:
            return

        self.plt_raw.clear()
        counts = self._counts_effective()
        if counts is None:
            return

        start = self.sp_start.value()
        stop = self.sp_stop.value()
        if self.chk_norm.isChecked():
            region = counts[start:stop]
            mean = float(region.mean()) if region.size > 0 else 1.0
            scale = mean if mean > 0 else 1.0
            y_display = counts / scale
            self.plt_raw.setLabel("left", "Counts / ⟨counts⟩_region")
            one_line = pg.InfiniteLine(
                angle=0,
                movable=False,
                pos=1.0,
                pen=pg.mkPen((150, 150, 150), width=1, style=QtCore.Qt.DashLine),
            )
            self.plt_raw.addItem(one_line)
        else:
            y_display = counts.astype(float)
            self.plt_raw.setLabel("left", "Counts")

        x_values = np.arange(self.n_bins)
        self.plt_raw.plot(x_values, y_display, pen=pg.mkPen((255, 204, 0), width=1.5))
        self.plt_raw.addItem(self.region)

        if self.zero_mask is not None:
            for start_idx, stop_idx in self._mask_to_ranges(self.zero_mask):
                zero_region = pg.LinearRegionItem(
                    values=[start_idx, stop_idx],
                    brush=(255, 0, 0, 60),
                    movable=False,
                )
                zero_region.setZValue(5)
                self.plt_raw.addItem(zero_region)

        threshold = self.sp_thresh.value()
        below = y_display < threshold
        for start_idx, stop_idx in self._mask_to_ranges(below):
            zero_region = pg.LinearRegionItem(
                values=[start_idx, stop_idx],
                brush=(0, 0, 255, 40),
                movable=False,
            )
            zero_region.setZValue(4)
            self.plt_raw.addItem(zero_region)

        if self.chk_zl.isChecked():
            left = self.sp_zl.value()
            self.left_cut_line.setValue(left)
            self.plt_raw.addItem(self.left_cut_line)
            if left > 0:
                self.plt_raw.addItem(
                    pg.LinearRegionItem(
                        values=[0, left],
                        brush=(120, 120, 120, 40),
                        movable=False,
                    )
                )

        if self.chk_zr.isChecked():
            right = self.sp_zr.value()
            self.right_cut_line.setValue(right)
            self.plt_raw.addItem(self.right_cut_line)
            if right < self.n_bins - 1:
                self.plt_raw.addItem(
                    pg.LinearRegionItem(
                        values=[right, self.n_bins],
                        brush=(120, 120, 120, 40),
                        movable=False,
                    )
                )

        self.offset_line.setValue(self.sp_noff.value())
        self.plt_raw.addItem(self.offset_line)
        self.thresh_line.setValue(threshold)
        self.plt_raw.addItem(self.thresh_line)
        self._update_zero_list()

    def _mask_to_ranges(self, mask: np.ndarray) -> list[tuple[int, int]]:
        """Convert a boolean mask into half-open ranges."""
        ranges: list[tuple[int, int]] = []
        index = 0
        size = len(mask)
        while index < size:
            if mask[index]:
                stop = index + 1
                while stop < size and mask[stop]:
                    stop += 1
                ranges.append((index, stop))
                index = stop
            else:
                index += 1
        return ranges

    def _update_zero_list(self) -> None:
        """Update the list of zeroed ranges."""
        self.lst_zero.clear()
        if self.zero_mask is not None:
            for start, stop in self._mask_to_ranges(self.zero_mask):
                self.lst_zero.addItem(f"[{start}, {stop})")

    def _apply_params(self) -> None:
        """Compute the current LUT."""
        if self.counts is None:
            return

        try:
            counts = self._counts_effective()
            start = self.sp_start.value()
            stop = self.sp_stop.value()
            self.current_table = build_linearization_table(
                counts,
                start,
                stop,
                self.sp_ntac.value(),
                self.sp_noff.value(),
            )

            self.info_label.setText(
                f"Range [{self.current_table['linear_start']}, {self.current_table['linear_stop']}) | "
                f"width={self.current_table['linear_stop'] - self.current_table['linear_start']} | "
                f"f={self.current_table['f']:.6f} | n_mean={self.current_table['n_mean']:.2f}"
            )
            self._update_after_plot()
        except Exception as exc:
            self.info_label.setText(f"Error: {exc}")
            self.current_table = None

    def _update_after_plot(self) -> None:
        """Update the corrected-preview histogram."""
        if self.current_table is None:
            return

        self.plt_after.clear()
        corrected = stochastic_rebin_ntac(
            self.micro,
            self.current_table["NTAC_fract"],
            self.current_table["noffset"],
            seed=self.sp_seed.value(),
            max_photons=self.sp_prev.value(),
            rounding="floor" if self.chk_wrap.isChecked() else "ceil",
            eps=self.sp_eps.value() if self.chk_wrap.isChecked() else 0.0,
        )

        ntac = self.sp_ntac.value()
        nt_mod = (corrected % ntac).astype(int)
        hist_corr, _ = np.histogram(nt_mod, bins=ntac, range=(0, ntac))
        self.plt_after.plot(
            np.arange(ntac),
            hist_corr,
            pen=pg.mkPen((80, 200, 255), width=1.5),
        )
        self.plt_after.setXRange(0, ntac, padding=0)

    def _save_lut(self) -> None:
        """Save the current LUT table."""
        if self.current_table is None:
            QtWidgets.QMessageBox.warning(self, "No LUT", "Compute a LUT first.")
            return

        dialog = QtWidgets.QFileDialog(self, "Save LUT")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setNameFilters(
            [
                "Text (*.txt)",
                "CSV (*.csv)",
                "NumPy binary (*.npy)",
                "Compressed NPZ (*.npz)",
            ]
        )

        if dialog.exec():
            path = dialog.selectedFiles()[0]
            try:
                save_lut(path, self.current_table)
                QtWidgets.QMessageBox.information(self, "Saved", f"LUT saved to {path}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))

    def _export_corrected(self) -> None:
        """Export corrected microtimes."""
        if self.current_table is None:
            QtWidgets.QMessageBox.warning(self, "No LUT", "Compute a LUT first.")
            return

        dialog = QtWidgets.QFileDialog(self, "Export Corrected Microtimes")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setNameFilters(
            [
                "NumPy binary (*.npy)",
                "Compressed NPZ (*.npz)",
                "CSV (*.csv)",
                "Text (*.txt)",
            ]
        )

        if dialog.exec():
            path = dialog.selectedFiles()[0]
            try:
                corrected = stochastic_rebin_ntac(
                    self.micro,
                    self.current_table["NTAC_fract"],
                    self.current_table["noffset"],
                    seed=self.sp_seed.value(),
                    rounding="floor" if self.chk_wrap.isChecked() else "ceil",
                    eps=self.sp_eps.value() if self.chk_wrap.isChecked() else 0.0,
                )

                ext = os.path.splitext(path)[1].lower()
                if ext == ".npy":
                    np.save(path, corrected)
                elif ext == ".npz":
                    np.savez_compressed(path, corrected_ntac=corrected)
                elif ext == ".csv":
                    np.savetxt(path, corrected, fmt="%d", delimiter=",")
                elif ext == ".txt":
                    np.savetxt(path, corrected, fmt="%d")
                else:
                    np.save(path, corrected)

                QtWidgets.QMessageBox.information(self, "Exported", f"Corrected microtimes saved to {path}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))

    def _on_params_changed(self) -> None:
        """Refresh plots after parameter changes."""
        self._update_plots()
        self._apply_params()

    def _on_wrap_changed(self) -> None:
        """Toggle wrap-spike epsilon controls."""
        self.sp_eps.setEnabled(self.chk_wrap.isChecked())
        self._apply_params()

    def _on_region_changed(self) -> None:
        """Sync spinboxes with the interactive region."""
        start, stop = [int(value) for value in self.region.getRegion()]
        self.sp_start.blockSignals(True)
        self.sp_stop.blockSignals(True)
        self.sp_start.setValue(start)
        self.sp_stop.setValue(stop)
        self.sp_start.blockSignals(False)
        self.sp_stop.blockSignals(False)
        self._on_params_changed()

    def _on_offset_line_changed(self) -> None:
        """Sync Noffset with the draggable offset line."""
        value = int(round(self.offset_line.value()))
        value = max(0, min(self.n_bins - 1, value))
        if self.sp_noff.value() != value:
            self.sp_noff.setValue(value)

    def _on_thresh_line_changed(self) -> None:
        """Sync threshold with the draggable threshold line."""
        value = self.thresh_line.value()
        if self.sp_thresh.value() != value:
            self.sp_thresh.setValue(value)

    def _on_left_cut_changed(self) -> None:
        """Sync left zero cut with its draggable line."""
        value = int(round(self.left_cut_line.value()))
        value = max(0, min(self.n_bins - 1, value))
        if self.sp_zl.value() != value:
            self.sp_zl.setValue(value)

    def _on_right_cut_changed(self) -> None:
        """Sync right zero cut with its draggable line."""
        value = int(round(self.right_cut_line.value()))
        value = max(0, min(self.n_bins - 1, value))
        if self.sp_zr.value() != value:
            self.sp_zr.setValue(value)

    def _on_zero_controls_changed(self) -> None:
        """Refresh plots after zero-control changes."""
        self._update_plots()
        self._apply_params()

    def _clear_zeroed(self) -> None:
        """Clear manually zeroed bins."""
        if self.zero_mask is not None:
            self.zero_mask[:] = False
        self._update_plots()
        self._apply_params()

    def _update_brush_state(self) -> None:
        """Update brush state."""

    def _on_mouse_clicked(self, event: object) -> None:
        """Handle raw-plot mouse clicks."""

    def _on_mouse_moved(self, pos: object) -> None:
        """Handle raw-plot mouse movement."""

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Handle drag and drop for the file list."""
        if obj is self.files_list:
            if event.type() == QtCore.QEvent.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            if event.type() == QtCore.QEvent.Drop:
                if event.mimeData().hasUrls():
                    paths = []
                    for url in event.mimeData().urls():
                        path = str(url.toLocalFile())
                        if path:
                            paths.append(path)
                    if paths:
                        self._process_files(paths)
                    event.acceptProposedAction()
                    return True
        return super().eventFilter(obj, event)
