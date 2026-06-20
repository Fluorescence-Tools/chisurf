"""Main widget for the Jordi G-Factor Calculator plugin.

This acts as a thin client for the Jordi G-Factor backend.
"""

from __future__ import annotations

import logging
import csv
import sys
import warnings
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

from qtpy import uic
from qtpy.QtWidgets import (
    QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QAbstractItemView,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QDialog
)
from qtpy.QtCore import Qt
import pyqtgraph as pg

# Optional ChiSurf I/O import for Jordi reading
try:
    from chisurf.core.fio import read_jordi as _read_jordi
except Exception:
    _read_jordi = None

from .client import JordiGFactorClient
from ..core.calculations import (
    shift_interp_on_axis,
    compute_rt,
    compute_background_levels,
    perrin_steady_state_anisotropy,
    estimate_lifetime_first_moment,
    solve_linked_l_from_steady_state,
)

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c


try:
    _DASH_LINE_STYLE = Qt.PenStyle.DashLine
except Exception:
    _DASH_LINE_STYLE = getattr(Qt, "DashLine", 2)


class DataCurve:
    def __init__(self, x=None, y=None, name=None):
        """Initialize a DataCurve with x, y, and name."""
        self.x = x
        self.y = y
        self.name = name


class FileDropList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def add_files(self, paths):
        existing = {self.item(i).text() for i in range(self.count())}
        for p in paths:
            if p and p not in existing:
                self.addItem(QListWidgetItem(p))
                existing.add(p)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = []
            for url in event.mimeData().urls():
                local = url.toLocalFile()
                if local:
                    paths.append(local)
            self.add_files(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class JordiDecayBatchWindow(QDialog):
    def __init__(self, settings_snapshot: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Jordi G-Factor Batch Decays")
        self.setWindowModality(Qt.ApplicationModal)
        self.snapshot = dict(settings_snapshot)
        self.results = []
        self._client = JordiGFactorClient()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Drop Jordi files here or use Add..."))

        self.file_list = FileDropList()
        layout.addWidget(self.file_list)

        buttons = QHBoxLayout()
        self.add_btn = QPushButton("Add...")
        self.add_btn.clicked.connect(self._on_add)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._on_remove)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear)
        self.run_btn = QPushButton("Run Batch")
        self.run_btn.clicked.connect(self._on_run)
        self.save_btn = QPushButton("Save CSV...")
        self.save_btn.clicked.connect(self._on_save)
        buttons.addWidget(self.add_btn)
        buttons.addWidget(self.remove_btn)
        buttons.addWidget(self.clear_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.save_btn)
        layout.addLayout(buttons)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "filename", "r_inf", "region_min", "region_max", "bg_vv", "bg_vh", "g_factor"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        self.setLayout(layout)
        self.resize(1000, 640)

    def _compute_file_result(self, file_path: str):
        try:
            if _read_jordi is not None:
                vv, vh = _read_jordi(file_path, split=True)
            else:
                vec = np.loadtxt(file_path)
                half = len(vec) // 2
                vv, vh = vec[:half], vec[half:]
            vv = np.asarray(vv, dtype=float)
            vh = np.asarray(vh, dtype=float)
            n = min(len(vv), len(vh))
            vv = vv[:n]
            vh = vh[:n]
            t = np.arange(n, dtype=float)

            flip = bool(self.snapshot.get('flip', False))
            if flip:
                vv, vh = vh, vv

            g_raw = float(self.snapshot.get('g_raw', np.nan))
            g_corr = float(self.snapshot.get('g_corr', np.nan))
            l1 = float(self.snapshot.get('l1', 0.0))
            l2 = float(self.snapshot.get('l2', 0.0))
            shift = float(self.snapshot.get('shift', 0.0))
            bg_vv = float(self.snapshot.get('bg_vv', 0.0))
            bg_vh = float(self.snapshot.get('bg_vh', 0.0))
            apply_bg = bool(self.snapshot.get('apply_bg', False))
            region_min = float(self.snapshot.get('region_min', 0.0))
            region_max = float(self.snapshot.get('region_max', float(max(0, n - 1))))

            if not apply_bg:
                bg_vv = 0.0
                bg_vh = 0.0

            vv_corr = np.maximum(vv - bg_vv, 0.0)
            vh_corr = np.maximum(vh - bg_vh, 0.0)
            vh_corr_shifted = shift_interp_on_axis(t, vh_corr, shift)
            r_corr = compute_rt(vv_corr, vh_corr_shifted, g_corr, l1=l1, l2=l2)

            rmin = max(float(t[0]), min(region_min, float(t[-1])))
            rmax = max(float(t[0]), min(region_max, float(t[-1])))
            if rmax < rmin:
                rmin, rmax = rmax, rmin
            if rmax <= rmin:
                rmax = min(float(t[-1]), rmin + 1.0)
            i0 = int(np.argmin(np.abs(t - rmin)))
            i1 = int(np.argmin(np.abs(t - rmax)))
            if i1 <= i0:
                i1 = min(len(t), i0 + 1)
            r_region = r_corr[i0:i1]
            r_region = r_region[np.isfinite(r_region)]
            r_inf = float(np.nanmean(r_region)) if r_region.size > 0 else np.nan

            return (Path(file_path).name, r_inf, rmin, rmax, bg_vv, bg_vh, g_corr)
        except Exception:
            return (
                Path(file_path).name,
                np.nan,
                float(self.snapshot.get('region_min', 0.0)),
                float(self.snapshot.get('region_max', 0.0)),
                float(self.snapshot.get('bg_vv', 0.0)),
                float(self.snapshot.get('bg_vh', 0.0)),
                float(self.snapshot.get('g_corr', np.nan)),
            )

    def _append_result_row(self, row_tuple):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, value in enumerate(row_tuple):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(r, c, item)
        self.results.append(row_tuple)

    def _clear_results(self):
        self.table.setRowCount(0)
        self.results = []

    def _on_add(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add Jordi Files", "", "Data Files (*.dat *.txt *.csv);;All Files (*)")
        if files:
            self.file_list.add_files(files)

    def _on_remove(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def _on_clear(self):
        self.file_list.clear()
        self._clear_results()

    def _on_run(self):
        paths = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not paths:
            QMessageBox.information(self, "Batch", "No files to process.")
            return
        self._clear_results()
        for p in paths:
            self._append_result_row(self._compute_file_result(p))
        QMessageBox.information(self, "Batch", f"Processed {len(paths)} file(s).")

    def _on_save(self):
        if not self.results:
            QMessageBox.information(self, "Save CSV", "No results to save.")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not out_path:
            return
        try:
            with open(out_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "r_inf", "region_min", "region_max", "bg_vv", "bg_vh", "g_factor"])
                writer.writerows(self.results)
            QMessageBox.information(self, "Save CSV", f"Saved: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save CSV", f"Failed to save CSV: {e}")


@persist_plugin_state("jordi_g_factor")
class JordiGFactorCalculator(QWidget):
    """Main widget for the Jordi G-Factor Calculator plugin client."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jordi G-Factor Calculator")

        # Data storage
        self.jordi_data = None
        self.time_axis = None
        self.parallel_data = None
        self.perpendicular_data = None
        self.g_factor = None
        self.g_factor_stddev = None
        self.g_factor_uncorrected = None
        self.g_factor_corrected = None
        self.g_factor_manual_value = None
        self.manual_g_override = False
        self.l1_estimate = None
        self.l2_estimate = None
        self.fp_manual_l1_override = False
        self.fp_manual_tau_override = False
        self.fp_manual_rs_override = False
        self._updating_fp_l1_value = False
        self._updating_fp_tau_value = False
        self._updating_fp_rs_value = False
        self._updating_g_value = False
        self.fp_tau_estimate_ns = None
        self.fp_rs_expected = None
        self.fp_estimate_available = False
        self.fp_file_path = None
        self.fp_parallel_data = None
        self.fp_perpendicular_data = None
        self._batch_window = None
        self.region_bounds = [0, 100]
        self.bg_region_bounds = [0, 100]
        self.use_background_correction = False
        self.decay_shift = 0.0

        # ZMQ Client
        self._client = JordiGFactorClient()

        # Create UI
        self.init_ui()

    def init_ui(self):
        uic.loadUi(str(Path(__file__).parents[1] / "wizard.ui"), self)

        self.load_button.clicked.connect(self.load_jordi_file)
        self.batch_button.clicked.connect(self.open_batch_window)
        self.bg_correction_checkbox.stateChanged.connect(self.on_bg_correction_changed)
        self.flip_checkbox.stateChanged.connect(lambda *_: (self.update_plot(), self.calculate_g_factor()))
        self.shift_spinbox.valueChanged.connect(self.on_shift_changed)
        self.fp_load_button.clicked.connect(self.load_fp_jordi_file)
        self.fp_rho_spinbox.valueChanged.connect(self.calculate_fp_mixing_estimate)
        self.fp_r0_spinbox.valueChanged.connect(self.calculate_fp_mixing_estimate)
        self.fp_dt_spinbox.valueChanged.connect(self.calculate_fp_mixing_estimate)
        self.fp_tau_value.valueChanged.connect(self.on_fp_tau_value_changed)
        self.fp_rs_value.valueChanged.connect(self.on_fp_rs_value_changed)
        self.fp_l1_value.valueChanged.connect(self.on_fp_l1_value_changed)
        self.g_factor_value.textChanged.connect(self.on_g_factor_text_changed)
        self.g_factor_value.editingFinished.connect(self.on_g_factor_value_changed)
        self.show_fast_checkbox.stateChanged.connect(self.on_plot_visibility_changed)
        self.show_slow_checkbox.stateChanged.connect(self.on_plot_visibility_changed)
        self.show_raw_checkbox.stateChanged.connect(self.on_plot_visibility_changed)
        self.show_corrected_checkbox.stateChanged.connect(self.on_plot_visibility_changed)

        self.shift_spinbox.setValue(self.decay_shift)
        self.fp_rho_spinbox.setValue(16.0)
        self.fp_r0_spinbox.setValue(0.38)
        self.fp_dt_spinbox.setValue(1.0)
        self._updating_fp_l1_value = True
        self.fp_l1_value.setValue(0.0)
        self._updating_fp_l1_value = False

        self.corrected_g_factor_label.setVisible(False)
        self.corrected_g_factor_value.setVisible(False)
        self.corrected_g_factor_stddev_label.setVisible(False)
        self.corrected_g_factor_stddev_value.setVisible(False)
        self.bg_parallel_label.setVisible(False)
        self.bg_parallel_value.setVisible(False)
        self.bg_perpendicular_label.setVisible(False)
        self.bg_perpendicular_value.setVisible(False)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Intensity')
        self.plot_widget.setLabel('bottom', 'Channel')
        self.plot_widget.setTitle('Full Decay Curves')
        self.plot_widget.addLegend()
        self.plot_widget.setLogMode(x=False, y=True)

        self.region = pg.LinearRegionItem(
            values=self.region_bounds,
            brush=pg.mkBrush(color=(50, 50, 200, 50)),
            movable=True
        )
        self.region.sigRegionChanged.connect(self.on_region_changed)

        self.bg_region = pg.LinearRegionItem(
            values=self.bg_region_bounds,
            brush=pg.mkBrush(color=(200, 50, 50, 50)),
            movable=True
        )
        self.bg_region.sigRegionChanged.connect(self.on_bg_region_changed)

        self.tail_plot_widget = pg.PlotWidget()
        self.tail_plot_widget.setLabel('left', 'r(t)')
        self.tail_plot_widget.setLabel('bottom', 'Channel')
        self.tail_plot_widget.setTitle('Time-Resolved Anisotropy r(t)')
        self.tail_plot_widget.addLegend()
        self.tail_plot_widget.setLogMode(x=False, y=False)

        self.plotContainerLayout.addWidget(self.plot_widget)
        self.tailPlotContainerLayout.addWidget(self.tail_plot_widget)
        self.resize(1200, 760)

    def _build_batch_snapshot(self) -> dict:
        g_raw = float(self.g_factor_uncorrected) if self.g_factor_uncorrected is not None else np.nan
        if self.g_factor is not None and np.isfinite(self.g_factor) and float(self.g_factor) > 0.0:
            g_corr = float(self.g_factor)
        elif self.g_factor_corrected is not None and np.isfinite(self.g_factor_corrected):
            g_corr = float(self.g_factor_corrected)
        else:
            g_corr = g_raw
        l1 = float(self.l1_estimate) if self.l1_estimate is not None and np.isfinite(self.l1_estimate) else 0.0
        l2 = float(self.l2_estimate) if self.l2_estimate is not None and np.isfinite(self.l2_estimate) else l1

        bg_vv = 0.0
        bg_vh = 0.0
        if self.time_axis is not None:
            par_arr, perp_arr = self._get_par_perp()
            if par_arr is not None and perp_arr is not None:
                time_axis = np.asarray(self.time_axis, dtype=float)
                shifted_time_axis = time_axis + float(self.decay_shift)
                bg_vv, bg_vh = compute_background_levels(
                    np.asarray(par_arr, dtype=float),
                    np.asarray(perp_arr, dtype=float),
                    time_axis,
                    shifted_time_axis,
                    self.bg_region_bounds,
                )

        return {
            'g_raw': g_raw,
            'g_corr': g_corr,
            'l1': l1,
            'l2': l2,
            'shift': float(self.decay_shift),
            'flip': bool(self.flip_checkbox.isChecked()),
            'bg_vv': float(bg_vv),
            'bg_vh': float(bg_vh),
            'apply_bg': bool(self.bg_correction_checkbox.isChecked()),
            'region_min': float(min(self.region_bounds)),
            'region_max': float(max(self.region_bounds)),
        }

    def open_batch_window(self):
        snap = self._build_batch_snapshot()
        self._batch_window = JordiDecayBatchWindow(snap, self)
        self._batch_window.exec_()

    def load_jordi_file(self, file_path=None):
        if isinstance(file_path, bool):
            file_path = None
        if file_path is None:
            try:
                import chisurf as _cs
                start_dir = str(getattr(_cs, 'working_path', '') or '')
            except Exception:
                start_dir = ""
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Jordi File", start_dir, "Data Files (*.dat);;All Files (*)"
            )
            if not file_path:
                return

        self.file_label.setText(str(file_path))
        logger.info("JordiGFactorCalculator: loading Jordi file from path %s", file_path)

        try:
            vv_data, vh_data = self._load_jordi_channels(file_path)
            vv_data = np.asarray(vv_data, dtype=float)
            vh_data = np.asarray(vh_data, dtype=float)
            logger.info("JordiGFactorCalculator: loaded %d data points", len(vv_data))
            self.time_axis = np.arange(len(vv_data), dtype=float)

            self.parallel_data = DataCurve(x=self.time_axis, y=vv_data, name="Parallel")
            self.perpendicular_data = DataCurve(x=self.time_axis, y=vh_data, name="Perpendicular")
        except Exception as e:
            logger.error("JordiGFactorCalculator: failed to load Jordi file: %s", e)
            self.file_label.setText(f"Error loading file: {str(e)}")
            return

        self.update_plot()
        self.plot_widget.addItem(self.region)

        if self.time_axis is None:
            return
        data_length = len(self.time_axis)
        self.region_bounds = [
            self.time_axis[int(data_length * 0.7)],
            self.time_axis[int(data_length * 0.9)]
        ]
        self.region.setRegion(self.region_bounds)

        self.bg_region_bounds = [
            self.time_axis[int(data_length * 0.05)],
            self.time_axis[int(data_length * 0.15)]
        ]
        self.bg_region.setRegion(self.bg_region_bounds)

        if self.use_background_correction:
            self.plot_widget.addItem(self.bg_region)

        self.calculate_g_factor()

    @staticmethod
    def _load_jordi_channels(file_path):
        if _read_jordi is not None:
            return _read_jordi(file_path, split=True)
        warnings.warn(
            "Direct Jordi reading via numpy.loadtxt is deprecated. Use chisurf.core.fio.read_jordi instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        jordi_data = np.loadtxt(file_path)
        half_length = len(jordi_data) // 2
        return jordi_data[:half_length], jordi_data[half_length:]

    def load_fp_jordi_file(self, file_path=None):
        self.fp_estimate_available = False
        if isinstance(file_path, bool):
            file_path = None
        if file_path is None:
            try:
                import chisurf as _cs
                start_dir = str(getattr(_cs, 'working_path', '') or '')
            except Exception:
                start_dir = ""
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load FP Jordi File", start_dir, "Data Files (*.dat);;All Files (*)"
            )
            if not file_path:
                return

        self.fp_file_label.setText(str(file_path))
        try:
            vv_data, vh_data = self._load_jordi_channels(file_path)
            vv_data = np.asarray(vv_data, dtype=float)
            vh_data = np.asarray(vh_data, dtype=float)
            n = min(len(vv_data), len(vh_data))
            self.fp_file_path = str(file_path)
            self.fp_parallel_data = vv_data[:n]
            self.fp_perpendicular_data = vh_data[:n]
        except Exception as e:
            self.fp_file_label.setText(f"Error loading FP file: {str(e)}")
            self.fp_file_path = None
            self.fp_parallel_data = None
            self.fp_perpendicular_data = None

        self.calculate_fp_mixing_estimate()

    def on_bg_correction_changed(self, state):
        self.use_background_correction = (state == Qt.Checked)
        if self.use_background_correction:
            self.plot_widget.addItem(self.bg_region)
        else:
            self.plot_widget.removeItem(self.bg_region)

        self.corrected_g_factor_label.setVisible(self.use_background_correction)
        self.corrected_g_factor_value.setVisible(self.use_background_correction)
        self.corrected_g_factor_stddev_label.setVisible(self.use_background_correction)
        self.corrected_g_factor_stddev_value.setVisible(self.use_background_correction)
        self.bg_parallel_label.setVisible(self.use_background_correction)
        self.bg_parallel_value.setVisible(self.use_background_correction)
        self.bg_perpendicular_label.setVisible(self.use_background_correction)
        self.bg_perpendicular_value.setVisible(self.use_background_correction)

        self.calculate_g_factor()

    def on_region_changed(self):
        self.region_bounds = self.region.getRegion()
        self.calculate_g_factor()

    def on_bg_region_changed(self):
        self.bg_region_bounds = self.bg_region.getRegion()
        self.calculate_g_factor()

    def on_shift_changed(self, value):
        self.decay_shift = value
        self.update_plot()
        self.calculate_g_factor()

    def calculate_g_factor(self):
        if self.parallel_data is None or self.perpendicular_data is None:
            return
        par_full, perp_full = self._get_par_perp()
        if par_full is None or perp_full is None:
            return

        try:
            logger.debug(
                "JordiGFactorCalculator: invoking RPC client.calculate with region_bounds=%s, shift=%f, use_bg=%s",
                self.region_bounds, self.decay_shift, self.use_background_correction
            )
            res = self._client.calculate(
                parallel_data=par_full.tolist() if isinstance(par_full, np.ndarray) else par_full,
                perpendicular_data=perp_full.tolist() if isinstance(perp_full, np.ndarray) else perp_full,
                region_bounds=self.region_bounds,
                decay_shift=self.decay_shift,
                use_bg=self.use_background_correction,
                bg_region_bounds=self.bg_region_bounds,
                flip=False,
            )
            logger.info("JordiGFactorCalculator: RPC client.calculate completed successfully. Result: %s", res)
        except Exception as e:
            logger.error("JordiGFactorCalculator: RPC calculation failed: %s", e)
            QMessageBox.critical(self, "Calculation Error", f"RPC calculation failed: {e}")
            return

        g_factor_uncorrected = res.get("g_factor_uncorrected")
        g_factor_stddev_uncorrected = res.get("g_factor_stddev_uncorrected")
        g_factor_corrected = res.get("g_factor_corrected")
        g_factor_stddev_corrected = res.get("g_factor_stddev_corrected")
        bg_parallel_avg = res.get("bg_parallel_avg")
        bg_perpendicular_avg = res.get("bg_perpendicular_avg")

        self.g_factor_uncorrected = g_factor_uncorrected
        self.g_factor_corrected = g_factor_corrected
        self.g_factor = res.get("g_factor")

        if g_factor_uncorrected is not None:
            if not self.manual_g_override:
                self._updating_g_value = True
                self.g_factor_value.setText(f"{g_factor_uncorrected:.4f}")
                self._updating_g_value = False
            self.g_factor_stddev_value.setText(f"{g_factor_stddev_uncorrected:.4f}")
            self.g_factor_value.selectAll()
            self.g_factor_value.clearFocus()
        else:
            if not self.manual_g_override:
                self._updating_g_value = True
                self.g_factor_value.setText("N/A")
                self._updating_g_value = False
            self.g_factor_stddev_value.setText("N/A")

        if self.use_background_correction:
            self.bg_parallel_value.setText(f"{bg_parallel_avg:.4f}")
            self.bg_perpendicular_value.setText(f"{bg_perpendicular_avg:.4f}")

            if g_factor_corrected is not None:
                self.corrected_g_factor_value.setText(f"{g_factor_corrected:.4f}")
                self.corrected_g_factor_stddev_value.setText(f"{g_factor_stddev_corrected:.4f}")
            else:
                self.corrected_g_factor_value.setText("N/A")
                self.corrected_g_factor_stddev_value.setText("N/A")

        if self.manual_g_override and self.g_factor_manual_value is not None:
            self.g_factor = float(self.g_factor_manual_value)
            self._updating_g_value = True
            self.g_factor_value.setText(f"{self.g_factor:.4f}")
            self._updating_g_value = False

        self.update_plot()
        self.update_rt_plot()
        self.calculate_fp_mixing_estimate()

    def calculate_fp_mixing_estimate(self):
        if self.fp_parallel_data is None or self.fp_perpendicular_data is None:
            self._set_fp_outputs(warning_text="Warning: Load FP Jordi data to estimate l1/l2.")
            return
        if self.g_factor is None or not np.isfinite(self.g_factor) or float(self.g_factor) <= 0.0:
            self._set_fp_outputs(
                warning_text="Warning: Determine a valid g-factor first using the fast-rotating dye tail match."
            )
            return

        n = min(len(self.fp_parallel_data), len(self.fp_perpendicular_data))
        if n < 3:
            self._set_fp_outputs(warning_text="Warning: FP file is too short for robust lifetime/mixing estimation.")
            return

        t = np.arange(n, dtype=float)
        par = np.asarray(self.fp_parallel_data[:n], dtype=float)
        perp = np.asarray(self.fp_perpendicular_data[:n], dtype=float)

        if self.flip_checkbox.isChecked():
            par, perp = perp, par

        shifted_t = t + float(self.decay_shift)
        perp_on_t = np.interp(t, shifted_t, perp, left=0.0, right=0.0)

        bg_par = 0.0
        bg_perp = 0.0
        if self.use_background_correction:
            bg_min_time, bg_max_time = self.bg_region_bounds
            bg_min_idx_parallel = np.argmin(np.abs(t - bg_min_time))
            bg_max_idx_parallel = np.argmin(np.abs(t - bg_max_time))
            bg_min_idx_perp = np.argmin(np.abs(shifted_t - bg_min_time))
            bg_max_idx_perp = np.argmin(np.abs(shifted_t - bg_max_time))
            if bg_max_idx_parallel > bg_min_idx_parallel:
                bg_par = float(np.mean(par[bg_min_idx_parallel:bg_max_idx_parallel]))
            if bg_max_idx_perp > bg_min_idx_perp:
                bg_perp = float(np.mean(perp[bg_min_idx_perp:bg_max_idx_perp]))

        par_corr = np.clip(par - bg_par, 0.0, None)
        perp_corr = np.clip(perp_on_t - bg_perp, 0.0, None)

        sp = float(np.sum(par_corr))
        ss = float(np.sum(perp_corr))
        if sp <= 0.0 or ss < 0.0:
            self._set_fp_outputs(
                warning_text="Warning: FP integrals are invalid after correction; adjust background/shift or use a different region/file."
            )
            return

        intensity = par_corr + 2.0 * float(self.g_factor) * perp_corr
        tau_channels = estimate_lifetime_first_moment(t, intensity)
        dt_ns = float(self.fp_dt_spinbox.value())
        tau_est_ns = float(tau_channels * dt_ns) if np.isfinite(tau_channels) else np.nan
        if self.fp_manual_tau_override:
            tau_ns = float(self.fp_tau_value.value())
        else:
            tau_ns = tau_est_ns

        try:
            rs_expected = self._client.perrin_steady_state(
                tau_ns=tau_ns,
                rho_ns=float(self.fp_rho_spinbox.value()),
                r0=float(self.fp_r0_spinbox.value()),
            )
        except Exception:
            rs_expected = perrin_steady_state_anisotropy(
                tau_ns=tau_ns,
                rho_ns=float(self.fp_rho_spinbox.value()),
                r0=float(self.fp_r0_spinbox.value()),
            )

        if self.fp_manual_rs_override:
            rs_expected = float(self.fp_rs_value.value())

        try:
            l_est = self._client.solve_linked_l(
                sp=sp,
                ss=ss,
                g_factor=float(self.g_factor),
                r_target=rs_expected,
            )
        except Exception:
            l_est = solve_linked_l_from_steady_state(
                sp=sp,
                ss=ss,
                g_factor=float(self.g_factor),
                r_target=rs_expected,
            )

        if self.fp_manual_l1_override:
            self._set_fp_outputs(
                tau_ns=tau_est_ns,
                rs_expected=rs_expected,
                l1=float(self.fp_l1_value.value()),
                l2=float(self.fp_l1_value.value()),
                warning_text="Warning: Manual l1/l2 override active; entered value is used directly.",
            )
            return

        if not np.isfinite(l_est):
            self._set_fp_outputs(
                tau_ns=tau_est_ns,
                rs_expected=rs_expected,
                warning_text="Warning: Could not solve linked l1=l2 from FP steady-state estimate (degenerate numeric condition).",
            )
            return

        if l_est < 0.0 or l_est > 0.5:
            self._set_fp_outputs(
                tau_ns=tau_est_ns,
                rs_expected=rs_expected,
                warning_text=(
                    f"Warning: linked l1=l2 estimate {l_est:.5f} is outside expected range [0, 0.5]. "
                    "Estimate shown for diagnostics only and will not be auto-applied."
                ),
            )
            return

        self._set_fp_outputs(tau_ns=tau_est_ns, rs_expected=rs_expected, l1=l_est, l2=l_est)

    def _set_fp_outputs(self, tau_ns=np.nan, rs_expected=np.nan, l1=np.nan, l2=np.nan, warning_text=None):
        self.fp_tau_estimate_ns = float(tau_ns) if np.isfinite(tau_ns) else None
        self.fp_rs_expected = float(rs_expected) if np.isfinite(rs_expected) else None

        auto_l = float(l1) if np.isfinite(l1) else np.nan
        manual_l = float(self.fp_l1_value.value()) if self.fp_manual_l1_override else np.nan
        final_l = manual_l if np.isfinite(manual_l) else auto_l

        self.l1_estimate = float(final_l) if np.isfinite(final_l) else None
        self.l2_estimate = self.l1_estimate
        self.fp_estimate_available = bool(
            self.fp_file_path
            and self.l1_estimate is not None
            and np.isfinite(self.l1_estimate)
        )

        if not self.fp_manual_tau_override:
            self._updating_fp_tau_value = True
            self.fp_tau_value.blockSignals(True)
            self.fp_tau_value.setValue(float(tau_ns) if np.isfinite(tau_ns) else 0.0)
            self.fp_tau_value.blockSignals(False)
            self._updating_fp_tau_value = False
        if not self.fp_manual_rs_override:
            self._updating_fp_rs_value = True
            self.fp_rs_value.blockSignals(True)
            self.fp_rs_value.setValue(float(rs_expected) if np.isfinite(rs_expected) else 0.0)
            self.fp_rs_value.blockSignals(False)
            self._updating_fp_rs_value = False
        if not self.fp_manual_l1_override:
            self._updating_fp_l1_value = True
            self.fp_l1_value.setValue(float(auto_l) if np.isfinite(auto_l) else 0.0)
            self._updating_fp_l1_value = False

        if warning_text:
            self.fp_warning_label.setText(warning_text)
        else:
            self.fp_warning_label.setText(
                "Warning: FP l1/l2 from steady-state anisotropy is an estimate; with one steady-state observable only one linked parameter can be determined."
            )

    def on_fp_l1_value_changed(self, _value):
        if self._updating_fp_l1_value:
            return
        self.fp_manual_l1_override = True
        v = float(self.fp_l1_value.value())
        self.l1_estimate = v
        self.l2_estimate = v
        self.update_rt_plot()
        self.calculate_fp_mixing_estimate()

    def on_fp_tau_value_changed(self, _value):
        if self._updating_fp_tau_value:
            return
        self.fp_manual_tau_override = True
        self.calculate_fp_mixing_estimate()

    def on_fp_rs_value_changed(self, _value):
        if self._updating_fp_rs_value:
            return
        self.fp_manual_rs_override = True
        self.calculate_fp_mixing_estimate()

    def on_g_factor_value_changed(self):
        self._apply_manual_g_from_text()

    def on_g_factor_text_changed(self, _text):
        self._apply_manual_g_from_text()

    def _apply_manual_g_from_text(self):
        if self._updating_g_value:
            return
        txt = str(self.g_factor_value.text()).strip()
        try:
            v = float(txt)
        except Exception:
            return
        if not np.isfinite(v) or v <= 0.0:
            return
        self.manual_g_override = True
        self.g_factor_manual_value = float(v)
        self.g_factor = float(v)
        self.update_plot()
        self.update_rt_plot()
        self.calculate_fp_mixing_estimate()

    def on_plot_visibility_changed(self, _state):
        self.update_plot()
        self.update_rt_plot()

    def _get_par_perp(self):
        if self.parallel_data is None or self.perpendicular_data is None:
            return None, None
        par = self.parallel_data.y
        perp = self.perpendicular_data.y
        flip = getattr(self, 'flip_checkbox', None)
        if flip is not None and flip.isChecked():
            return perp, par
        return par, perp

    def _get_fp_par_perp(self):
        if self.fp_parallel_data is None or self.fp_perpendicular_data is None:
            return None, None
        par = self.fp_parallel_data
        perp = self.fp_perpendicular_data
        flip = getattr(self, 'flip_checkbox', None)
        if flip is not None and flip.isChecked():
            return perp, par
        return par, perp

    def update_plot(self):
        self.plot_widget.clear()
        if self.time_axis is None:
            return
        par_raw, perp_raw = self._get_par_perp()
        if par_raw is None or perp_raw is None:
            return

        time_axis = np.asarray(self.time_axis, dtype=float)
        shifted_time_axis = time_axis + float(self.decay_shift)

        bg_vv, bg_vh = 0.0, 0.0
        if self.use_background_correction:
            bg_vv, bg_vh = compute_background_levels(
                par_raw, perp_raw, time_axis, shifted_time_axis, self.bg_region_bounds
            )

        g_unc = float(self.g_factor_uncorrected) if self.g_factor_uncorrected is not None else np.nan
        if self.g_factor is not None and np.isfinite(self.g_factor) and float(self.g_factor) > 0.0:
            g_cor = float(self.g_factor)
        else:
            g_cor = float(self.g_factor_corrected) if self.g_factor_corrected is not None else np.nan

        show_fast = bool(self.show_fast_checkbox.isChecked())
        show_slow = bool(self.show_slow_checkbox.isChecked())
        show_raw = bool(self.show_raw_checkbox.isChecked())
        show_corrected = bool(self.show_corrected_checkbox.isChecked())

        if show_fast:
            self._plot_decay_set(
                self.plot_widget, time_axis, shifted_time_axis,
                par_raw, perp_raw, bg_vv, bg_vh, g_unc, g_cor,
                prefix="fast", show_raw=show_raw, show_corrected=show_corrected
            )

        fp_par, fp_perp = self._get_fp_par_perp()
        if show_slow and fp_par is not None and fp_perp is not None:
            n_fp = min(len(fp_par), len(fp_perp))
            if n_fp > 0:
                fp_time = np.arange(n_fp, dtype=float)
                fp_shifted_time = fp_time + float(self.decay_shift)
                bg_fp_vv, bg_fp_vh = 0.0, 0.0
                if self.use_background_correction:
                    bg_fp_vv, bg_fp_vh = compute_background_levels(
                        fp_par, fp_perp, fp_time, fp_shifted_time, self.bg_region_bounds
                    )
                self._plot_decay_set(
                    self.plot_widget, fp_time, fp_shifted_time,
                    fp_par[:n_fp], fp_perp[:n_fp], bg_fp_vv, bg_fp_vh, g_unc, g_cor,
                    prefix="slow", show_raw=show_raw, show_corrected=show_corrected
                )

        if self.use_background_correction:
            self.plot_widget.addItem(self.bg_region)
        self.plot_widget.addItem(self.region)

    @staticmethod
    def _plot_decay_set(plot_widget, time_axis, shifted_time_axis, par_raw, perp_raw, bg_par, bg_perp, g_unc, g_cor, prefix, show_raw=True, show_corrected=True):
        par_corr = np.maximum(par_raw - bg_par, 0.0)
        perp_corr = np.maximum(perp_raw - bg_perp, 0.0)

        is_fast = str(prefix).lower().startswith('fast')
        if is_fast:
            vv_raw_color = (31, 119, 180, 90)
            vh_raw_color = (214, 39, 40, 90)
            vv_cor_color = (23, 190, 207, 220)
            vh_cor_color = (255, 127, 14, 220)
        else:
            vv_raw_color = (44, 160, 44, 90)
            vh_raw_color = (255, 152, 0, 90)
            vv_cor_color = (46, 204, 113, 220)
            vh_cor_color = (241, 196, 15, 220)

        if show_raw:
            plot_widget.plot(time_axis, par_raw, pen=pg.mkPen(vv_raw_color, width=1.6), name=f'{prefix} VV raw')
            plot_widget.plot(shifted_time_axis, perp_raw, pen=pg.mkPen(vh_raw_color, width=1.6), name=f'{prefix} VH raw')
        if show_corrected:
            plot_widget.plot(time_axis, par_corr, pen=pg.mkPen(vv_cor_color, width=1.6), name=f'{prefix} VV corr')

        if show_corrected and np.isfinite(g_cor) and g_cor > 0.0:
            plot_widget.plot(
                shifted_time_axis,
                perp_corr * g_cor,
                pen=pg.mkPen(vh_cor_color, width=1.6, style=_DASH_LINE_STYLE),
                name=f'{prefix} VH corr * G ({g_cor:.3f})'
            )

    def update_rt_plot(self):
        self.tail_plot_widget.clear()

        l1_corr = float(self.l1_estimate) if self.l1_estimate is not None and np.isfinite(self.l1_estimate) else 0.0
        l2_corr = float(self.l2_estimate) if self.l2_estimate is not None and np.isfinite(self.l2_estimate) else l1_corr
        g_unc = float(self.g_factor_uncorrected) if self.g_factor_uncorrected is not None else np.nan
        if self.g_factor is not None and np.isfinite(self.g_factor) and float(self.g_factor) > 0.0:
            g_cor = float(self.g_factor)
        else:
            g_cor = float(self.g_factor_corrected) if self.g_factor_corrected is not None else np.nan
        show_fast = bool(self.show_fast_checkbox.isChecked())
        show_slow = bool(self.show_slow_checkbox.isChecked())
        show_raw = bool(self.show_raw_checkbox.isChecked())
        show_corr = bool(self.show_corrected_checkbox.isChecked())

        def _plot_dataset_rt(prefix, time_axis, par_raw, perp_raw):
            shifted_time_axis = time_axis + self.decay_shift
            perp_on_t_raw = np.asarray(perp_raw, dtype=float)
            bg_par, bg_perp = compute_background_levels(par_raw, perp_raw, time_axis, shifted_time_axis, self.bg_region_bounds)
            par_corr = np.maximum(par_raw - bg_par, 0.0)
            perp_corr_raw = np.maximum(perp_raw - bg_perp, 0.0)
            perp_on_t_corr = np.interp(time_axis, shifted_time_axis, perp_corr_raw, left=0.0, right=0.0)

            is_fast = str(prefix).lower().startswith('fast')
            if is_fast:
                rt_raw_color = (0, 200, 255, 95)
                rt_cor_color = (255, 215, 0, 230)
            else:
                rt_raw_color = (120, 255, 120, 95)
                rt_cor_color = (0, 255, 120, 230)

            if show_raw and np.isfinite(g_unc) and g_unc > 0.0:
                r_unc = compute_rt(par_raw, perp_on_t_raw, g_unc, l1=0.0, l2=0.0)
                r_unc = np.clip(r_unc, -0.5, 1.5)
                self.tail_plot_widget.plot(
                    time_axis,
                    r_unc,
                    pen=pg.mkPen(rt_raw_color, width=1.8),
                    name=f'{prefix} r(t) raw, G={g_unc:.4f}'
                )
            if show_corr and np.isfinite(g_cor) and g_cor > 0.0:
                r_cor = compute_rt(par_corr, perp_on_t_corr, g_cor, l1=l1_corr, l2=l2_corr)
                r_cor = np.clip(r_cor, -0.5, 1.5)
                self.tail_plot_widget.plot(
                    time_axis,
                    r_cor,
                    pen=pg.mkPen(rt_cor_color, width=1.8, style=_DASH_LINE_STYLE),
                    name=f'{prefix} r(t) corr, G={g_cor:.4f}, l1={l1_corr:.4f}, l2={l2_corr:.4f}'
                )

        if show_fast and self.time_axis is not None:
            par_full, perp_full = self._get_par_perp()
            if par_full is not None and perp_full is not None:
                fast_time = np.asarray(self.time_axis, dtype=float)
                _plot_dataset_rt('fast', fast_time, np.asarray(par_full, dtype=float), np.asarray(perp_full, dtype=float))

        fp_par, fp_perp = self._get_fp_par_perp()
        if show_slow and fp_par is not None and fp_perp is not None:
            n_fp = min(len(fp_par), len(fp_perp))
            if n_fp > 0:
                fp_time = np.arange(n_fp, dtype=float)
                _plot_dataset_rt('slow', fp_time, np.asarray(fp_par[:n_fp], dtype=float), np.asarray(fp_perp[:n_fp], dtype=float))

        self.tail_plot_widget.setTitle(f'r(t): fast+slow, raw+corr (Shift: {self.decay_shift:.3f} ch)')
        self.tail_plot_widget.setYRange(-0.5, 1.5, padding=0.0)
