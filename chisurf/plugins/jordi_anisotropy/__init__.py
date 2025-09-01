"""
Jordi Anisotropy Calculator

This plugin provides interactive computation and visualization of fluorescence anisotropy r(t)
for Jordi files.

Features
- Load Jordi ASCII files and split into VV (parallel) and VH (perpendicular) decays
- Compute r(t) = (VV − g·VH) / (VV + 2·g·VH)
- Apply user-specified g-factor, optional constant backgrounds (BG VV, BG VH), and a fractional
  channel shift between VV and VH (VH relative to VV)
- Plot background-corrected decays on a semilogarithmic axis and r(t) using pyqtgraph
- Select a region on r(t) to estimate r∞; r∞ is subtracted from r(t) and saved alongside the data
- Use channel indices (0..N−1) for the x-axis
- Fix the anisotropy y-range to [0, 0.45] for visual consistency
- Save outputs via “Save…”:
  • Shifted decays as a Jordi file (<base>_shifted.dat)
  • Anisotropy decay as text with columns: channel, r(t), r(t)−r∞
  • r∞ metadata CSV including source filename, region bounds, BG VV, BG VH, and g-factor
- Batch processing with drag-and-drop file list (Batch…) and CSV export of per-file r∞

This widget can run as a ChiSurf plugin (see chisurf.plugins.jordi_anisotropy.__plugin__) or standalone.
"""

import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QFileDialog, QLabel, QDoubleSpinBox, QLineEdit, QCheckBox,
    QListWidget, QListWidgetItem, QAbstractItemView, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import pyqtgraph as pg

try:
    from chisurf.fio import read_jordi as _read_jordi
except Exception:
    _read_jordi = None

# Optional writer for Jordi files
try:
    from chisurf.fio.jordi import write_jordi as _write_jordi
except Exception:
    _write_jordi = None


name = "Fluorescence decay:Jordi Anisotropy Decay"

# Plugin icon used by the Plugin Manager and window decoration
try:
    _plugin_dir = Path(__file__).parent
    _png = _plugin_dir / "icon.png"
    _svg = _plugin_dir / "icon.svg"
    if _png.exists():
        icon = QIcon(str(_png))
    elif _svg.exists():
        icon = QIcon(str(_svg))
    else:
        icon = QIcon()
except Exception:
    icon = QIcon()


class FileDropList(QListWidget):
    """QListWidget that accepts file drops (multiple) and deduplicates entries."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def add_files(self, paths):
        existing = set(self._all_paths())
        for p in paths:
            if p and p not in existing:
                self.addItem(QListWidgetItem(p))
                existing.add(p)

    def _all_paths(self):
        return [self.item(i).text() for i in range(self.count())]

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


class JordiAnisotropyBatchWindow(QWidget):
    """
    Batch processing window for Jordi Anisotropy.
    Allows dropping multiple files and processes them with the same settings
    snapshot taken from the main window.
    """
    def __init__(self, settings_snapshot: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Jordi Anisotropy Batch Processor")
        try:
            self.setWindowIcon(icon)
        except Exception:
            pass
        self.snapshot = settings_snapshot
        self.results = []  # list of tuples for table/CSV
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # File list and controls
        self.file_list = FileDropList()
        layout.addWidget(QLabel("Drop Jordi files here or use Add…"))
        layout.addWidget(self.file_list)

        buttons = QHBoxLayout()
        self.add_btn = QPushButton("Add…")
        self.add_btn.clicked.connect(self._on_add)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._on_remove)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear)
        self.run_btn = QPushButton("Run Batch")
        self.run_btn.clicked.connect(self._on_run)
        self.save_btn = QPushButton("Save CSV…")
        self.save_btn.clicked.connect(self._on_save)
        buttons.addWidget(self.add_btn)
        buttons.addWidget(self.remove_btn)
        buttons.addWidget(self.clear_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.save_btn)
        layout.addLayout(buttons)

        # Results table
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "filename", "r_inf", "region_min", "region_max", "bg_vv", "bg_vh", "g_factor"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        self.setLayout(layout)
        self.resize(900, 600)

    def _on_add(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add Jordi Files", "", "Data Files (*.dat *.txt);;All Files (*)")
        if files:
            self.file_list.add_files(files)

    def _on_remove(self):
        for item in self.file_list.selectedItems():
            row = self.file_list.row(item)
            self.file_list.takeItem(row)

    def _on_clear(self):
        self.file_list.clear()
        self._clear_results()

    def _clear_results(self):
        self.table.setRowCount(0)
        self.results = []

    @staticmethod
    def _shift_interp_on_axis(t: np.ndarray, y: np.ndarray, shift: float) -> np.ndarray:
        if y is None or t is None:
            return y
        if shift == 0.0:
            return y.astype(float).copy()
        xq = t - shift
        out = np.full_like(y, np.nan, dtype=float)
        mask = (xq >= t[0]) & (xq <= t[-1])
        if np.any(mask):
            out[mask] = np.interp(xq[mask], t, y)
        return out

    def _compute_rinf_for_file(self, filepath: str) -> tuple:
        # returns (filename, r_inf, region_min, region_max, bg_vv, bg_vh, g)
        try:
            if _read_jordi is not None:
                vv, vh = _read_jordi(filepath, split=True)
            else:
                vec = np.loadtxt(filepath)
                half = len(vec) // 2
                vv, vh = vec[:half], vec[half:]
        except Exception as e:
            return (Path(filepath).name, np.nan, self.snapshot['region_min'], self.snapshot['region_max'],
                    self.snapshot['bg_vv'], self.snapshot['bg_vh'], self.snapshot['g'])

        t = np.arange(len(vv), dtype=float)
        vv = np.asarray(vv, dtype=float)
        vh = np.asarray(vh, dtype=float)

        # Apply optional flip
        if bool(self.snapshot.get('flip', False)):
            vv, vh = vh, vv

        if self.snapshot['apply_bg']:
            vv = vv - float(self.snapshot['bg_vv'])
            vh = vh - float(self.snapshot['bg_vh'])

        vh_shift = self._shift_interp_on_axis(t, vh, float(self.snapshot['shift']))
        denom = vv + 2.0 * float(self.snapshot['g']) * vh_shift
        num = vv - float(self.snapshot['g']) * vh_shift
        r = np.full_like(vv, np.nan, dtype=float)
        valid = (~np.isnan(denom)) & (denom > 0)
        r[valid] = num[valid] / denom[valid]

        # Region indices (clamped)
        rmin = float(self.snapshot['region_min'])
        rmax = float(self.snapshot['region_max'])
        # Clamp to available t range
        rmin = max(t[0], min(rmin, t[-1]))
        rmax = max(t[0], min(rmax, t[-1]))
        if rmax <= rmin:
            rmax = min(t[-1], rmin + 1.0)
        i0 = int(np.argmin(np.abs(t - rmin)))
        i1 = int(np.argmin(np.abs(t - rmax)))
        if i1 <= i0:
            i1 = min(len(t), i0 + 1)
        r_region = r[i0:i1]
        r_region = r_region[~np.isnan(r_region)]
        r_inf = float(np.nanmean(r_region)) if r_region.size > 0 else np.nan

        return (Path(filepath).name, r_inf, rmin, rmax, float(self.snapshot['bg_vv']), float(self.snapshot['bg_vh']), float(self.snapshot['g']))

    def _on_run(self):
        paths = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not paths:
            QMessageBox.information(self, "Batch", "No files to process.")
            return
        self._clear_results()
        for p in paths:
            res = self._compute_rinf_for_file(p)
            self._append_result_row(res)
        QMessageBox.information(self, "Batch", f"Processed {len(paths)} file(s).")

    def _append_result_row(self, res_tuple: tuple):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, val in enumerate(res_tuple):
            item = QTableWidgetItem(str(val))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, col, item)
        self.results.append(res_tuple)

    def _on_save(self):
        if not self.results:
            QMessageBox.information(self, "Save CSV", "No results to save.")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not out_path:
            return
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("filename,r_inf,region_min,region_max,bg_vv,bg_vh,g_factor\n")
                for row in self.results:
                    f.write(",".join(str(x) for x in row) + "\n")
            QMessageBox.information(self, "Save CSV", f"Saved: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save CSV", f"Failed to save CSV: {e}")


class JordiAnisotropyCalculator(QWidget):
    """
    Jordi Anisotropy Calculator
    - Loads a Jordi file (VV, VH)
    - Computes r(t) = (VV - g*VH)/(VV + 2*g*VH)
    - Considers user-specified g-factor, VV/VH background values
    - Supports time shift between VV and VH (VH shifted relative to VV)
    - Plots decays (log-y) and r(t) with a LinearRegionItem to define the r∞ region
      and subtract it from r(t) as an offset.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jordi Anisotropy Calculator")
        try:
            self.setWindowIcon(icon)
        except Exception:
            pass

        # Data
        self.time_axis = None
        self.vv = None
        self.vh = None
        self.loaded_file = None

        # Parameters
        self.g_factor = 1.0
        self.bg_vv = 0.0
        self.bg_vh = 0.0
        self.decay_shift = 0.0  # channels; VH shifted relative to VV

        # Computed
        self.r_t = None
        self.r_infty = np.nan

        # UI
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout()

        controls = QGridLayout()

        # File load
        self.load_button = QPushButton("Load Jordi File")
        self.load_button.clicked.connect(self.load_jordi_file)
        self.file_label = QLineEdit("No file loaded")
        self.file_label.setReadOnly(True)
        self.save_button = QPushButton("Save…")
        self.save_button.clicked.connect(self.save_outputs)
        controls.addWidget(self.load_button, 0, 0)
        controls.addWidget(self.file_label, 0, 1, 1, 5)
        controls.addWidget(self.save_button, 0, 6)
        self.batch_button = QPushButton("Batch…")
        self.batch_button.clicked.connect(self.open_batch)
        controls.addWidget(self.batch_button, 0, 7)

        # g-factor
        controls.addWidget(QLabel("g-factor:"), 1, 0)
        self.g_spin = QDoubleSpinBox()
        self.g_spin.setDecimals(5)
        self.g_spin.setRange(0.0, 10.0)
        self.g_spin.setSingleStep(0.001)
        self.g_spin.setValue(self.g_factor)
        self.g_spin.valueChanged.connect(self._on_param_changed)
        controls.addWidget(self.g_spin, 1, 1)

        # Backgrounds
        self.bg_checkbox = QCheckBox("Apply backgrounds")
        self.bg_checkbox.setChecked(True)
        self.bg_checkbox.stateChanged.connect(self._on_param_changed)
        controls.addWidget(self.bg_checkbox, 1, 2, 1, 2)

        controls.addWidget(QLabel("BG VV:"), 1, 4)
        self.bg_vv_spin = QDoubleSpinBox()
        self.bg_vv_spin.setDecimals(3)
        self.bg_vv_spin.setRange(-1e9, 1e9)
        self.bg_vv_spin.setSingleStep(1.0)
        self.bg_vv_spin.setValue(self.bg_vv)
        self.bg_vv_spin.valueChanged.connect(self._on_param_changed)
        controls.addWidget(self.bg_vv_spin, 1, 5)

        controls.addWidget(QLabel("BG VH:"), 1, 6)
        self.bg_vh_spin = QDoubleSpinBox()
        self.bg_vh_spin.setDecimals(3)
        self.bg_vh_spin.setRange(-1e9, 1e9)
        self.bg_vh_spin.setSingleStep(1.0)
        self.bg_vh_spin.setValue(self.bg_vh)
        self.bg_vh_spin.valueChanged.connect(self._on_param_changed)
        controls.addWidget(self.bg_vh_spin, 1, 7)

        # Flip VV<->VH
        self.flip_checkbox = QCheckBox("Flip VV↔VH (data swapped in Jordi)")
        self.flip_checkbox.setChecked(False)
        self.flip_checkbox.stateChanged.connect(self._on_param_changed)
        controls.addWidget(self.flip_checkbox, 2, 0, 1, 2)

        # Shift
        controls.addWidget(QLabel("Shift VH (channels):"), 2, 2)
        self.shift_spin = QDoubleSpinBox()
        self.shift_spin.setDecimals(3)
        self.shift_spin.setRange(-150.0, 150.0)
        self.shift_spin.setSingleStep(0.5)
        self.shift_spin.setValue(self.decay_shift)
        self.shift_spin.valueChanged.connect(self._on_param_changed)
        controls.addWidget(self.shift_spin, 2, 3)

        # r infinity display and subtraction
        controls.addWidget(QLabel("r∞:"), 2, 4)
        self.rinf_line = QLineEdit("N/A")
        self.rinf_line.setReadOnly(True)
        controls.addWidget(self.rinf_line, 2, 5)


        main_layout.addLayout(controls)

        # Plots
        plots_layout = QHBoxLayout()

        # Decays plot
        self.decay_plot = pg.PlotWidget()
        self.decay_plot.setLabel('left', 'Intensity')
        self.decay_plot.setLabel('bottom', 'Channel')
        self.decay_plot.setTitle('Decays (VV, VH)')
        self.decay_plot.addLegend()
        self.decay_plot.setLogMode(x=False, y=True)
        plots_layout.addWidget(self.decay_plot)

        # r(t) plot
        self.r_plot = pg.PlotWidget()
        self.r_plot.setLabel('left', 'r(t)')
        self.r_plot.setLabel('bottom', 'Channel')
        self.r_plot.setTitle('Anisotropy r(t)')
        self.r_plot.addLegend()
        # Fix y-axis range for anisotropy to [0, 0.45]
        try:
            self.r_plot.enableAutoRange('y', False)
        except Exception:
            pass
        self.r_plot.setYRange(0.0, 0.45)
        plots_layout.addWidget(self.r_plot)

        # Region for r∞ in r(t) plot
        self.region_bounds = [0.0, 1.0]
        self.region = pg.LinearRegionItem(
            values=self.region_bounds,
            brush=pg.mkBrush(color=(50, 200, 50, 50)),
            movable=True
        )
        self.region.sigRegionChanged.connect(self._on_region_changed)

        main_layout.addLayout(plots_layout)
        self.setLayout(main_layout)
        self.resize(1100, 650)

    # --------- Loading and computations ---------
    def load_jordi_file(self, file_path=None):
        # PyQt's clicked(bool) passes a boolean when connected directly. Treat booleans as no path.
        if isinstance(file_path, bool):
            file_path = None
        if file_path is None:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Jordi File", "", "Data Files (*.dat *.txt);;All Files (*)"
            )
            if not file_path:
                return
        self.file_label.setText(str(file_path))
        self.loaded_file = str(file_path)

        try:
            if _read_jordi is not None:
                vv, vh = _read_jordi(file_path, split=True)
            else:
                vec = np.loadtxt(file_path)
                half = len(vec) // 2
                vv, vh = vec[:half], vec[half:]
        except Exception as e:
            self.file_label.setText(f"Error loading file: {e}")
            return

        # Create channel axis (0..N-1)
        self.time_axis = np.arange(len(vv), dtype=float)
        # Store raw and keep backward-compatible attributes
        self.vv_raw = np.asarray(vv, dtype=float)
        self.vh_raw = np.asarray(vh, dtype=float)
        self.vv = self.vv_raw
        self.vh = self.vh_raw

        # Set region initially to last 20% for r∞
        n = len(self.time_axis)
        self.region_bounds = [self.time_axis[int(n*0.7)], self.time_axis[int(n*0.9)]]
        # Add region to r plot
        if self.region not in self.r_plot.items():
            self.r_plot.addItem(self.region)
        self.region.setRegion(self.region_bounds)

        self._recompute_and_update_plots()

    def _on_param_changed(self, *args, **kwargs):
        self._recompute_and_update_plots()

    def open_batch(self):
        """Open the batch processing window with a snapshot of current settings."""
        try:
            region_min, region_max = (self.region_bounds if self.region_bounds else [0.0, 0.0])
        except Exception:
            region_min, region_max = 0.0, 0.0
        snapshot = {
            'apply_bg': bool(self.bg_checkbox.isChecked()),
            'bg_vv': float(self.bg_vv_spin.value()) if hasattr(self, 'bg_vv_spin') else 0.0,
            'bg_vh': float(self.bg_vh_spin.value()) if hasattr(self, 'bg_vh_spin') else 0.0,
            'g': float(self.g_spin.value()) if hasattr(self, 'g_spin') else 1.0,
            'shift': float(self.shift_spin.value()) if hasattr(self, 'shift_spin') else 0.0,
            'flip': bool(self.flip_checkbox.isChecked()) if hasattr(self, 'flip_checkbox') else False,
            'region_min': float(region_min),
            'region_max': float(region_max),
        }
        self._batch_window = JordiAnisotropyBatchWindow(snapshot, None)
        self._batch_window.show()
        try:
            self._batch_window.raise_()
            self._batch_window.activateWindow()
        except Exception:
            pass

    def _on_region_changed(self):
        # Keep internal bounds and recompute r∞/plots
        try:
            self.region_bounds = list(self.region.getRegion())
        except Exception:
            pass
        self._recompute_and_update_plots()

    def _apply_shift_interp(self, y: np.ndarray, shift: float) -> np.ndarray:
        """
        Shift a 1D array y by a fractional number of channels using interpolation
        onto the same channel grid defined by self.time_axis.
        Implements y_shifted[i] = y(t[i] - shift), where t is the channel axis.
        Values evaluated outside the original domain become NaN.
        """
        if y is None or self.time_axis is None:
            return None if y is None else y
        if shift == 0.0:
            return y.copy()
        t = self.time_axis
        xq = t - shift
        out = np.full_like(y, np.nan, dtype=float)
        # Only interpolate where the query points lie within the domain
        mask = (xq >= t[0]) & (xq <= t[-1])
        if np.any(mask):
            out[mask] = np.interp(xq[mask], t, y)
        return out

    def _get_vv_vh(self):
        """Return VV, VH arrays, applying the VV<->VH flip if requested."""
        vv = getattr(self, 'vv_raw', self.vv)
        vh = getattr(self, 'vh_raw', self.vh)
        flip = getattr(self, 'flip_checkbox', None)
        if flip is not None and flip.isChecked():
            return vh, vv
        return vv, vh

    def save_outputs(self):
        """Save shifted decays (as Jordi), anisotropy decay, and r∞ info.
        Generates three files based on a user-chosen base filename:
        - <base>_shifted.dat  (Jordi format: [VV, VH_shifted])
        - <base>_anisotropy.txt (columns: channel, r(t), r(t)-r∞)
        - <base>_rinf.csv (r∞, region_min, region_max)
        """
        # Ensure data is available
        if self.time_axis is None or self.vv is None or self.vh is None:
            return

        base_path, _ = QFileDialog.getSaveFileName(
            self, "Save Outputs", "", "Data Files (*.dat *.txt *.csv);;All Files (*)"
        )
        if not base_path:
            return
        base = Path(base_path)
        stem = base.with_suffix("")

        # Prepare VV and shifted VH with current background setting
        vv_curr, vh_curr = self._get_vv_vh()
        vv_arr = vv_curr.astype(float).copy()
        vh_arr = vh_curr.astype(float).copy()
        if self.bg_checkbox.isChecked():
            vv_arr = vv_arr - float(self.bg_vv_spin.value())
            vh_arr = vh_arr - float(self.bg_vh_spin.value())
        # Shift VH onto VV channel grid
        shift = float(self.shift_spin.value())
        vh_shift = self._apply_shift_interp(vh_arr, shift)
        # Replace NaNs and non-positive values with 0 for saving stability
        vv_save = np.where(np.isfinite(vv_arr) & (vv_arr > 0), vv_arr, 0.0)
        vh_save = np.where(np.isfinite(vh_shift) & (vh_shift > 0), vh_shift, 0.0)

        # 1) Save shifted decays as Jordi file
        jordi_out = np.vstack([vv_save, vh_save])  # shape (2, N)
        jordi_path = stem.parent / f"{stem.name}_shifted.dat"
        try:
            if _write_jordi is not None:
                _write_jordi(jordi_out, jordi_path)
            else:
                # Fallback: concatenate and save
                vec = np.hstack([vv_save, vh_save])
                np.savetxt(jordi_path.as_posix(), vec)
        except Exception:
            # If writing fails, silently ignore to not crash the GUI
            pass

        # 2) Save anisotropy decay as TXT (channel, r(t), r(t)-r∞)
        r_raw, r_inf = None, np.nan
        try:
            r_raw, r_inf = self._compute_r()
        except TypeError:
            # Older return signature fallback
            r_tuple = self._compute_r()
            if isinstance(r_tuple, tuple) and len(r_tuple) >= 2:
                r_raw, r_inf = r_tuple[0], r_tuple[-1]
        # If our class stores the corrected array, use it; otherwise compute correction locally
        if r_raw is None:
            return
        r_inf_val = r_inf if np.isfinite(r_inf) else np.nan
        r_corr = r_raw - r_inf_val if np.isfinite(r_inf_val) else r_raw.copy()
        aniso_path = stem.parent / f"{stem.name}_anisotropy.txt"
        try:
            data_mat = np.column_stack([self.time_axis, r_raw, r_corr])
            header = "channel\tr(t)\tr(t)-r_inf"
            np.savetxt(aniso_path.as_posix(), data_mat, header=header, comments="", delimiter="\t", fmt="%.10g")
        except Exception:
            pass

        # 3) Save r∞ and selection range as CSV
        rinfty_path = stem.parent / f"{stem.name}_rinf.csv"
        try:
            rmin, rmax = (self.region_bounds if self.region_bounds else [self.time_axis[0], self.time_axis[-1]])
            with open(rinfty_path, "w", encoding="utf-8") as f:
                f.write("filename,r_inf,region_min,region_max,bg_vv,bg_vh,g_factor\n")
                src_name = Path(self.loaded_file).name if getattr(self, 'loaded_file', None) else ""
                bg_vv_val = float(self.bg_vv_spin.value()) if hasattr(self, 'bg_vv_spin') else np.nan
                bg_vh_val = float(self.bg_vh_spin.value()) if hasattr(self, 'bg_vh_spin') else np.nan
                g_val = float(self.g_spin.value()) if hasattr(self, 'g_spin') else np.nan
                f.write(f"{src_name},{r_inf_val},{rmin},{rmax},{bg_vv_val},{bg_vh_val},{g_val}\n")
        except Exception:
            pass


    def _compute_r(self):
        if self.time_axis is None or self.vv is None or self.vh is None:
            # No data loaded yet
            return None, np.nan

        g = float(self.g_spin.value())
        vv_curr, vh_curr = self._get_vv_vh()
        vv = vv_curr.astype(float)
        vh = vh_curr.astype(float)

        if self.bg_checkbox.isChecked():
            vv = vv - float(self.bg_vv_spin.value())
            vh = vh - float(self.bg_vh_spin.value())
        # No clipping; mask invalid later

        # Shift VH by decay_shift using interpolation onto the VV time grid
        shift = float(self.shift_spin.value())
        vh_shift = self._apply_shift_interp(vh, shift)

        denom = vv + 2.0 * g * vh_shift
        num = vv - g * vh_shift

        r = np.full_like(vv, np.nan, dtype=float)
        valid = (~np.isnan(denom)) & (denom > 0)
        r[valid] = num[valid] / denom[valid]

        # Compute r∞ in selected region
        tmin, tmax = self.region_bounds if self.region_bounds else (self.time_axis[0], self.time_axis[-1])
        idx_min = int(np.argmin(np.abs(self.time_axis - tmin)))
        idx_max = int(np.argmin(np.abs(self.time_axis - tmax)))
        if idx_max <= idx_min:
            idx_max = min(len(self.time_axis), idx_min + 1)
        r_region = r[idx_min:idx_max]
        r_region = r_region[~np.isnan(r_region)]
        r_inf = float(np.nanmean(r_region)) if r_region.size > 0 else np.nan

        return r, r_inf

    # --------- Plotting ---------
    def _recompute_and_update_plots(self):
        # Recompute r and update both plots
        r, r_inf = self._compute_r()
        self.r_t, self.r_infty = r, r_inf
        if np.isfinite(r_inf):
            self.rinf_line.setText(f"{r_inf:.5f}")
        else:
            self.rinf_line.setText("N/A")

        self._update_decay_plot()
        self._update_r_plot()

    def _update_decay_plot(self):
        self.decay_plot.clear()
        if self.time_axis is None:
            return

        # Determine whether to apply background correction
        apply_bg = self.bg_checkbox.isChecked()
        vv_plot = None
        vh_plot = None
        vv_curr, vh_curr = self._get_vv_vh()
        if vv_curr is not None:
            vv_plot = vv_curr.astype(float).copy()
            if apply_bg:
                vv_plot = vv_plot - float(self.bg_vv_spin.value())
                # For log plotting, mask non-positive values as NaN
                vv_plot = np.where(vv_plot > 0, vv_plot, np.nan)
            self.decay_plot.plot(self.time_axis, vv_plot, pen=pg.mkPen('b', width=2),
                                 name='VV (BG corrected)' if apply_bg else 'VV')

        if vh_curr is not None:
            vh_plot = vh_curr.astype(float).copy()
            if apply_bg:
                vh_plot = vh_plot - float(self.bg_vh_spin.value())
                vh_plot = np.where(vh_plot > 0, vh_plot, np.nan)
            shift = float(self.shift_spin.value())
            # For visualization, shift the time axis of VH
            shifted_time = self.time_axis + shift
            self.decay_plot.plot(shifted_time, vh_plot, pen=pg.mkPen('r', width=2),
                                 name=(f'VH (shift {shift:.3f} ch, BG corrected)'
                                       if apply_bg else f'VH (shift {shift:.3f} ch)'))

    def _update_r_plot(self):
        self.r_plot.clear()
        if self.time_axis is None or self.r_t is None:
            return
        # raw r(t)
        self.r_plot.plot(self.time_axis, self.r_t, pen=pg.mkPen('m', width=2), name='r(t)')
        # ensure region is present
        if self.region not in self.r_plot.items():
            self.r_plot.addItem(self.region)
        # keep region at current bounds
        try:
            self.region.setRegion(self.region_bounds)
        except Exception:
            pass
        # Enforce fixed y-range after replotting
        self.r_plot.setYRange(0.0, 0.45)


if __name__ == '__main__':
    # Simple manual test runner
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = JordiAnisotropyCalculator()
    w.show()
    sys.exit(app.exec_())

elif __name__ == "plugin":
    window = JordiAnisotropyCalculator()
    window.show()