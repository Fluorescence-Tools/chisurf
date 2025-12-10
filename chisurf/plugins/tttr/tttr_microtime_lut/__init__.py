"""
TTTR:BH TAC Linearization

This plugin provides an interactive tool for Time-Tagged Time-Resolved (TTTR) TAC linearization
using the Felekyan et al. algorithm (RSI 2005). It allows users to compute microtime LUTs
from TTTR data with an intuitive GUI for parameter tuning and visualization.

Features:
- Interactive selection of linear region in TAC histogram
- Parameter controls for NTAC requirements, Noffset, RNG seed, and preview settings
- Real-time preview of linearized histogram
- Support for various TTTR file formats (.spc, .ht3, .ptu, etc.)
- Export of LUTs in multiple formats (.txt, .csv, .npy, .npz)
- Drag-and-drop file loading
- Zero-out tools for data cleanup

The plugin implements the paper-faithful Felekyan algorithm for TAC linearization,
which corrects for non-linearities in time-to-amplitude converter (TAC) histograms
by computing relative bin widths and cumulative scaling factors.

Dependencies:
- tttrlib: Required for loading TTTR files
- pyqtgraph: For interactive plotting
- PyQt5: For GUI components

Author: Based on compute_lut.py by Felekyan et al.
"""

import os
import sys
import glob
import math
import click
import numpy as np

# Required runtime dependencies
try:
    import tttrlib
except Exception as e:
    raise SystemExit("ERROR: tttrlib is required. Install from conda/pip.") from e

try:
    from qtpy import QtWidgets as QtW, QtCore as QtC
    import pyqtgraph as pg
except Exception as e:
    raise SystemExit("ERROR: qtpy/pyqtgraph are required for GUI mode.") from e

# Define the plugin name - this will appear in the Plugins menu
name = "TTTR:Compute Microtime LUT"


# -------------------------
# File and data utilities
# -------------------------
def expand_globs(patterns):
    files = []
    for p in patterns:
        matches = glob.glob(p, recursive=True)
        if not matches:
            click.echo(f"WARNING: pattern matched no files: {p}", err=True)
        files.extend(matches)
    # de-duplicate (preserve order)
    out, seen = [], set()
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def load_microtimes(file_list):
    parts = []
    for fn in file_list:
        click.echo(f"Loading {fn}")
        t = tttrlib.TTTR(fn)
        mt = t.micro_times
        if mt is None or len(mt) == 0:
            click.echo(f"  WARNING: no microtimes in {fn}", err=True)
            continue
        click.echo(f"  {len(mt):,} events")
        parts.append(mt)
    if not parts:
        raise RuntimeError("No microtimes found in any input file.")
    all_micro = np.concatenate(parts)
    click.echo(f"Total microtimes: {len(all_micro):,}")
    return all_micro


def infer_n_bins(micro, n_bins_opt):
    if n_bins_opt and n_bins_opt > 0:
        return int(n_bins_opt)
    # infer from data
    nb = int(np.max(micro)) + 1
    # snap to "nice" powers if very close (e.g. 4096, 8192, 65536)
    for k in (4096, 8192, 16384, 32768, 65536):
        if abs(nb - k) <= max(8, int(0.002 * k)):
            nb = k
            break
    click.echo(f"[auto] inferred n_bins = {nb}")
    return nb


def histogram_micro(micro, n_bins):
    counts, _ = np.histogram(micro, bins=n_bins, range=(0, n_bins))
    return counts


# ----------------------------------
# Auto-detect linear plateau (robust)
# ----------------------------------
def rolling_mean(x, win):
    if win <= 1:
        return x.astype(float)
    c = np.cumsum(np.insert(x, 0, 0))
    out = (c[win:] - c[:-win]) / float(win)
    pad_left = win // 2
    pad_right = len(x) - len(out) - pad_left
    return np.pad(out, (pad_left, pad_right), mode='edge')


def find_longest_true_run(mask):
    best_len, best_start = 0, -1
    cur_len, cur_start = 0, -1
    for i, v in enumerate(mask):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    if best_len == 0:
        return None
    return best_start, best_start + best_len  # [start, stop)


def autodetect_linear_region(
    counts,
    noffset_guess=0,
    tail_exclude_frac=0.0,
    win=31,
    rel_dev_thresh=0.10,
    slope_thresh=0.02,
    min_width=32,
):
    """
    Attempt to find a flat/stable plateau:
    - Ignore early 'noffset_guess' bins and a tail fraction.
    - Use rolling mean and relative deviation & slope.
    - Pick the longest contiguous stable run.

    Returns (linear_start, linear_stop). Can raise ValueError.
    """
    n = len(counts)
    lo = int(np.clip(noffset_guess, 0, n - 2))
    hi = int(np.clip(n - int(n * tail_exclude_frac), lo + 1, n))

    roi = slice(lo, hi)
    c = counts[roi].astype(float)

    mu = rolling_mean(c, win=win)
    mu = np.where(mu <= 0, 1.0, mu)

    rel_dev = np.abs(c / mu - 1.0)
    dmu = np.gradient(mu)
    rel_slope = np.abs(dmu / np.maximum(mu, 1.0))

    stable = (rel_dev <= rel_dev_thresh) & (rel_slope <= slope_thresh)
    run = find_longest_true_run(stable)
    if run is None:
        raise ValueError("no plateau")
    start_rel, stop_rel = run
    if (stop_rel - start_rel) < min_width:
        raise ValueError(f"plateau too short ({stop_rel - start_rel} < {min_width})")

    return lo + start_rel, lo + stop_rel


# ---------------------------------------
# Felekyan et al. linearization (core)
# ---------------------------------------
def build_linearization_table(counts, linear_start, linear_stop, ntac_required, noffset):
    region = counts[linear_start:linear_stop]
    if region.sum() == 0:
        raise ValueError("Chosen linear region has zero counts.")

    n_mean = float(region.mean())

    w = counts.astype(np.float64) / (n_mean if n_mean != 0 else 1.0)
    cum = np.cumsum(w)
    total = cum[-1]
    if total <= 0:
        raise ValueError("Cumulative width is zero.")

    f = float(ntac_required) / float(total)
    ntac_fract = f * cum  # length n_bins; right-edge cumulative positions

    table = {
        "NTAC_fract": ntac_fract,
        "w": w,
        "f": f,
        "n_bins": int(len(counts)),
        "linear_start": int(linear_start),
        "linear_stop": int(linear_stop),
        "ntac_required": int(ntac_required),
        "noffset": int(noffset),
        "n_mean": float(n_mean),
        "total_counts": int(counts.sum()),
    }
    return table


def stochastic_rebin_ntac(raw_micro, ntac_fract, noffset, seed=None, max_photons=None,
                           rounding="ceil", eps=0.0):
    """Apply LUT to raw microtimes and return corrected integer NTAC indices.
    If max_photons is set, use only the first N photons (for preview speed)."""
    rng = np.random.default_rng(seed)
    # optional small epsilon to avoid mapping exactly to the last boundary
    if eps and eps > 0:
        ntac_fract = np.array(ntac_fract, copy=True)
        ntac_fract[-1] = np.nextafter(ntac_fract[-1] - float(eps), -np.inf)
    n_bins = ntac_fract.shape[0]
    raw = raw_micro.astype(np.int64)
    if max_photons is not None and max_photons > 0:
        raw = raw[:max_photons]
    raw = np.clip(raw, 0, n_bins - 1)

    left = np.zeros_like(raw, dtype=np.float64)
    right = ntac_fract[raw].astype(np.float64)
    mask = raw > 0
    left[mask] = ntac_fract[raw[mask] - 1]

    span = right - left
    span = np.where(span > 0, span, 0.0)

    u = rng.random(raw.shape[0])
    frac_pos = left + u * span

    if rounding == "ceil":
        ntac_int = np.ceil(frac_pos).astype(np.int64)
    elif rounding == "floor":
        ntac_int = np.floor(frac_pos).astype(np.int64)
    elif rounding == "stochastic":
        ntac_int = np.floor(frac_pos + rng.random(frac_pos.shape)).astype(np.int64)
    else:
        ntac_int = np.ceil(frac_pos).astype(np.int64)
    ntac = ntac_int - noffset
    return ntac


def save_lut(path, table):
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        data = table["NTAC_fract"].reshape(-1, 1)
        np.savetxt(path, data, fmt="%.9f", header="NTAC_fract")
        click.echo(f"Saved TXT: {path}")

    elif ext == ".csv":
        data = table["NTAC_fract"].reshape(-1, 1)
        np.savetxt(path, data, delimiter=",", fmt="%.9f", header="NTAC_fract")
        click.echo(f"Saved CSV: {path}")

    elif ext == ".npy":
        np.save(path, table["NTAC_fract"])
        click.echo(f"Saved NPY: {path}")

    elif ext == ".npz":
        np.savez_compressed(path, **table)
        click.echo(f"Saved NPZ: {path}")

    else:
        raise click.UsageError(f"Unknown output extension '{ext}'. Use .txt / .csv / .npy / .npz.")


# ---------------------------------------
# GUI (pyqtgraph) for interactive tuning + parameter controls
# ---------------------------------------
class TACLinearizationWidget(QtW.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAC Linearization – Select Linear Region (Felekyan et al.)")
        self.resize(1200, 800)

        # Create advanced parameter widgets (stored in main widget for dialog access)
        self.sp_seed = QtW.QSpinBox()
        self.sp_seed.setRange(0, 2**31 - 1)
        self.sp_seed.setValue(12345)

        self.chk_wrap = QtW.QCheckBox("Mitigate wrap spike (floor + ε)")
        self.chk_wrap.setChecked(False)  # Changed default

        self.sp_eps = QtW.QDoubleSpinBox()
        self.sp_eps.setDecimals(9)
        self.sp_eps.setSingleStep(1e-7)
        self.sp_eps.setRange(0.0, 1e-2)
        self.sp_eps.setValue(1e-6)

        self.sp_thresh = QtW.QDoubleSpinBox()
        self.sp_thresh.setDecimals(6)
        self.sp_thresh.setSingleStep(0.01)
        self.sp_thresh.setRange(0.0, 1e12)
        self.sp_thresh.setValue(0.2)

        # Zero-out controls
        self.chk_brush = QtW.QCheckBox("Enable zero brush")
        self.chk_brush.setChecked(False)
        self.sp_brush = QtW.QSpinBox()
        self.sp_brush.setRange(1, 500)
        self.sp_brush.setValue(5)
        self.chk_erase = QtW.QCheckBox("Eraser mode")

        self.chk_zl = QtW.QCheckBox("Zero left")
        self.sp_zl = QtW.QSpinBox()
        self.sp_zl.setRange(0, 100000)
        self.sp_zl.setValue(0)
        self.chk_zr = QtW.QCheckBox("Zero right")
        self.sp_zr = QtW.QSpinBox()
        self.sp_zr.setRange(0, 100000)
        self.sp_zr.setValue(0)

        self.lst_zero = QtW.QListWidget()
        self.lst_zero.setSelectionMode(QtW.QAbstractItemView.ExtendedSelection)
        self.btn_zero_clear = QtW.QPushButton("Clear all zeroed")

        # Create UI
        self._create_ui()
        self._setup_connections()

        # Initialize UI states
        self.sp_eps.setEnabled(self.chk_wrap.isChecked())

        # Initialize with empty plots
        self._init_plots()

    class AdvancedDialog(QtW.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Advanced Parameters")
            self.setModal(True)
            self.resize(400, 600)

            # Create the advanced parameters UI using parent's widgets
            layout = QtW.QVBoxLayout(self)

            # RNG seed
            seed_group = QtW.QGroupBox("Random Number Generation")
            seed_layout = QtW.QFormLayout(seed_group)
            seed_layout.addRow("RNG seed", parent.sp_seed)
            layout.addWidget(seed_group)

            # Spike mitigation
            spike_group = QtW.QGroupBox("Spike Mitigation")
            spike_layout = QtW.QVBoxLayout(spike_group)
            spike_layout.addWidget(parent.chk_wrap)
            eps_layout = QtW.QFormLayout()
            eps_layout.addRow("ε (wrap)", parent.sp_eps)
            spike_layout.addLayout(eps_layout)
            layout.addWidget(spike_group)

            # Threshold control
            thresh_group = QtW.QGroupBox("Data Thresholding")
            thresh_layout = QtW.QFormLayout(thresh_group)
            thresh_layout.addRow("Low-count threshold", parent.sp_thresh)
            layout.addWidget(thresh_group)

            # Zero-out controls
            zero_group = QtW.QGroupBox("Zero-out bins (cleanup)")
            zero_layout = QtW.QVBoxLayout(zero_group)

            brush_layout = QtW.QHBoxLayout()
            brush_layout.addWidget(parent.chk_brush)
            brush_layout.addWidget(QtW.QLabel("Radius"))
            brush_layout.addWidget(parent.sp_brush)
            brush_layout.addStretch()
            brush_layout.addWidget(parent.chk_erase)

            cuts_layout = QtW.QGridLayout()
            cuts_layout.addWidget(parent.chk_zl, 0, 0)
            cuts_layout.addWidget(parent.sp_zl, 0, 1)
            cuts_layout.addWidget(parent.chk_zr, 1, 0)
            cuts_layout.addWidget(parent.sp_zr, 1, 1)

            zero_layout.addLayout(brush_layout)
            zero_layout.addLayout(cuts_layout)
            zero_layout.addWidget(parent.lst_zero)
            zero_layout.addWidget(parent.btn_zero_clear)

            layout.addWidget(zero_group)

            # OK/Cancel buttons
            buttons = QtW.QDialogButtonBox(QtW.QDialogButtonBox.Ok | QtW.QDialogButtonBox.Cancel)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

    def _show_advanced_dialog(self):
        """Show the advanced parameters dialog."""
        dialog = self.AdvancedDialog(self)
        if dialog.exec() == QtW.QDialog.Accepted:
            # Apply changes
            self._apply_params()
            self._update_plots()

    def _create_ui(self):
        """Create the main UI components."""
        # Central widget
        central_widget = QtW.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtW.QHBoxLayout(central_widget)

        # Left side: plots
        plots_widget = QtW.QWidget()
        plots_layout = QtW.QVBoxLayout(plots_widget)

        self.plt_raw = pg.PlotWidget()
        self.plt_raw.setTitle("Raw TAC histogram (drag the orange region)")
        self.plt_raw.setLabel('left', 'Counts')
        self.plt_raw.setLabel('bottom', 'TAC bin')

        self.plt_after = pg.PlotWidget()
        self.plt_after.setTitle("After linearization (actual corrected preview)")
        self.plt_after.setLabel('left', 'Counts')
        self.plt_after.setLabel('bottom', 'Equal-width bin within one SYNC')

        plots_layout.addWidget(self.plt_raw)
        plots_layout.addWidget(self.plt_after)

        # Right side: controls
        controls_widget = QtW.QWidget()
        controls_layout = QtW.QVBoxLayout(controls_widget)

        # File loading section
        files_group = QtW.QGroupBox("Files")
        files_layout = QtW.QVBoxLayout(files_group)

        self.files_list = QtW.QListWidget()
        self.files_list.setSelectionMode(QtW.QAbstractItemView.ExtendedSelection)
        self.files_list.setAcceptDrops(True)
        self.files_list.installEventFilter(self)

        btn_load = QtW.QPushButton("Load TTTR Files...")
        btn_load.clicked.connect(self._load_files)

        btn_clear = QtW.QPushButton("Clear list")
        btn_clear.clicked.connect(self._clear_files)

        files_layout.addWidget(self.files_list)
        files_layout.addWidget(btn_load)
        files_layout.addWidget(btn_clear)

        # Parameters section
        params_group = QtW.QGroupBox("Parameters")
        params_layout = QtW.QFormLayout(params_group)

        self.lbl_nbins = QtW.QLabel("N/A")
        params_layout.addRow("n_bins (hist)", self.lbl_nbins)

        self.sp_start = QtW.QSpinBox()
        self.sp_start.setRange(0, 100000)
        self.sp_start.setValue(1000)
        params_layout.addRow("linear_start", self.sp_start)

        self.sp_stop = QtW.QSpinBox()
        self.sp_stop.setRange(1, 100000)
        self.sp_stop.setValue(2000)
        params_layout.addRow("linear_stop", self.sp_stop)

        self.sp_ntac = QtW.QSpinBox()
        self.sp_ntac.setRange(2, 1000000)
        self.sp_ntac.setValue(4096)
        params_layout.addRow("ntac_required", self.sp_ntac)

        self.sp_noff = QtW.QSpinBox()
        self.sp_noff.setRange(0, 100000)
        self.sp_noff.setValue(500)
        params_layout.addRow("Noffset", self.sp_noff)

        self.sp_prev = QtW.QSpinBox()
        self.sp_prev.setRange(1000, 10000000)
        self.sp_prev.setValue(500000)
        params_layout.addRow("preview photons", self.sp_prev)

        # Normalization checkbox - now in basic params but default disabled
        self.chk_norm = QtW.QCheckBox("Normalize raw TAC by region mean")
        self.chk_norm.setChecked(False)  # Changed default
        params_layout.addRow("", self.chk_norm)

        # Advanced parameters button
        self.btn_advanced = QtW.QPushButton("Advanced Parameters...")
        self.btn_advanced.clicked.connect(self._show_advanced_dialog)
        params_layout.addRow("", self.btn_advanced)

        # Action buttons
        buttons_layout = QtW.QHBoxLayout()
        self.btn_apply = QtW.QPushButton("Apply params")
        self.btn_save = QtW.QPushButton("Save LUT")
        self.btn_export = QtW.QPushButton("Export corrected…")

        buttons_layout.addWidget(self.btn_apply)
        buttons_layout.addWidget(self.btn_save)
        buttons_layout.addWidget(self.btn_export)

        # Info label
        self.info_label = QtW.QLabel("")
        self.info_label.setWordWrap(True)

        controls_layout.addWidget(files_group)
        controls_layout.addWidget(params_group)
        controls_layout.addLayout(buttons_layout)
        controls_layout.addWidget(self.info_label)
        controls_layout.addStretch()

        main_layout.addWidget(plots_widget, 2)
        main_layout.addWidget(controls_widget, 1)

    def _setup_connections(self):
        """Set up signal connections."""
        # Parameter changes
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

        # Zero-out controls
        self.chk_zl.stateChanged.connect(self._on_zero_controls_changed)
        self.chk_zr.stateChanged.connect(self._on_zero_controls_changed)
        self.sp_zl.valueChanged.connect(self._on_zero_controls_changed)
        self.sp_zr.valueChanged.connect(self._on_zero_controls_changed)
        self.btn_zero_clear.clicked.connect(self._clear_zeroed)

        # Buttons
        self.btn_apply.clicked.connect(self._apply_params)
        self.btn_save.clicked.connect(self._save_lut)
        self.btn_export.clicked.connect(self._export_corrected)

        # Brush controls
        self.chk_brush.stateChanged.connect(self._update_brush_state)

    def _init_plots(self):
        """Initialize empty plots."""
        # Create region selector
        self.region = pg.LinearRegionItem(values=[1000, 2000], brush=(255, 165, 0, 60), movable=True)
        self.plt_raw.addItem(self.region)

        # Create Noffset line
        self.offset_line = pg.InfiniteLine(angle=90, movable=True, pos=500, pen=pg.mkPen((200, 0, 0), width=2))
        self.plt_raw.addItem(self.offset_line)

        # Create threshold line
        self.thresh_line = pg.InfiniteLine(angle=0, movable=True, pos=0.2, pen=pg.mkPen((0, 180, 0), width=2))
        self.plt_raw.addItem(self.thresh_line)

        # Create cut lines
        self.left_cut_line = pg.InfiniteLine(angle=90, movable=True, pos=0, pen=pg.mkPen((120, 120, 120), width=2, style=QtC.Qt.DashLine))
        self.right_cut_line = pg.InfiniteLine(angle=90, movable=True, pos=0, pen=pg.mkPen((120, 120, 120), width=2, style=QtC.Qt.DashLine))

        # Connect signals
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.offset_line.sigPositionChanged.connect(self._on_offset_line_changed)
        self.thresh_line.sigPositionChanged.connect(self._on_thresh_line_changed)
        self.left_cut_line.sigPositionChanged.connect(self._on_left_cut_changed)
        self.right_cut_line.sigPositionChanged.connect(self._on_right_cut_changed)

        # Mouse events for brush
        self.plt_raw.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.plt_raw.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self._brush_active = False

    def _load_files(self):
        """Load TTTR files."""
        dialog = QtW.QFileDialog(self, "Load TTTR Files")
        dialog.setFileMode(QtW.QFileDialog.ExistingFiles)
        dialog.setNameFilter("TTTR files (*.spc *.ht3 *.ptu *.t3r *.t2r);;All files (*.*)")

        if dialog.exec():
            files = dialog.selectedFiles()
            self._process_files(files)

    def _clear_files(self):
        """Clear all files and reset the interface."""
        # Clear file list
        self.files_list.clear()
        self.files = []

        # Reset data
        self.counts = None
        self.micro = None
        self.n_bins = None
        self.zero_mask = None
        self.current_table = None

        # Reset UI elements
        self.lbl_nbins.setText("N/A")
        self.sp_start.setRange(0, 100000)
        self.sp_stop.setRange(1, 100000)
        self.sp_prev.setRange(1000, 10000000)

        # Reset spinbox values to defaults
        self.sp_start.setValue(1000)
        self.sp_stop.setValue(2000)
        self.sp_ntac.setValue(4096)
        self.sp_noff.setValue(500)
        self.sp_prev.setValue(500000)

        # Clear plots
        self.plt_raw.clear()
        self.plt_after.clear()

        # Reset plot titles and labels
        self.plt_raw.setTitle("Raw TAC histogram (drag the orange region)")
        self.plt_raw.setLabel('left', 'Counts')
        self.plt_raw.setLabel('bottom', 'TAC bin')

        self.plt_after.setTitle("After linearization (actual corrected preview)")
        self.plt_after.setLabel('left', 'Counts')
        self.plt_after.setLabel('bottom', 'Equal-width bin within one SYNC')

        # Clear info label
        self.info_label.setText("")

        # Re-add the region selector to empty plot
        self.region.setRegion([1000, 2000])
        self.plt_raw.addItem(self.region)
        self.offset_line.setValue(500)
        self.plt_raw.addItem(self.offset_line)
        self.thresh_line.setValue(0.2)
        self.plt_raw.addItem(self.thresh_line)

    def _process_files(self, files):
        """Process loaded files."""
        try:
            self.files = files
            self.files_list.clear()
            self.files_list.addItems([os.path.basename(f) for f in files])

            self.micro = load_microtimes(files)
            self.n_bins = infer_n_bins(self.micro, None)
            self.counts = histogram_micro(self.micro, self.n_bins)
            self.zero_mask = np.zeros(self.n_bins, dtype=bool)

            # Update UI
            self.lbl_nbins.setText(str(self.n_bins))
            self.sp_start.setRange(0, self.n_bins - 2)
            self.sp_stop.setRange(1, self.n_bins - 1)
            self.sp_prev.setRange(1000, max(1000, len(self.micro)))

            # Set default region
            region = self._pick_initial_region()
            self.sp_start.setValue(region[0])
            self.sp_stop.setValue(region[1])
            self.region.setRegion(region)

            # Update plots
            self._update_plots()
            self._apply_params()

        except Exception as e:
            QtW.QMessageBox.critical(self, "Error", f"Failed to load files: {str(e)}")

    def _pick_initial_region(self):
        """Pick initial linear region."""
        if self.counts is None:
            return (1000, 2000)

        n = len(self.counts)
        nz = np.where(self.counts > 0)[0]
        if nz.size < 4:
            a = max(0, n // 4)
            b = min(n, a + max(32, n // 10))
            return (a, b)

        lo = nz[0]
        hi = nz[-1] + 1
        width = max(32, (hi - lo) // 5)
        start = lo + (hi - lo - width) // 2
        stop = start + width
        return (start, stop)

    def _counts_effective(self):
        """Get effective counts after applying zero masks and cuts."""
        if self.counts is None:
            return None

        ce = self.counts.copy()
        if self.zero_mask is not None:
            ce[self.zero_mask] = 0

        if self.chk_zl.isChecked():
            zl = self.sp_zl.value()
            ce[:zl] = 0

        if self.chk_zr.isChecked():
            zr = self.sp_zr.value()
            ce[max(0, zr):] = 0

        # Apply threshold
        s = self.sp_start.value()
        e = self.sp_stop.value()
        if self.chk_norm.isChecked():
            reg = ce[s:e]
            nmean = float(reg.mean()) if reg.size > 0 else 1.0
            scale = nmean if nmean > 0 else 1.0
            y_disp = ce / scale
        else:
            y_disp = ce.astype(float)

        thr = self.sp_thresh.value()
        ce[y_disp < thr] = 0

        return ce

    def _update_plots(self):
        """Update the raw TAC plot."""
        if self.counts is None:
            return

        self.plt_raw.clear()
        ce = self._counts_effective()

        s = self.sp_start.value()
        e = self.sp_stop.value()

        if self.chk_norm.isChecked():
            reg = ce[s:e]
            nmean = float(reg.mean()) if reg.size > 0 else 1.0
            scale = nmean if nmean > 0 else 1.0
            y_disp = ce / scale
            self.plt_raw.setLabel('left', 'Counts / ⟨counts⟩_region')
            one_line = pg.InfiniteLine(angle=0, movable=False, pos=1.0, pen=pg.mkPen((150, 150, 150), width=1, style=QtC.Qt.DashLine))
            self.plt_raw.addItem(one_line)
        else:
            y_disp = ce.astype(float)
            self.plt_raw.setLabel('left', 'Counts')

        x = np.arange(self.n_bins)
        self.plt_raw.plot(x, y_disp, pen=pg.mkPen((255, 204, 0), width=1.5))

        # Add region
        self.plt_raw.addItem(self.region)

        # Add zero overlays
        if self.zero_mask is not None:
            ranges = self._mask_to_ranges(self.zero_mask)
            for a, b in ranges:
                zr = pg.LinearRegionItem(values=[a, b], brush=(255, 0, 0, 60), movable=False)
                zr.setZValue(5)
                self.plt_raw.addItem(zr)

        # Add threshold zeroed
        thr = self.sp_thresh.value()
        below = y_disp < thr
        ranges = self._mask_to_ranges(below)
        for a, b in ranges:
            zb = pg.LinearRegionItem(values=[a, b], brush=(0, 0, 255, 40), movable=False)
            zb.setZValue(4)
            self.plt_raw.addItem(zb)

        # Add cut lines and shaded areas
        if self.chk_zl.isChecked():
            zl = self.sp_zl.value()
            self.left_cut_line.setValue(zl)
            self.plt_raw.addItem(self.left_cut_line)
            if zl > 0:
                self.plt_raw.addItem(pg.LinearRegionItem(values=[0, zl], brush=(120, 120, 120, 40), movable=False))

        if self.chk_zr.isChecked():
            zr = self.sp_zr.value()
            self.right_cut_line.setValue(zr)
            self.plt_raw.addItem(self.right_cut_line)
            if zr < self.n_bins - 1:
                self.plt_raw.addItem(pg.LinearRegionItem(values=[zr, self.n_bins], brush=(120, 120, 120, 40), movable=False))

        # Add offset and threshold lines
        self.offset_line.setValue(self.sp_noff.value())
        self.plt_raw.addItem(self.offset_line)

        self.thresh_line.setValue(thr)
        self.plt_raw.addItem(self.thresh_line)

        # Update zero list
        self._update_zero_list()

    def _mask_to_ranges(self, mask):
        """Convert boolean mask to ranges."""
        ranges = []
        i = 0
        N = len(mask)
        while i < N:
            if mask[i]:
                j = i + 1
                while j < N and mask[j]:
                    j += 1
                ranges.append((i, j))
                i = j
            else:
                i += 1
        return ranges

    def _update_zero_list(self):
        """Update the zero ranges list."""
        self.lst_zero.clear()
        if self.zero_mask is not None:
            ranges = self._mask_to_ranges(self.zero_mask)
            for a, b in ranges:
                self.lst_zero.addItem(f"[{a}, {b})")

    def _apply_params(self):
        """Apply current parameters and compute LUT."""
        if self.counts is None:
            return

        try:
            ce = self._counts_effective()
            s = self.sp_start.value()
            e = self.sp_stop.value()

            self.current_table = build_linearization_table(
                ce, s, e, self.sp_ntac.value(), self.sp_noff.value()
            )

            self.info_label.setText(
                f"Range [{self.current_table['linear_start']}, {self.current_table['linear_stop']}) | "
                f"width={self.current_table['linear_stop'] - self.current_table['linear_start']} | "
                f"f={self.current_table['f']:.6f} | n_mean={self.current_table['n_mean']:.2f}"
            )

            self._update_after_plot()

        except Exception as e:
            self.info_label.setText(f"Error: {str(e)}")
            self.current_table = None

    def _update_after_plot(self):
        """Update the after-linearization plot."""
        if self.current_table is None:
            return

        self.plt_after.clear()

        corr = stochastic_rebin_ntac(
            self.micro,
            self.current_table["NTAC_fract"],
            self.current_table["noffset"],
            seed=self.sp_seed.value(),
            max_photons=self.sp_prev.value(),
            rounding=("floor" if self.chk_wrap.isChecked() else "ceil"),
            eps=(self.sp_eps.value() if self.chk_wrap.isChecked() else 0.0),
        )

        nt_mod = (corr % self.sp_ntac.value()).astype(int)
        hist_corr, _ = np.histogram(nt_mod, bins=self.sp_ntac.value(), range=(0, self.sp_ntac.value()))

        self.plt_after.plot(
            np.arange(self.sp_ntac.value()),
            hist_corr,
            pen=pg.mkPen((80, 200, 255), width=1.5),
        )
        self.plt_after.setXRange(0, self.sp_ntac.value(), padding=0)

    def _save_lut(self):
        """Save the current LUT."""
        if self.current_table is None:
            QtW.QMessageBox.warning(self, "No LUT", "Compute a LUT first.")
            return

        dialog = QtW.QFileDialog(self, "Save LUT")
        dialog.setAcceptMode(QtW.QFileDialog.AcceptSave)
        dialog.setNameFilters([
            "Text (*.txt)",
            "CSV (*.csv)",
            "NumPy binary (*.npy)",
            "Compressed NPZ (*.npz)",
        ])

        if dialog.exec():
            path = dialog.selectedFiles()[0]
            try:
                save_lut(path, self.current_table)
                QtW.QMessageBox.information(self, "Saved", f"LUT saved to {path}")
            except Exception as e:
                QtW.QMessageBox.critical(self, "Save failed", str(e))

    def _export_corrected(self):
        """Export corrected microtimes."""
        if self.current_table is None:
            QtW.QMessageBox.warning(self, "No LUT", "Compute a LUT first.")
            return

        dialog = QtW.QFileDialog(self, "Export Corrected Microtimes")
        dialog.setAcceptMode(QtW.QFileDialog.AcceptSave)
        dialog.setNameFilters([
            "NumPy binary (*.npy)",
            "Compressed NPZ (*.npz)",
            "CSV (*.csv)",
            "Text (*.txt)",
        ])

        if dialog.exec():
            path = dialog.selectedFiles()[0]
            try:
                corr = stochastic_rebin_ntac(
                    self.micro,
                    self.current_table["NTAC_fract"],
                    self.current_table["noffset"],
                    seed=self.sp_seed.value(),
                    rounding=("floor" if self.chk_wrap.isChecked() else "ceil"),
                    eps=(self.sp_eps.value() if self.chk_wrap.isChecked() else 0.0)
                )

                ext = os.path.splitext(path)[1].lower()
                if ext == ".npy":
                    np.save(path, corr)
                elif ext == ".npz":
                    np.savez_compressed(path, corrected_ntac=corr)
                elif ext == ".csv":
                    np.savetxt(path, corr, fmt="%d", delimiter=",")
                elif ext == ".txt":
                    np.savetxt(path, corr, fmt="%d")
                else:
                    np.save(path, corr)

                QtW.QMessageBox.information(self, "Exported", f"Corrected microtimes saved to {path}")
            except Exception as e:
                QtW.QMessageBox.critical(self, "Export failed", str(e))

    # Event handlers
    def _on_params_changed(self):
        self._update_plots()
        self._apply_params()

    def _on_wrap_changed(self):
        self.sp_eps.setEnabled(self.chk_wrap.isChecked())
        self._apply_params()

    def _on_region_changed(self):
        s, e = [int(v) for v in self.region.getRegion()]
        self.sp_start.blockSignals(True)
        self.sp_stop.blockSignals(True)
        self.sp_start.setValue(s)
        self.sp_stop.setValue(e)
        self.sp_start.blockSignals(False)
        self.sp_stop.blockSignals(False)
        self._on_params_changed()

    def _on_offset_line_changed(self):
        val = int(round(self.offset_line.value()))
        val = max(0, min(self.n_bins - 1, val))
        if self.sp_noff.value() != val:
            self.sp_noff.setValue(val)

    def _on_thresh_line_changed(self):
        val = self.thresh_line.value()
        if self.sp_thresh.value() != val:
            self.sp_thresh.setValue(val)

    def _on_left_cut_changed(self):
        val = int(round(self.left_cut_line.value()))
        val = max(0, min(self.n_bins - 1, val))
        if self.sp_zl.value() != val:
            self.sp_zl.setValue(val)

    def _on_right_cut_changed(self):
        val = int(round(self.right_cut_line.value()))
        val = max(0, min(self.n_bins - 1, val))
        if self.sp_zr.value() != val:
            self.sp_zr.setValue(val)

    def _on_zero_controls_changed(self):
        self._update_plots()
        self._apply_params()

    def _clear_zeroed(self):
        if self.zero_mask is not None:
            self.zero_mask[:] = False
        self._update_plots()
        self._apply_params()

    def _update_brush_state(self):
        pass  # Brush functionality can be implemented if needed

    def _on_mouse_clicked(self, event):
        pass  # Brush functionality can be implemented if needed

    def _on_mouse_moved(self, pos):
        pass  # Brush functionality can be implemented if needed

    def eventFilter(self, obj, event):
        """Handle drag and drop for file list."""
        if obj is self.files_list:
            if event.type() == QtC.QEvent.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            elif event.type() == QtC.QEvent.Drop:
                if event.mimeData().hasUrls():
                    paths = []
                    for url in event.mimeData().urls():
                        try:
                            p = str(url.toLocalFile())
                            if p:
                                paths.append(p)
                        except:
                            pass
                    if paths:
                        self._process_files(paths)
                    event.acceptProposedAction()
                    return True
        return super().eventFilter(obj, event)


# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the TACLinearizationWidget class
    window = TACLinearizationWidget()
    # Show the window
    window.show()
