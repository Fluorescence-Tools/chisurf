import sys
import numpy as np
import tttrlib

name = "Single-Molecule:FIDA2D"

"""
2D Fluorescence Intensity Distribution Analysis (FIDA2D)

This plugin provides a graphical interface for analyzing the joint distribution of 
fluorescence intensities in two detection channels. Features include:

- Loading and processing of photon data from TTTR files
- Calculation of 2D photon count histograms
- Fitting of theoretical models to experimental data
- Determination of concentrations and specific brightnesses of fluorescent species
- Visualization of results with interactive 2D plots

FIDA2D extends the capabilities of standard FIDA by analyzing correlations between 
different detection channels, making it powerful for studying species with different 
brightness characteristics in multiple channels, such as FRET-labeled molecules.
"""
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout,
    QFormLayout, QScrollArea, QLineEdit, QTextEdit, QLabel, QPushButton,
    QFileDialog, QMessageBox, QDoubleSpinBox, QSpinBox, QProgressDialog
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from scipy.optimize import least_squares
from numpy.fft import fft2, ifft2

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LogNorm

# --- FIDA Theory Functions ---
# Eq. (7) in Kask et al., 2000

def dVdx(x, a1=-0.4, a2=0.08):
    return np.exp(-x) * (1 + a1 * x + a2 * x**2)

# Eq. (6) in Kask et al., 2000

def compute_generating_function(lam1, lam2, T, species_params, x_vals, dx, bg1=0.0, bg2=0.0):
    G_log = - (lam1 - 1) * bg1 * T - (lam2 - 1) * bg2 * T
    for c, q1, q2 in species_params:
        for x in x_vals:
            B = np.exp(-x)
            weight = dVdx(x) * dx
            term = np.exp((lam1 - 1) * q1 * B * T + (lam2 - 1) * q2 * B * T) - 1
            G_log += c * weight * term
    return np.exp(G_log)

# Generate Fourier-domain grid

def generate_lambda_grid(N1, N2):
    l1 = np.exp(2j * np.pi * np.arange(N1) / N1)
    l2 = np.exp(2j * np.pi * np.arange(N2) / N2)
    return np.meshgrid(l1, l2, indexing='ij')

# Compute joint distribution and normalize

def compute_joint_distribution(T, shape, species_params, x_vals, dx, bg1=0.0, bg2=0.0):
    N1, N2 = shape
    lam1, lam2 = generate_lambda_grid(N1, N2)
    G = compute_generating_function(lam1, lam2, T, species_params, x_vals, dx, bg1, bg2)
    P = np.real(ifft2(G))
    return np.maximum(P, 0) / np.sum(P)

class FIDAApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D-FIDA App")
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)

        # Plot widgets
        self.trace_plot = pg.PlotWidget(title="Intensity Trace")
        grid.addWidget(self.trace_plot, 0, 0)

        # Matplotlib Canvas for 2D histogram
        self.hist_fig = plt.Figure()
        self.hist_canvas = FigureCanvas(self.hist_fig)
        self.hist_ax = self.hist_fig.add_subplot(111)
        grid.addWidget(self.hist_canvas, 1, 0)

        # Control panel
        ctrl = QWidget()
        vbox = QVBoxLayout(ctrl)

        form = QFormLayout()
        # File input
        self.le_file = QLineEdit(); self.le_file.setReadOnly(True)
        btn_load = QPushButton("Load TTTR File")
        btn_load.clicked.connect(self.load_file)
        form.addRow("TTTR File:", self.le_file)
        form.addRow(btn_load)

        # Channels input
        self.le_ch1 = QLineEdit("0 8")
        self.le_ch2 = QLineEdit("2 9")
        form.addRow("Channels ch1 (space-separated):", self.le_ch1)
        form.addRow("Channels ch2 (space-separated):", self.le_ch2)

        # Bin time
        self.spin_bin = QDoubleSpinBox()
        self.spin_bin.setRange(0.1, 1000.0)
        self.spin_bin.setSingleStep(0.01)
        self.spin_bin.setSuffix(" ms")
        self.spin_bin.setValue(0.05)
        form.addRow("Bin Time:", self.spin_bin)

        # Species count
        self.spin_species = QSpinBox()
        self.spin_species.setRange(1, 10)
        self.spin_species.setValue(2)
        self.spin_species.valueChanged.connect(self.update_species_inputs)
        form.addRow("Species Count:", self.spin_species)

        # Species parameters area
        self.species_area = QScrollArea()
        self.species_widget = QWidget()
        self.species_layout = QFormLayout(self.species_widget)
        self.species_area.setWidgetResizable(True)
        self.species_area.setWidget(self.species_widget)
        self.update_species_inputs(self.spin_species.value())

        # Compute histogram and fit buttons
        btn_hist = QPushButton("Compute Experimental Histogram")
        btn_hist.clicked.connect(self.compute_experimental_hist2d)
        self.btn_fit = QPushButton("Fit 2D-FIDA")
        self.btn_fit.clicked.connect(self.fit_fida)
        self.results_edit = QTextEdit(); self.results_edit.setReadOnly(True)

        vbox.addLayout(form)
        vbox.addWidget(self.species_area)
        vbox.addWidget(btn_hist)
        vbox.addWidget(self.btn_fit)
        vbox.addWidget(QLabel("Fit Info:"))
        vbox.addWidget(self.results_edit)
        grid.addWidget(ctrl, 0, 1, 2, 1)

    def update_species_inputs(self, count):
        count = int(count)
        for i in reversed(range(self.species_layout.count())):
            self.species_layout.removeRow(i)
        self.c_boxes, self.q1_boxes, self.q2_boxes = [], [], []
        for i in range(count):
            cb = QDoubleSpinBox(); cb.setValue(1.0)
            q1b = QDoubleSpinBox(); q1b.setRange(0, 1e6); q1b.setValue(50e3)
            q2b = QDoubleSpinBox(); q2b.setRange(0, 1e6); q2b.setValue(30e3)
            self.c_boxes.append(cb)
            self.q1_boxes.append(q1b)
            self.q2_boxes.append(q2b)
            self.species_layout.addRow(f"Species {i+1} Conc:", cb)
            self.species_layout.addRow(f"Species {i+1} q1:", q1b)
            self.species_layout.addRow(f"Species {i+1} q2:", q2b)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open TTTR", "", "TTTR (*.ptu *.ht3 *.t2r *.t3r)")
        if path:
            try:
                self.tttr = tttrlib.TTTR(path)
                self.le_file.setText(path)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _plot_log_imshow(self, data):
        # replace zeros with small epsilon for log scale
        eps = np.nanmin(data[np.nonzero(data)]) / 10.0
        data = data.copy()
        data[data == 0] = eps
        # clear entire figure and reset axis to avoid stacking
        self.hist_fig.clear()
        self.hist_ax = self.hist_fig.add_subplot(111)
        im = self.hist_ax.imshow(data, origin='lower', aspect='auto', norm=LogNorm(vmin=eps))
        self._cbar = self.hist_fig.colorbar(im, ax=self.hist_ax)
        self.hist_canvas.draw()

    def compute_experimental_hist2d(self):
        if not hasattr(self, 'tttr'):
            QMessageBox.warning(self, "Error", "Load a TTTR file first.")
            return
        try:
            chans1 = set(map(int, self.le_ch1.text().split()))
            chans2 = set(map(int, self.le_ch2.text().split()))
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid channel lists.")
            return
        bin_width = self.spin_bin.value() * 1e-3
        times = self.tttr.macro_times * self.tttr.header.macro_time_resolution
        routing = self.tttr.routing_channels
        idxs = (times // bin_width).astype(np.int64)
        num_windows = idxs.max() + 1
        mask1 = np.isin(routing, list(chans1))
        mask2 = np.isin(routing, list(chans2))
        c1_counts = np.bincount(idxs[mask1], minlength=num_windows)
        c2_counts = np.bincount(idxs[mask2], minlength=num_windows)

        # update trace plot
        self.trace_plot.clear()
        tcent = np.arange(num_windows) * bin_width
        max_pts = 1000
        if num_windows > max_pts:
            sel = np.linspace(0, num_windows - 1, max_pts, dtype=int)
            self.trace_plot.plot(tcent[sel], c1_counts[sel], pen='g')
            self.trace_plot.plot(tcent[sel], c2_counts[sel], pen='r')
        else:
            self.trace_plot.plot(tcent, c1_counts, pen='g')
            self.trace_plot.plot(tcent, c2_counts, pen='r')

        # compute and plot histogram
        hist2d, xedges, yedges = np.histogram2d(c1_counts, c2_counts, bins=[64, 64])
        self.hist2d = hist2d / hist2d.sum()
        self._plot_log_imshow(self.hist2d.T)

    def fit_fida(self):
        if not hasattr(self, 'hist2d'):
            QMessageBox.warning(self, "Error", "Compute the experimental histogram first.")
            return
        T = self.spin_bin.value() * 1e-3
        x_vals = np.linspace(0, 5, 200)
        dx = x_vals[1] - x_vals[0]
        comps = int(self.spin_species.value())
        init = [cb.value() for cb in self.c_boxes] + [q1.value() for q1 in self.q1_boxes] + [q2.value() for q2 in self.q2_boxes]
        def model(p):
            c = p[:comps]
            q1 = p[comps:2*comps]
            q2 = p[2*comps:3*comps]
            species = list(zip(c, q1, q2))
            return compute_joint_distribution(T, (64, 64), species, x_vals, dx)
        def resid(p):
            return (model(p) - self.hist2d).ravel()

        res = least_squares(resid, init, bounds=(0, np.inf))
        fit = model(res.x)
        self._plot_log_imshow(fit.T)

        lines = [f"Species {i+1}: c={res.x[i]:.3f}, q1={res.x[comps+i]:.1f}, q2={res.x[2*comps+i]:.1f}" for i in range(comps)]
        self.results_edit.setPlainText("\n".join(lines))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FIDAApp()
    window.show()
    sys.exit(app.exec_())


name = "Single-Molecule:FIDA-2D"


if __name__ == "plugin":
    window = FIDAApp()
    window.show()
