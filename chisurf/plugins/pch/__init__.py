"""
Photon Counting Histogram (PCH) Analysis

This plugin provides tools for analyzing the distribution of photon counts in
fluorescence time traces. PCH analysis can reveal information about:
- Molecular brightness (ε)
- Number of molecules in the detection volume (⟨N⟩)
- Presence of multiple species with different brightness values

The plugin supports loading TTTR files, calculating PCH histograms, and fitting
them with theoretical models for single or multiple species.
"""

import sys
import numpy as np
import tttrlib
from scipy.stats import poisson
from scipy.signal import fftconvolve
from scipy.optimize import least_squares
from numba import njit
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout,
    QGroupBox, QPushButton, QFileDialog, QLabel, QMessageBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QScrollArea, QLineEdit, QTextEdit
)
import pyqtgraph as pg

name = "Single-Molecule:Photon Counting Histogram"


@njit(fastmath=True)
def compute_p1(k_vals, brightness, x_vals, dx):
    n = k_vals.shape[0]
    p1 = np.zeros(n, np.float64)
    for i in range(1, n):
        k = int(k_vals[i])
        fact = 1.0
        for j in range(1, k + 1):
            fact *= j
        total = 0.0
        for xi in x_vals:
            exp_term = np.exp(-2.0 * xi * xi)
            total += (brightness * exp_term) ** k / fact * np.exp(-brightness * exp_term)
        p1[i] = total * dx
    p1[0] = 1.0 - p1[1:].sum()
    return p1

def pch_single_species(k_vals, brightness):
    x_vals = np.linspace(0, 5, 1000)
    dx = x_vals[1] - x_vals[0]
    return compute_p1(k_vals, brightness, x_vals, dx)

@njit(fastmath=True)
def convolve_pch_numba(p1, N, length):
    pk = np.zeros(length, np.float64)
    if N == 0:
        pk[0] = 1.0
        return pk
    temp = p1.copy()
    for _ in range(1, N):
        out = np.zeros(length, np.float64)
        for i in range(length):
            for j in range(length - i):
                out[i + j] += temp[i] * p1[j]
        temp = out
    return temp

def pch_open_system(k_vals, brightness, avgN, maxN=30):
    p1 = pch_single_species(k_vals, brightness)
    length = k_vals.shape[0]
    pk_tot = np.zeros(length)
    for N in range(maxN + 1):
        pk_tot += poisson.pmf(N, avgN) * convolve_pch_numba(p1, N, length)
    return pk_tot

def pch_mixture(k_vals, epsilons, avgNs):
    pk = np.zeros_like(k_vals, dtype=float)
    pk[0] = 1.0
    for eps, n in zip(epsilons, avgNs):
        pj = pch_open_system(k_vals, eps, n)
        pk = fftconvolve(pk, pj)[:len(k_vals)]
    return pk

class PCHApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(name)
        self.resize(900, 600)
        self.last_params = None
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        self.spin_mt_min = QSpinBox()
        self.spin_mt_min.setRange(0, 65535)
        self.spin_mt_min.setValue(0)
        self.spin_mt_max = QSpinBox()
        self.spin_mt_max.setRange(0, 65535)
        self.spin_mt_max.setValue(65535)

        # Plot area on left
        plot_container = QWidget()
        vplot = QVBoxLayout(plot_container)
        self.trace_plot = pg.PlotWidget(title="Intensity Trace")
        self.trace_plot.setLabel("bottom", "Time (s)")
        self.trace_plot.setLabel("left", "Photon Counts")
        self.hist_plot = pg.PlotWidget(title="Photon Counting Histogram")
        self.hist_plot.setLogMode(y=True)
        self.hist_plot.setLabel("bottom", "Photon Count k")
        self.hist_plot.setLabel("left", "P(k)")
        # allow each handle to move on its own rather than dragging the whole block
        self.region = pg.LinearRegionItem([0, 1], swapMode='handle')
        self.region.sigRegionChanged.connect(self.update_region_info)
        self.hist_plot.addItem(self.region)
        vplot.addWidget(self.trace_plot)
        vplot.addWidget(self.hist_plot)
        grid.addWidget(plot_container, 0, 0, 2, 1)

        # Data Settings on right top
        grp_data = QGroupBox("Data Settings")
        data_layout = QGridLayout(grp_data)
        self.le_file = QLineEdit()
        self.le_file.setReadOnly(True)
        self.btn_load = QPushButton("Load TTTR File")
        self.btn_load.clicked.connect(self.load_file)
        data_layout.addWidget(self.le_file, 0, 0)
        data_layout.addWidget(self.btn_load, 0, 1)
        data_layout.addWidget(QLabel("Channels:"), 1, 0)
        self.le_ch = QLineEdit("0,2")
        data_layout.addWidget(self.le_ch, 1, 1)
        data_layout.addWidget(QLabel("Bin Time (μs):"), 2, 0)
        self.spin_bin = QDoubleSpinBox()
        self.spin_bin.setRange(0.1, 1e6)
        self.spin_bin.setValue(100.0)
        self.spin_bin.setSuffix(" μs")
        data_layout.addWidget(self.spin_bin, 2, 1)
        data_layout.addWidget(QLabel("Micro Time Range:"), 3, 0)
        mt_range_layout = QHBoxLayout()
        mt_range_layout.addWidget(self.spin_mt_min)
        mt_range_layout.addWidget(QLabel("to"))
        mt_range_layout.addWidget(self.spin_mt_max)
        data_layout.addLayout(mt_range_layout, 3, 1)
        self.btn_trace = QPushButton("Compute Trace & PCH")
        self.btn_trace.clicked.connect(self.compute_trace_pch)
        data_layout.addWidget(self.btn_trace, 5, 0, 1, 2)
        self.btn_save = QPushButton("Save Results")
        self.btn_save.clicked.connect(self.save_results)
        data_layout.addWidget(self.btn_save, 4, 0, 1, 2)
        grid.addWidget(grp_data, 0, 1)

        # Model Fit on right bottom
        grp_fit = QGroupBox("Model Fit")
        vfit = QVBoxLayout(grp_fit)
        ffit = QFormLayout()
        self.spin_comp = QSpinBox()
        self.spin_comp.setRange(1, 10)
        self.spin_comp.setValue(1)
        self.spin_comp.valueChanged.connect(self.update_species_inputs)
        ffit.addRow("Components:", self.spin_comp)
        vfit.addLayout(ffit)
        self.species_area = QScrollArea()
        self.species_area.setWidgetResizable(True)
        self.species_widget = QWidget()
        self.species_layout = QFormLayout(self.species_widget)
        self.species_area.setWidget(self.species_widget)
        vfit.addWidget(self.species_area)
        self.update_species_inputs(1)
        self.btn_fit = QPushButton("Fit PCH")
        self.btn_fit.clicked.connect(self.fit_pch)
        vfit.addWidget(self.btn_fit)
        vfit.addWidget(QLabel("Fit Results:"))
        self.results_edit = QTextEdit()
        self.results_edit.setReadOnly(True)
        vfit.addWidget(self.results_edit)
        grid.addWidget(grp_fit, 1, 1)

        # Central grid
        grid = QGridLayout(central)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(0)

        # Plot area
        vplot = QVBoxLayout(plot_container)
        vplot.setContentsMargins(0, 0, 0, 0)
        vplot.setSpacing(0)

        # Data settings group
        data_layout = QGridLayout(grp_data)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setHorizontalSpacing(0)
        data_layout.setVerticalSpacing(0)

        # Fit group
        vfit = QVBoxLayout(grp_fit)
        vfit.setContentsMargins(0, 0, 0, 0)
        vfit.setSpacing(0)

        ffit = QFormLayout()
        ffit.setContentsMargins(0, 0, 0, 0)
        ffit.setSpacing(0)

        # Species inputs
        self.species_layout.setContentsMargins(0, 0, 0, 0)
        self.species_layout.setSpacing(0)

    def update_species_inputs(self, count):
        old_eps = [b.value() for b in getattr(self, 'eps_boxes', [])]
        old_Ns = [b.value() for b in getattr(self, 'N_boxes', [])]
        for i in reversed(range(self.species_layout.count())):
            self.species_layout.removeRow(i)
        self.eps_boxes = []
        self.N_boxes = []
        for i in range(count):
            eb = QDoubleSpinBox()
            eb.setRange(0, 1e6)
            eb.setSingleStep(0.1)
            eb.setValue(old_eps[i] if i < len(old_eps) else 2.0)
            nb = QDoubleSpinBox()
            nb.setRange(0, 1e6)
            nb.setSingleStep(0.1)
            nb.setValue(old_Ns[i] if i < len(old_Ns) else 3.0)
            self.eps_boxes.append(eb)
            self.N_boxes.append(nb)
            self.species_layout.addRow(f"ε {i + 1}", eb)
            self.species_layout.addRow(f"⟨N⟩ {i + 1}", nb)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open TTTR", "", "TTTR (*.ptu *.ht3 *.t2r *.t3r)")
        if path:
            try:
                self.tttr = tttrlib.TTTR(path)
                self.le_file.setText(path)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def compute_trace_pch(self):
        if not hasattr(self, 'tttr') or self.tttr is None:
            QMessageBox.warning(self, "No File", "Load TTTR first.")
            return
        txt = self.le_ch.text().strip()
        try:
            channels = list(map(int, txt.split(','))) if txt else np.unique(self.tttr.routing_channels)
        except:
            QMessageBox.warning(self, "Error", "Invalid channels.")
            return
        mask = np.isin(self.tttr.routing_channels, channels)
        bin_t = self.spin_bin.value() * 1e-6
        masks_mt = (self.tttr.micro_times >= self.spin_mt_min.value()) & (self.tttr.micro_times <= self.spin_mt_max.value())
        combined_mask = mask & masks_mt
        times = self.tttr.macro_times[combined_mask] * self.tttr.header.macro_time_resolution
        counts, edges = np.histogram(times, bins=int(np.ceil(times.max() / bin_t)),
                                     range=(0, bin_t * int(np.ceil(times.max() / bin_t))))
        tcent = (edges[:-1] + edges[1:]) / 2
        self.trace_plot.clear()
        self.trace_plot.plot(tcent, counts, stepMode=False)
        self.hist_counts = np.bincount(counts, minlength=counts.max() + 1)
        self.total_bins = counts.size
        self.k_vals = np.arange(self.hist_counts.size)
        self.p_exp = self.hist_counts / self.total_bins
        self.hist_plot.clear()
        self.hist_plot.addItem(self.region)
        self.hist_plot.plot(self.k_vals, self.p_exp, pen=None, symbol='o')
        self.region.setRegion([0, self.k_vals.max()])

    def fit_pch(self):
        if not hasattr(self, 'p_exp'):
            QMessageBox.warning(self, "No Data", "Compute trace first.")
            return
        comps = self.spin_comp.value()
        params = [b.value() for b in self.eps_boxes] + [b.value() for b in self.N_boxes]
        reg_min, reg_max = self.region.getRegion()
        low, high = int(np.ceil(reg_min)), int(np.floor(reg_max))
        mask_reg = (self.k_vals >= low) & (self.k_vals <= high)

        def resid(p):
            pmod = pch_mixture(self.k_vals, p[:comps], p[comps:])
            pmod /= pmod.sum()
            return pmod[mask_reg] - self.p_exp[mask_reg]

        res = least_squares(resid, np.array(params), bounds=(0, np.inf))
        self.last_params = res.x
        for i in range(comps):
            self.eps_boxes[i].setValue(res.x[i])
            self.N_boxes[i].setValue(res.x[comps + i])
        p_fit = pch_mixture(self.k_vals, res.x[:comps], res.x[comps:])
        p_fit /= p_fit.sum()
        exp_cnt = p_fit * self.total_bins
        obs_cnt = self.hist_counts
        mask_chi = mask_reg & (exp_cnt > 0)
        chi2 = np.sum((obs_cnt[mask_chi] - exp_cnt[mask_chi]) ** 2 / exp_cnt[mask_chi])
        dof = mask_chi.sum() - (comps * 2)
        red_chi2 = chi2 / dof if dof > 0 else np.nan
        self.hist_plot.clear()
        self.hist_plot.addItem(self.region)
        self.hist_plot.plot(self.k_vals, obs_cnt / self.total_bins, pen=None, symbol='o')
        self.hist_plot.plot(self.k_vals[mask_reg], p_fit[mask_reg], pen=pg.mkPen('r', width=2))


        comps = self.spin_comp.value()
        eps = res.x[:comps]
        Ns = res.x[comps:]
        # compute percent fractions by molecule count
        fractions = Ns / Ns.sum() * 100.0

        # a brief description of your two fit parameters
        legend = (
            "ε (epsilon): molecular brightness parameter\n"
            "⟨N⟩: average number of molecules in the detection volume\n"
            "x: species fraction\n"
        )

        # build your lines
        lines = [f"Region: {low}–{high}"]
        for i in range(comps):
            lines.append(
                f"Comp{i + 1}: ε={eps[i]:.3f}, ⟨N⟩={Ns[i]:.3f}, "
                f"x={fractions[i]:.1f}%"
            )
        lines += [f"Chi2: {chi2:.3f}", f"Reduced Chi2: {red_chi2:.3f} (dof={dof})"]

        self.results_edit.setPlainText(legend + "\n".join(lines))

    def update_region_info(self):
        if self.last_params is None:
            return
        comps = self.spin_comp.value()
        params = self.last_params
        p_fit = pch_mixture(self.k_vals, params[:comps], params[comps:])
        p_fit /= p_fit.sum()
        exp_cnt = p_fit * self.total_bins
        obs_cnt = self.hist_counts
        reg_min, reg_max = self.region.getRegion()
        low, high = int(np.ceil(reg_min)), int(np.floor(reg_max))
        mask_reg = (self.k_vals >= low) & (self.k_vals <= high)
        mask_chi = mask_reg & (exp_cnt > 0)
        chi2 = np.sum((obs_cnt[mask_chi] - exp_cnt[mask_chi]) ** 2 / exp_cnt[mask_chi])
        dof = mask_chi.sum() - (comps * 2)
        red_chi2 = chi2 / dof if dof > 0 else np.nan
        lines = [f"Region: {low}–{high}"] + [
            f"Comp{i + 1}: ε={params[i]:.3f}, ⟨N⟩={params[comps + i]:.3f}" for i in range(comps)
        ] + [f"Chi2: {chi2:.3f}", f"Reduced Chi2: {red_chi2:.3f} (dof={dof})"]
        # a brief description of your two fit parameters
        legend = (
            "ε (epsilon): molecular brightness parameter\n"
            "⟨N⟩: average number of molecules in the detection volume\n\n"
        )
        self.results_edit.setPlainText(legend + "\n".join(lines))

    def save_results(self):
        if not hasattr(self, 'p_exp'):
            QMessageBox.warning(self, "No Data", "Compute trace and fit first.")
            return

        # Ask for base filename
        fname_base, _ = QFileDialog.getSaveFileName(self, "Save Base Name", "results", "All Files (*)")
        if not fname_base:
            return

        # Prepare file paths
        npz_path  = f"{fname_base}.npz"
        png_win   = f"{fname_base}_window.png"
        png_hist  = f"{fname_base}_histogram.png"
        csv_path  = f"{fname_base}.csv"
        txt_path  = f"{fname_base}.txt"

        # Compute region and model
        reg_min, reg_max = self.region.getRegion()
        low, high = int(np.ceil(reg_min)), int(np.floor(reg_max))
        p_fit = pch_mixture(self.k_vals, self.last_params[:self.spin_comp.value()],
                            self.last_params[self.spin_comp.value():])
        p_fit /= p_fit.sum()

        # Save data arrays and fit text in npz
        np.savez(npz_path,
                 t_centers=self.trace_plot.listDataItems()[0].xData,
                 trace_counts=self.trace_plot.listDataItems()[0].yData,
                 k_vals=self.k_vals,
                 p_exp=self.p_exp,
                 p_fit=p_fit,
                 fit_results=self.results_edit.toPlainText().splitlines())

        # Screenshot of full window
        pix_win = self.grab()
        pix_win.save(png_win)

        # Screenshot of histogram plot
        hist_item = self.hist_plot
        pix_hist = hist_item.grab()
        pix_hist.save(png_hist)

        # Save CSV: k_vals, p_exp, p_fit
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['k', 'P_exp', 'P_fit'])
            for k, expv, fitv in zip(self.k_vals, self.p_exp, p_fit):
                writer.writerow([int(k), expv, fitv])

        # Save fit text as txt with utf-8 encoding
        with open(txt_path, 'w', encoding='utf-8') as ftxt:
            ftxt.write(self.results_edit.toPlainText())

        QMessageBox.information(self, "Saved",
                                f"Results saved as:\n{npz_path}\n{png_win}\n{png_hist}\n{csv_path}\n{txt_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PCHApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "plugin":
    window = PCHApp()
    window.show()
