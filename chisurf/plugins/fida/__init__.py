import sys
import numpy as np
import tttrlib
from scipy.signal import fftconvolve
from scipy.optimize import least_squares
from numba import njit
from scipy.fft import ifft

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout,
    QGroupBox, QPushButton, QFileDialog, QLabel, QMessageBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QScrollArea,
    QLineEdit, QTextEdit, QCheckBox
)
import pyqtgraph as pg

name = "Single-Molecule:FIDA"

"""
Fluorescence Intensity Distribution Analysis (FIDA)

This plugin provides a graphical interface for analyzing the distribution of 
fluorescence intensities in single-molecule experiments. Features include:

- Loading and processing of photon data from TTTR files
- Calculation of photon count histograms
- Fitting of theoretical models to experimental data
- Determination of concentrations and specific brightnesses of fluorescent species
- Visualization of results with interactive plots

FIDA is a powerful method for characterizing heterogeneous samples and resolving 
different molecular species based on their brightness.
"""


# --- FIDA Core Functions ---
@njit(fastmath=True)
def get_dV_dx_fida(x_val, a_coeffs):
    if x_val <= 1e-9:
        return 0.0
    val = 0.0
    for k_idx, coeff in enumerate(a_coeffs):
        val += coeff * (x_val ** (k_idx + 1))
    return val


@njit(fastmath=True)
def spatial_integral_term_fida_numba(xi_real, xi_imag, qj, T_bin, B0, a_coeffs, x_points):
    integral_val_real = 0.0
    integral_val_imag = 0.0
    xi_minus_1_real = xi_real - 1.0
    xi_minus_1_imag = xi_imag
    for i in range(len(x_points) - 1):
        x_i = x_points[i]
        dx_i = x_points[i + 1] - x_i
        if dx_i == 0:
            continue
        term_coeff = qj * B0 * np.exp(-x_i) * T_bin
        exp_arg_real = xi_minus_1_real * term_coeff
        exp_arg_imag = xi_minus_1_imag * term_coeff
        exp_val_real = np.exp(exp_arg_real) * np.cos(exp_arg_imag)
        exp_val_imag = np.exp(exp_arg_real) * np.sin(exp_arg_imag)
        term_real = exp_val_real - 1.0
        term_imag = exp_val_imag
        dV_dx_val = get_dV_dx_fida(x_i, a_coeffs)
        integral_val_real += term_real * dV_dx_val * dx_i
        integral_val_imag += term_imag * dV_dx_val * dx_i
    return integral_val_real, integral_val_imag


def calculate_G_xi_fida(xi_array, concentrations, specific_brightnesses,
                        T_bin, B0, a_coeffs, x_min, x_max, num_x):
    logG_real = np.zeros(len(xi_array), dtype=np.float64)
    logG_imag = np.zeros(len(xi_array), dtype=np.float64)
    x_points = np.linspace(x_min, x_max, num_x)
    a_coeffs_np = np.array(a_coeffs, dtype=np.float64)
    for c_j, q_j in zip(concentrations, specific_brightnesses):
        if c_j <= 0 or q_j <= 0:
            continue
        for idx, xi in enumerate(xi_array):
            r, i = spatial_integral_term_fida_numba(
                np.real(xi), np.imag(xi), q_j, T_bin, B0,
                a_coeffs_np, x_points
            )
            logG_real[idx] += c_j * r
            logG_imag[idx] += c_j * i
    exp_real = np.exp(logG_real)
    return exp_real * (np.cos(logG_imag) + 1j * np.sin(logG_imag))


def calculate_P_n_fida(k_vals, concentrations, specific_brightnesses,
                       T_bin, B0, a_coeffs, background_rate,
                       x_min=1e-5, x_max=10, num_x=1000):
    max_k = int(k_vals.max())
    N = 1 << ((max_k + 1).bit_length() + 1)
    phi = np.arange(N) * (2 * np.pi / N)
    xi = np.exp(1j * phi)
    G_signal = calculate_G_xi_fida(
        xi, concentrations, specific_brightnesses,
        T_bin, B0, a_coeffs, x_min, x_max, num_x
    )
    if background_rate > 0:
        G_bg = np.exp(background_rate * (xi - 1.0))
        G_tot = G_signal * G_bg
    else:
        G_tot = G_signal
    P_complex = ifft(G_tot, n=N)
    P = np.real(P_complex[:max_k + 1])
    P[P < 0] = 0
    if P.sum() > 0:
        P /= P.sum()
    if len(P) < len(k_vals):
        P = np.pad(P, (0, len(k_vals) - len(P)))
    return P[:len(k_vals)]


# --- GUI Application ---
class FIDAApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(name)
        self.resize(1000, 700)
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)

        # Plot area
        plot_w = QWidget()
        vplot = QVBoxLayout(plot_w)
        self.trace_plot = pg.PlotWidget(title="Intensity Trace")
        self.trace_plot.setLabel("bottom", "Time (s)")
        self.trace_plot.setLabel("left", "Photon Counts per Bin")
        self.hist_plot = pg.PlotWidget(title="Photon Count Distribution (FIDA)")
        self.hist_plot.setLogMode(y=True)
        self.hist_plot.setLabel("bottom", "Photon Count n")
        self.hist_plot.setLabel("left", "P(n)")
        self.region = pg.LinearRegionItem([0, 1], swapMode='handle')
        self.region.sigRegionChanged.connect(self.update_region_fit_display)
        self.hist_plot.addItem(self.region)
        vplot.addWidget(self.trace_plot)
        vplot.addWidget(self.hist_plot)
        grid.addWidget(plot_w, 0, 0, 3, 1)

        # Data Settings
        grp_data = QGroupBox("Data Settings")
        dlay = QFormLayout(grp_data)
        self.le_file = QLineEdit();
        self.le_file.setReadOnly(True)
        btn_load = QPushButton("Load TTTR File");
        btn_load.clicked.connect(self.load_file)
        dlay.addRow(btn_load, self.le_file)
        self.le_ch = QLineEdit("0")
        dlay.addRow("Routing Channels:", self.le_ch)
        self.spin_bin = QDoubleSpinBox()
        self.spin_bin.setRange(0.1, 1e6);
        self.spin_bin.setValue(40.0)
        self.spin_bin.setSuffix(" μs");
        self.spin_bin.setDecimals(1)
        dlay.addRow("Bin Time T (μs):", self.spin_bin)
        # MicroTime range
        hmt = QHBoxLayout()
        self.spin_mt_min = QSpinBox();
        self.spin_mt_min.setRange(0, 65535);
        self.spin_mt_min.setValue(0)
        self.spin_mt_max = QSpinBox();
        self.spin_mt_max.setRange(0, 65535);
        self.spin_mt_max.setValue(65535)
        hmt.addWidget(QLabel("MicroTime Range:"));
        hmt.addWidget(self.spin_mt_min)
        hmt.addWidget(QLabel("to"));
        hmt.addWidget(self.spin_mt_max)
        dlay.addRow(hmt)
        btn_trace = QPushButton("Compute Trace & Distribution");
        btn_trace.clicked.connect(self.compute_trace_distribution)
        dlay.addRow(btn_trace)
        btn_save = QPushButton("Save Results");
        btn_save.clicked.connect(self.save_results)
        dlay.addRow(btn_save)
        grid.addWidget(grp_data, 0, 1)

        # FIDA Optical Parameters
        grp_opt = QGroupBox("FIDA Optical Parameters")
        olay = QFormLayout(grp_opt)
        self.spin_B0 = QDoubleSpinBox();
        self.spin_B0.setRange(0.1, 1e6);
        self.spin_B0.setValue(1.0)
        self.spin_B0.setDecimals(2)
        olay.addRow("B(0):", self.spin_B0)
        # a1, a2, a3 with fit checkboxes
        self.spin_a1 = QDoubleSpinBox();
        self.spin_a1.setRange(-1e3, 1e3);
        self.spin_a1.setDecimals(5);
        self.spin_a1.setValue(1.0)
        self.chk_a1 = QCheckBox("Fit a1");
        self.chk_a1.setChecked(False)
        box1 = QHBoxLayout();
        box1.addWidget(self.spin_a1);
        box1.addWidget(self.chk_a1)
        olay.addRow("a1 coeff:", box1)
        self.spin_a2 = QDoubleSpinBox();
        self.spin_a2.setRange(-1e3, 1e3);
        self.spin_a2.setDecimals(5);
        self.spin_a2.setValue(0.5)
        self.chk_a2 = QCheckBox("Fit a2");
        self.chk_a2.setChecked(False)
        box2 = QHBoxLayout();
        box2.addWidget(self.spin_a2);
        box2.addWidget(self.chk_a2)
        olay.addRow("a2 coeff:", box2)
        self.spin_a3 = QDoubleSpinBox();
        self.spin_a3.setRange(-1e3, 1e3);
        self.spin_a3.setDecimals(5);
        self.spin_a3.setValue(0.1)
        self.chk_a3 = QCheckBox("Fit a3");
        self.chk_a3.setChecked(False)
        box3 = QHBoxLayout();
        box3.addWidget(self.spin_a3);
        box3.addWidget(self.chk_a3)
        olay.addRow("a3 coeff:", box3)
        self.spin_bg = QDoubleSpinBox();
        self.spin_bg.setRange(0, 1e6);
        self.spin_bg.setDecimals(3)
        self.spin_bg.setValue(0.1)
        olay.addRow("Background rate (counts/bin):", self.spin_bg)
        # Integration parameters
        hin = QHBoxLayout()
        self.spin_x_min = QDoubleSpinBox();
        self.spin_x_min.setValue(1e-4);
        self.spin_x_min.setDecimals(5);
        self.spin_x_min.setSingleStep(0.001)
        self.spin_x_max = QDoubleSpinBox();
        self.spin_x_max.setValue(10.0);
        self.spin_x_max.setDecimals(1);
        self.spin_x_max.setSingleStep(1)
        self.spin_x_pts = QSpinBox();
        self.spin_x_pts.setRange(100, 10000);
        self.spin_x_pts.setValue(1000)
        hin.addWidget(QLabel("x_min:"));
        hin.addWidget(self.spin_x_min)
        hin.addWidget(QLabel("x_max:"));
        hin.addWidget(self.spin_x_max)
        hin.addWidget(QLabel("N_pts:"));
        hin.addWidget(self.spin_x_pts)
        olay.addRow("Integration (ln B0/B):", hin)
        grid.addWidget(grp_opt, 1, 1)

        # Species Fit
        grp_fit = QGroupBox("FIDA Species Fit")
        vfit = QVBoxLayout(grp_fit)
        fhl = QFormLayout()
        self.spin_comp = QSpinBox();
        self.spin_comp.setRange(1, 10);
        self.spin_comp.setValue(1)
        self.spin_comp.valueChanged.connect(self.update_species_inputs)
        fhl.addRow("# Species:", self.spin_comp)
        vfit.addLayout(fhl)
        self.spec_area = QScrollArea();
        self.spec_area.setWidgetResizable(True)
        self.spec_widget = QWidget();
        self.spec_layout = QFormLayout(self.spec_widget)
        self.spec_area.setWidget(self.spec_widget)
        vfit.addWidget(self.spec_area)
        self.update_species_inputs(1)
        btn_fit = QPushButton("Fit Distribution (FIDA)");
        btn_fit.clicked.connect(self.fit_distribution_fida)
        vfit.addWidget(btn_fit)
        vfit.addWidget(QLabel("Fit Results:"))
        self.results_edit = QTextEdit();
        self.results_edit.setReadOnly(True)
        vfit.addWidget(self.results_edit)
        grid.addWidget(grp_fit, 2, 1)
        grid.setColumnStretch(0, 2);
        grid.setColumnStretch(1, 1)

        # Avoid AttributeError
        self.last_params = None

    def update_species_inputs(self, count):
        old_q = [b.value() for b in getattr(self, 'q_boxes', [])]
        old_c = [b.value() for b in getattr(self, 'c_boxes', [])]
        for i in reversed(range(self.spec_layout.count())):
            item = self.spec_layout.takeAt(i)
            if item.widget(): item.widget().deleteLater()
        self.q_boxes = [];
        self.c_boxes = []
        for i in range(count):
            qb = QDoubleSpinBox();
            qb.setRange(0.01, 1e6);
            qb.setDecimals(3)
            qb.setValue(old_q[i] if i < len(old_q) else 10.0)
            cb = QDoubleSpinBox();
            cb.setRange(0.001, 1e6);
            cb.setDecimals(4)
            cb.setValue(old_c[i] if i < len(old_c) else 0.1)
            self.spec_layout.addRow(f"Species {i + 1} q_j:", qb)
            self.spec_layout.addRow(f"Species {i + 1} c_j:", cb)
            self.q_boxes.append(qb);
            self.c_boxes.append(cb)
        self.spec_widget.adjustSize()

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open TTTR File", "", "TTTR (*.ptu *.ht3 *.t2r *.t3r)")
        if path:
            try:
                self.tttr = tttrlib.TTTR(path)
                self.le_file.setText(path)
                QMessageBox.information(self, "File Loaded", f"Loaded {len(self.tttr)} records.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                self.tttr = None

    def compute_trace_distribution(self):
        if not hasattr(self, 'tttr') or self.tttr is None:
            QMessageBox.warning(self, "No File", "Please load a TTTR first")
            return
        try:
            txt_ch = self.le_ch.text().strip()
            av = np.unique(self.tttr.routing_channels)
            if not txt_ch:
                chs = av
            else:
                chs = list(map(int, txt_ch.split(',')))
                if not all(c in av for c in chs):
                    QMessageBox.warning(self, "Invalid Channels", f"Available: {av}")
                    return
            Tbin = self.spin_bin.value() * 1e-6
            mtmin = self.spin_mt_min.value();
            mtmax = self.spin_mt_max.value()
            mch = np.isin(self.tttr.routing_channels, chs)
            if self.tttr.micro_times.size > 0:
                mmt = (self.tttr.micro_times >= mtmin) & (self.tttr.micro_times <= mtmax)
                mask = mch & mmt
            else:
                mask = mch
            if not np.any(mask) and self.tttr.macro_times.size > 0:
                QMessageBox.warning(self, "No Photons", "No photons after filtering")
                self._setup_empty()
                return
            data = self.tttr.macro_times[mask] * self.tttr.header.macro_time_resolution
            if data.size == 0:
                QMessageBox.warning(self, "No Photons", "No photons remain")
                self._setup_empty()
                return
            tmax = data.max();
            nb = int(np.ceil(tmax / Tbin)) or 1
            counts, edges = np.histogram(data, bins=nb, range=(0, nb * Tbin))
            self.time_centers = (edges[:-1] + edges[1:]) / 2.0
            self.trace_plot.clear();
            self.trace_plot.plot(self.time_centers, counts, pen='b')
            self.hist_counts = np.bincount(counts, minlength=counts.max() + 1)
            self.total_bins = counts.size
            self.k_vals = np.arange(self.hist_counts.size)
            if self.total_bins > 0:
                self.p_exp = self.hist_counts / self.total_bins
            else:
                self.p_exp = np.zeros_like(self.k_vals);
                self.p_exp[0] = 1.0
            self.hist_plot.clear();
            self.hist_plot.addItem(self.region)
            self.hist_plot.plot(self.k_vals, self.p_exp, pen=None, symbol='o', symbolBrush='g')
            maxk = self.k_vals.max() if self.k_vals.size > 0 else 10
            self.region.setRegion([0, min(maxk, 50 if maxk > 5 else maxk + 1)])
            self.last_params = None
        except Exception as e:
            QMessageBox.critical(self, "Error Computing Trace", str(e))

    def _setup_empty(self):
        self.time_centers = np.array([]);
        self.k_vals = np.array([0]);
        self.p_exp = np.array([1.0]);
        self.total_bins = 0
        self.trace_plot.clear();
        self.hist_plot.clear();
        self.hist_plot.addItem(self.region)
        self.hist_plot.plot(self.k_vals, self.p_exp, pen=None, symbol='o', symbolBrush='g')
        self.last_params = None

    def fit_distribution_fida(self):
        if not hasattr(self, 'p_exp'):
            QMessageBox.warning(self, "No Data", "Compute first");
            return
        m = self.spin_comp.value()
        # initial a
        a_vals = [self.spin_a1.value(), self.spin_a2.value(), self.spin_a3.value()]
        self.a_fit_indices = []
        self.a_fixed_list = []
        for i, chk in enumerate([self.chk_a1, self.chk_a2, self.chk_a3]):
            if chk.isChecked():
                self.a_fit_indices.append(i)
            else:
                self.a_fixed_list.append((i, a_vals[i]))
        init_params = []
        for i in self.a_fit_indices:
            init_params.append(a_vals[i])
        q0 = [b.value() for b in self.q_boxes]
        c0 = [b.value() for b in self.c_boxes]
        init_params += q0 + c0
        init_params = np.array(init_params)
        # bounds
        lower = []
        upper = []
        for _ in self.a_fit_indices:
            lower.append(-np.inf);
            upper.append(np.inf)
        lower += [1e-3] * m + [1e-6] * m
        upper += [np.inf] * (2 * m)
        Tbin = self.spin_bin.value() * 1e-6
        B0 = self.spin_B0.value()
        bg = self.spin_bg.value()
        xmin = self.spin_x_min.value();
        xmax = self.spin_x_max.value();
        xpts = self.spin_x_pts.value()
        kl, kh = self.region.getRegion();
        kl = int(np.ceil(kl));
        kh = int(np.floor(kh))
        kl = max(0, kl);
        kh = min(len(self.k_vals) - 1, kh)
        mask = (self.k_vals >= kl) & (self.k_vals <= kh)
        if not np.any(mask): QMessageBox.warning(self, "Fit Region", "Invalid region"); return
        kfit = self.k_vals[mask];
        pfit = self.p_exp[mask]

        def residuals(params):
            idx = 0
            a_recon = [None] * 3
            for ai in self.a_fit_indices:
                a_recon[ai] = params[idx];
                idx += 1
            for fi, val in self.a_fixed_list:
                a_recon[fi] = val
            q = params[idx:idx + m];
            c = params[idx + m:idx + 2 * m]
            p_model = calculate_P_n_fida(self.k_vals, q, c, Tbin, B0,
                                         tuple(a_recon), bg,
                                         xmin, xmax, xpts)
            diff = p_model[mask] - pfit
            w = np.sqrt(self.total_bins * pfit + 1e-9)
            return diff * w

        try:
            resl = least_squares(residuals, init_params, bounds=(lower, upper),
                                 method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8)
            self.last_params = resl.x
            self.update_fit_gui_and_results(resl.x)
        except Exception as e:
            QMessageBox.critical(self, "Fit Error", str(e))

    def update_fit_gui_and_results(self, params):
        m = self.spin_comp.value()
        idx = 0
        for i in self.a_fit_indices:
            val = params[idx];
            idx += 1
            getattr(self, f'spin_a{i + 1}').setValue(val)
        q_fit = params[idx:idx + m];
        c_fit = params[idx + m:idx + 2 * m]
        for i in range(m):
            self.q_boxes[i].setValue(q_fit[i])
            self.c_boxes[i].setValue(c_fit[i])
        Tbin = self.spin_bin.value() * 1e-6;
        B0 = self.spin_B0.value();
        bg = self.spin_bg.value()
        a_recon = [self.spin_a1.value(), self.spin_a2.value(), self.spin_a3.value()]
        pmod = calculate_P_n_fida(self.k_vals, q_fit, c_fit, Tbin, B0,
                                  tuple(a_recon), bg,
                                  self.spin_x_min.value(),
                                  self.spin_x_max.value(),
                                  self.spin_x_pts.value())
        for it in self.hist_plot.listDataItems():
            if hasattr(it, '_is_fit'): self.hist_plot.removeItem(it)
        fit_cur = self.hist_plot.plot(self.k_vals, pmod, pen=pg.mkPen('r', width=2))
        fit_cur._is_fit = True
        kl, kh = self.region.getRegion();
        kl = int(np.ceil(kl));
        kh = int(np.floor(kh))
        kl = max(0, kl);
        kh = min(len(self.k_vals) - 1, kh)
        mask = (self.k_vals >= kl) & (self.k_vals <= kh)
        pexp = self.p_exp[mask];
        pfit2 = pmod[mask]
        valid = pfit2 > 1e-9
        chi2 = self.total_bins * np.sum((pexp[valid] - pfit2[valid]) ** 2 / pfit2[valid]) if np.any(valid) else np.nan
        dof = valid.sum() - len(params)
        redchi = chi2 / dof if dof > 0 else np.nan
        results = [f"Fit region k: {kl}-{kh}\n", "Optical Profile Coeffs:"]
        for i, chk in enumerate([self.chk_a1, self.chk_a2, self.chk_a3]):
            val = getattr(self, f'spin_a{i + 1}').value()
            results.append(f" a{i + 1}: {val:.5g} ({'fitted' if chk.isChecked() else 'fixed'})")
        results.append(f" T_bin: {Tbin * 1e6:.1f} µs, B0: {B0:.2f}, Background: {bg:.3f}\n")
        for i in range(m):
            results.append(f"Species {i + 1}: q={q_fit[i]:.3e}, c={c_fit[i]:.4f}")
        results.append(f"\nChi2: {chi2:.3f}, DoF: {dof}, RedChi2: {redchi:.3f}")
        self.results_edit.setPlainText("\n".join(results))

    def update_region_fit_display(self):
        if self.last_params is not None:
            self.update_fit_gui_and_results(self.last_params)

    def save_results(self):
        if not hasattr(self, 'p_exp') or self.last_params is None:
            QMessageBox.warning(self, "Save Error", "Compute and fit first")
            return
        base, _ = QFileDialog.getSaveFileName(self, "Save Base", "fida_results", "*")
        if not base: return
        npz = f"{base}_data.npz";
        png = f"{base}_hist.png";
        csvf = f"{base}_dist.csv";
        txtf = f"{base}_sum.txt"
        m = self.spin_comp.value()
        q = self.last_params[len(self.a_fit_indices):len(self.a_fit_indices) + m]
        c = self.last_params[len(self.a_fit_indices) + m:]
        Tbin = self.spin_bin.value() * 1e-6;
        B0 = self.spin_B0.value();
        bg = self.spin_bg.value()
        a_recon = [self.spin_a1.value(), self.spin_a2.value(), self.spin_a3.value()]
        pmod = calculate_P_n_fida(self.k_vals, q, c, Tbin, B0, tuple(a_recon), bg,
                                  self.spin_x_min.value(), self.spin_x_max.value(), self.spin_x_pts.value())
        items = self.trace_plot.listDataItems()
        t = items[0].xData if items else np.array([])
        y = items[0].yData if items else np.array([])
        np.savez(npz, trace_time=t, trace_counts=y, k_vals=self.k_vals,
                 p_exp=self.p_exp, p_fit=pmod, q=q, c=c,
                 Tbin=Tbin, B0=B0, a_coeffs=tuple(a_recon), bg=bg,
                 summary=self.results_edit.toPlainText().splitlines())
        try:
            exporter = pg.exporters.ImageExporter(self.hist_plot.plotItem)
            exporter.export(png)
        except:
            pass
        import csv
        try:
            with open(csvf, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(['k', 'P_exp', 'P_fit'])
                for kval, pe, pf in zip(self.k_vals, self.p_exp, pmod):
                    writer.writerow([int(kval), pe, pf])
        except:
            pass
        try:
            with open(txtf, 'w') as tf:
                tf.write(self.results_edit.toPlainText())
        except:
            pass
        QMessageBox.information(self, "Saved", f"Saved: {npz}, {png}, {csvf}, {txtf}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FIDAApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "plugin":
    window = FIDAApp()
    window.show()
