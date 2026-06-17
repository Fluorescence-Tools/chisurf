from __future__ import annotations

import csv
import logging
from typing import Any

import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..api.models import FitResult, PchResult
from .client import PCHClient

logger = logging.getLogger(__name__)


class PCHApp(QMainWindow):
    name = "Spectroscopy:Single-Molecule:PCH"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photon Counting Histogram (PCH)")
        self.resize(1000, 650)

        self._client = PCHClient()
        self._result: PchResult | None = None
        self._fit_result: FitResult | None = None
        self._filename: str = ""

        self._setup_toolbar()
        self._setup_central()
        self._setup_statusbar()

    # ── toolbar ────────────────────────────────────────────────────

    def _setup_toolbar(self):
        tb = QToolBar("PCH Controls")
        tb.setIconSize(tb.iconSize())
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        self.action_load = tb.addAction("Load TTTR")
        self.action_load.triggered.connect(self._on_load)

        tb.addSeparator()

        self.action_compute = tb.addAction("Compute PCH")
        self.action_compute.setEnabled(False)
        self.action_compute.triggered.connect(self._on_compute)

        self.action_fit = tb.addAction("Fit Model")
        self.action_fit.setEnabled(False)
        self.action_fit.triggered.connect(self._on_fit)

        tb.addSeparator()

        self.action_save = tb.addAction("Save Results")
        self.action_save.setEnabled(False)
        self.action_save.triggered.connect(self._on_save)

    # ── central layout ─────────────────────────────────────────────

    def _setup_central(self):
        splitter = QSplitter(Qt.Horizontal)

        left = self._build_plots()
        right = self._build_settings_panel()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.setCentralWidget(splitter)

    def _build_plots(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.trace_plot = pg.PlotWidget(title="Intensity Trace")
        self.trace_plot.setLabel("bottom", "Time (s)")
        self.trace_plot.setLabel("left", "Photon Counts")

        self.hist_plot = pg.PlotWidget(title="Photon Counting Histogram")
        self.hist_plot.setLogMode(y=True)
        self.hist_plot.setLabel("bottom", "Photon Count k")
        self.hist_plot.setLabel("left", "P(k)")
        self.region = pg.LinearRegionItem([0, 1], swapMode="handle")
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.hist_plot.addItem(self.region)

        layout.addWidget(self.trace_plot, stretch=1)
        layout.addWidget(self.hist_plot, stretch=1)
        return container

    def _build_settings_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ── Data Settings ──
        grp_data = QGroupBox("Data Settings")
        form = QFormLayout(grp_data)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(4)

        self.le_file = QLineEdit()
        self.le_file.setReadOnly(True)
        form.addRow("File:", self.le_file)

        self.le_ch = QLineEdit("0,2")
        form.addRow("Channels:", self.le_ch)

        self.spin_bin = QDoubleSpinBox()
        self.spin_bin.setRange(0.1, 1e6)
        self.spin_bin.setValue(100.0)
        self.spin_bin.setSuffix(" μs")
        self.spin_bin.valueChanged.connect(self._on_param_changed)
        form.addRow("Bin Time:", self.spin_bin)

        mt_row = QHBoxLayout()
        self.spin_mt_min = QSpinBox()
        self.spin_mt_min.setRange(0, 65535)
        self.spin_mt_min.setValue(0)
        self.spin_mt_min.valueChanged.connect(self._on_param_changed)
        self.spin_mt_max = QSpinBox()
        self.spin_mt_max.setRange(0, 65535)
        self.spin_mt_max.setValue(65535)
        self.spin_mt_max.valueChanged.connect(self._on_param_changed)
        mt_row.addWidget(self.spin_mt_min)
        mt_row.addWidget(QLabel("to"))
        mt_row.addWidget(self.spin_mt_max)
        form.addRow("Micro Time:", mt_row)

        layout.addWidget(grp_data)

        # ── Model Fit ──
        grp_fit = QGroupBox("Model Fit")
        vfit = QVBoxLayout(grp_fit)
        vfit.setContentsMargins(6, 6, 6, 6)
        vfit.setSpacing(4)
        ffit = QFormLayout()
        ffit.setSpacing(4)

        self.spin_comp = QSpinBox()
        self.spin_comp.setRange(1, 10)
        self.spin_comp.setValue(1)
        self.spin_comp.valueChanged.connect(self._update_species_inputs)
        ffit.addRow("Components:", self.spin_comp)
        vfit.addLayout(ffit)

        self.species_area = QScrollArea()
        self.species_area.setWidgetResizable(True)
        self.species_area.setMaximumHeight(200)
        self.species_widget = QWidget()
        self.species_layout = QFormLayout(self.species_widget)
        self.species_layout.setContentsMargins(0, 0, 0, 0)
        self.species_layout.setSpacing(2)
        self.species_area.setWidget(self.species_widget)
        vfit.addWidget(self.species_area)

        self._init_species_inputs(1)

        vfit.addWidget(QLabel("Fit Results:"))
        self.results_edit = QTextEdit()
        self.results_edit.setReadOnly(True)
        self.results_edit.setMaximumHeight(140)
        vfit.addWidget(self.results_edit)

        layout.addWidget(grp_fit)
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _setup_statusbar(self):
        self.statusBar().showMessage("Ready. Load a TTTR file to begin.")

    # ── species inputs ─────────────────────────────────────────────

    def _init_species_inputs(self, count: int):
        self.eps_boxes: list[QDoubleSpinBox] = []
        self.N_boxes: list[QDoubleSpinBox] = []
        for i in range(count):
            self._add_species_row(i)

    def _add_species_row(self, idx: int):
        eb = QDoubleSpinBox()
        eb.setRange(0, 1e6)
        eb.setSingleStep(0.1)
        eb.setValue(2.0)
        nb = QDoubleSpinBox()
        nb.setRange(0, 1e6)
        nb.setSingleStep(0.1)
        nb.setValue(3.0)
        self.eps_boxes.append(eb)
        self.N_boxes.append(nb)
        self.species_layout.addRow(f"ε {idx + 1}", eb)
        self.species_layout.addRow(f"⟨N⟩ {idx + 1}", nb)

    def _update_species_inputs(self, count: int):
        old_eps = [b.value() for b in getattr(self, "eps_boxes", [])]
        old_Ns = [b.value() for b in getattr(self, "N_boxes", [])]
        for _ in range(len(self.species_layout.count()), 0, -1):
            self.species_layout.removeRow(0)
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

    # ── event handlers ─────────────────────────────────────────────

    def _on_param_changed(self):
        pass

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TTTR", "", "TTTR (*.ptu *.ht3 *.t2r *.t3r)"
        )
        if not path:
            return
        try:
            info = self._client.load_tttr(path)
            self._filename = path
            self.le_file.setText(path)
            self._result = None
            self._fit_result = None
            self.action_compute.setEnabled(True)
            self.action_save.setEnabled(False)
            self.statusBar().showMessage(
                f"Loaded: {path} ({info.get('n_photons', 0):,} photons)"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_compute(self):
        if not self._filename:
            QMessageBox.warning(self, "No File", "Load a TTTR file first.")
            return
        try:
            txt = self.le_ch.text().strip()
            channels = (
                list(map(int, txt.split(",")))
                if txt
                else None
            )
            result = self._client.compute(
                filename=self._filename,
                channels=channels,
                bin_time_us=self.spin_bin.value(),
                micro_time_min=self.spin_mt_min.value(),
                micro_time_max=self.spin_mt_max.value(),
            )
            self._result = PchResult.from_dict(result)
            self._fit_result = None
            self._plot_trace()
            self._plot_hist()
            self.region.setRegion([0, max(self._result.k_vals)])
            self.action_fit.setEnabled(True)
            self.action_save.setEnabled(True)
            self.statusBar().showMessage(
                f"Computed PCH: {self._result.total_bins:,} bins, "
                f"{len(self._result.k_vals)} k-values"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_fit(self):
        if self._result is None:
            QMessageBox.warning(self, "No Data", "Compute PCH first.")
            return
        try:
            n_comp = self.spin_comp.value()
            init_eps = [b.value() for b in self.eps_boxes]
            init_Ns = [b.value() for b in self.N_boxes]
            low, high = self.region.getRegion()
            fit_low = int(low)
            fit_high = int(high)

            fit_result = self._client.fit(
                k_vals=self._result.k_vals,
                p_exp=self._result.p_exp,
                hist_counts=self._result.hist_counts,
                total_bins=self._result.total_bins,
                n_components=n_comp,
                initial_epsilons=init_eps,
                initial_Ns=init_Ns,
                fit_low=fit_low,
                fit_high=fit_high,
            )
            self._fit_result = FitResult.from_dict(fit_result)

            for i in range(n_comp):
                self.eps_boxes[i].setValue(self._fit_result.epsilons[i])
                self.N_boxes[i].setValue(self._fit_result.avg_Ns[i])

            self._plot_fit()
            self._update_results_text()
            self.statusBar().showMessage(
                f"Fit complete: χ²={self._fit_result.chi2:.2f}, "
                f"red. χ²={self._fit_result.reduced_chi2:.3f}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_save(self):
        if self._result is None:
            QMessageBox.warning(self, "No Data", "Compute PCH first.")
            return

        fname_base, _ = QFileDialog.getSaveFileName(
            self, "Save Base Name", "results", "All Files (*)"
        )
        if not fname_base:
            return

        try:
            self._save_outputs(fname_base)
            QMessageBox.information(
                self,
                "Saved",
                f"Results saved as:\n{fname_base}.npz\n"
                f"{fname_base}_window.png\n"
                f"{fname_base}_histogram.png\n"
                f"{fname_base}.csv\n"
                f"{fname_base}.txt",
            )
            self.statusBar().showMessage(f"Saved results to {fname_base}.*")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_region_changed(self):
        if self._fit_result is not None and self._result is not None:
            self._update_results_text()

    # ── plotting ───────────────────────────────────────────────────

    def _plot_trace(self):
        self.trace_plot.clear()
        if self._result is not None:
            self.trace_plot.plot(
                self._result.trace_t,
                self._result.trace_counts,
                stepMode=False,
            )

    def _plot_hist(self):
        self.hist_plot.clear()
        self.hist_plot.addItem(self.region)
        if self._result is not None:
            self.hist_plot.plot(
                self._result.k_vals,
                self._result.p_exp,
                pen=None,
                symbol="o",
            )

    def _plot_fit(self):
        if self._result is None or self._fit_result is None:
            return
        self.hist_plot.clear()
        self.hist_plot.addItem(self.region)
        self.hist_plot.plot(
            self._result.k_vals,
            self._result.p_exp,
            pen=None,
            symbol="o",
        )
        low = self._fit_result.fit_low
        high = self._fit_result.fit_high
        mask = (
            (np.array(self._result.k_vals) >= low)
            & (np.array(self._result.k_vals) <= high)
        )
        k_arr = np.array(self._result.k_vals)
        self.hist_plot.plot(
            k_arr[mask],
            np.array(self._fit_result.p_fit)[mask],
            pen=pg.mkPen("r", width=2),
        )

    # ── results text ───────────────────────────────────────────────

    def _update_results_text(self):
        if self._result is None or self._fit_result is None:
            return
        if not hasattr(self, "region"):
            return
        fr = self._fit_result
        low, high = self.region.getRegion()
        fit_low, fit_high = int(low), int(high)

        k_arr = np.array(self._result.k_vals)
        p_fit = np.array(fr.p_fit)
        exp_cnt = p_fit * self._result.total_bins
        obs_cnt = np.array(self._result.hist_counts)
        mask = (k_arr >= fit_low) & (k_arr <= fit_high) & (exp_cnt > 0)
        chi2 = float(np.sum((obs_cnt[mask] - exp_cnt[mask]) ** 2 / exp_cnt[mask]))
        dof = int(mask.sum()) - (fr.n_components * 2)
        red_chi2 = chi2 / dof if dof > 0 else float("nan")

        lines = [
            "ε: molecular brightness  ⟨N⟩: molecules in volume",
            "",
            f"Region: {fit_low}–{fit_high}",
        ]
        for i in range(fr.n_components):
            lines.append(
                f"Comp{i + 1}: ε={fr.epsilons[i]:.4f}  "
                f"⟨N⟩={fr.avg_Ns[i]:.4f}  "
                f"x={fr.fractions[i]:.1f}%"
            )
        lines += [
            "",
            f"χ² = {chi2:.2f}   red. χ² = {red_chi2:.3f}   dof = {dof}",
        ]
        self.results_edit.setPlainText("\n".join(lines))

    # ── file I/O ───────────────────────────────────────────────────

    def _save_outputs(self, fname_base: str):
        import numpy as np
        from ..api.algorithms import pch_mixture

        npz_path = f"{fname_base}.npz"
        png_win = f"{fname_base}_window.png"
        png_hist = f"{fname_base}_histogram.png"
        csv_path = f"{fname_base}.csv"
        txt_path = f"{fname_base}.txt"

        p_fit_arr = (
            np.array(self._fit_result.p_fit)
            if self._fit_result is not None
            else np.array([])
        )

        np.savez(
            npz_path,
            t_centers=self._result.trace_t,
            trace_counts=self._result.trace_counts,
            k_vals=self._result.k_vals,
            p_exp=self._result.p_exp,
            p_fit=p_fit_arr,
            fit_results=self.results_edit.toPlainText().splitlines(),
        )

        pix_win = self.grab()
        pix_win.save(png_win)
        pix_hist = self.hist_plot.grab()
        pix_hist.save(png_hist)

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["k", "P_exp", "P_fit"])
            for i, k in enumerate(self._result.k_vals):
                fitv = p_fit_arr[i] if len(p_fit_arr) > i else 0.0
                writer.writerow([int(k), self._result.p_exp[i], fitv])

        with open(txt_path, "w", encoding="utf-8") as ftxt:
            ftxt.write(self.results_edit.toPlainText())


# Allow standalone run for testing
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = PCHApp()
    win.show()
    sys.exit(app.exec())
