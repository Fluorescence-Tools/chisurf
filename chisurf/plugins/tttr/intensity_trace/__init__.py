"""
Intensity Trace Analysis for Single-Molecule Data

This plugin provides tools for analyzing fluorescence intensity time traces from 
single-molecule experiments. It enables researchers to extract dynamic information 
from photon counting data, particularly for studying conformational changes, 
molecular interactions, and reaction kinetics at the single-molecule level.

Features:
- Loading and displaying Time-Tagged Time-Resolved (TTTR) data as intensity traces
- Histogram analysis of photon counts with customizable binning
- Hidden Markov Model (HMM) analysis for state detection and classification
- Bayesian Information Criterion (BIC) calculation for optimal state number determination
- Dwell time analysis for extracting kinetic information and rate constants
- FRET efficiency calculation and state-specific distribution analysis
- Transition probability matrix visualization and analysis
- Exponential fitting of dwell time distributions
- Interactive visualization with adjustable parameters
- Support for multi-channel data analysis (donor/acceptor channels)

The plugin implements a comprehensive workflow for single-molecule state analysis:
1. Load TTTR data and convert to binned intensity traces
2. Visualize traces and photon count distributions
3. Apply HMM to identify discrete states in noisy data
4. Analyze state transitions and dwell times to extract kinetic information
5. For FRET data, calculate efficiency distributions for each state

Ideal for analyzing single-molecule FRET, protein folding/unfolding, enzyme dynamics,
ligand binding, blinking behavior, or any other dynamic processes that can be 
observed in fluorescence intensity traces. The HMM approach is particularly powerful
for detecting states in noisy data with overlapping distributions.
"""

name = "Single-Molecule:Intensity trace"


import sys
import pathlib
import numpy as np
from scipy.optimize import curve_fit

from qtpy.QtCore import Qt
from qtpy.QtGui import QPainterPath, QBrush, QColor, QPen
from qtpy.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QToolButton, QFileDialog, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QDialog, QCheckBox, QGridLayout, QTabWidget, QGroupBox, QDialogButtonBox, 
    QGraphicsPathItem, QSizePolicy
)
from qtpy import QtWidgets
from qtpy import QtGui, QtCore

import pyqtgraph as pg
from pyqtgraph import TextItem, ImageItem, colormap

import tttrlib
from hmmlearn.hmm import GaussianHMM

# Chisurf imports for detector setup
from chisurf.gui.widgets.wizard.tttr_channel_definition import (
    DetectorWizardPage, DetectorWizard, load_detector_setups
)

# Logging
from chisurf import logging


def save_burst_ids(hmm_states, time_axis, time_window_s, tttr_obj, output_dir=".", file_path=None):
    burst_ids = {}

    for state in np.unique(hmm_states):
        mask = hmm_states == state
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        # Calculate burst intervals
        bursts = []
        start_idx = indices[0]

        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                stop_idx = indices[i - 1]
                bursts.append((start_idx, stop_idx))
                start_idx = indices[i]

        # Append the last burst
        bursts.append((start_idx, indices[-1]))

        burst_ids[state] = bursts

        # Optionally save burst IDs to file
        if file_path:
            base_name = pathlib.Path(file_path).stem
            output_file = pathlib.Path(output_dir) / f"{base_name}_state_{state}.bst"
        else:
            output_file = pathlib.Path(output_dir) / f"burst_ids_state_{state}.bst"
        with open(output_file, 'w') as f:
            for start_bin, stop_bin in bursts:
                # Convert bin indices back to TTTR indices
                start_time = time_axis[start_bin]
                stop_time = time_axis[stop_bin] + time_window_s

                macro_time_resolution = tttr_obj.header.macro_time_resolution
                start_tttr_idx = np.searchsorted(tttr_obj.macro_times, start_time / macro_time_resolution)
                stop_tttr_idx = np.searchsorted(tttr_obj.macro_times, stop_time / macro_time_resolution)

                f.write(f"{start_tttr_idx}\t{stop_tttr_idx}\n")

    return burst_ids


def compute_bic_curve(data, max_states=10):
    bics = []
    n_samples, n_features = data.shape

    for n in range(1, max_states + 1):
        try:
            model = GaussianHMM(n_components=n, covariance_type="full", n_iter=1000, random_state=0)
            model.fit(data)
            logL = model.score(data)
            n_params = n * (n - 1) + n * n_features * 2  # transitions + mean + cov
            bic = np.log(n_samples) * n_params - 2 * logL
            bics.append((n, bic))
        except Exception as e:
            print(f"HMM fit failed for {n} states: {e}")
            bics.append((n, np.nan))
    return bics


def compute_dwell_times(state_sequence, time_step=1.0):
    """
    Compute dwell times for each state in a state sequence.

    Parameters:
    -----------
    state_sequence : np.ndarray
        Array of inferred HMM state labels (1D).
    time_step : float
        Duration represented by each step in the state sequence (e.g., in seconds).

    Returns:
    --------
    dwell_times : dict
        Dictionary mapping state index to a list of dwell times in units of `time_step`.
    """
    dwell_times = {}
    if len(state_sequence) == 0:
        return dwell_times

    current_state = state_sequence[0]
    dwell_count = 1

    for s in state_sequence[1:]:
        if s == current_state:
            dwell_count += 1
        else:
            if current_state not in dwell_times:
                dwell_times[current_state] = []
            dwell_times[current_state].append(dwell_count * time_step)
            current_state = s
            dwell_count = 1

    # Save the last dwell
    if current_state not in dwell_times:
        dwell_times[current_state] = []
    dwell_times[current_state].append(dwell_count * time_step)

    return dwell_times


class DistPlotWindow(QtWidgets.QDialog):
    def __init__(self, fret_distributions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FRET Efficiency Distributions by State")
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)

        plots = []
        y_max = 0
        all_bins = np.linspace(0, 1, 51)
        bin_edges = all_bins

        # First pass: compute max y for shared scaling
        hists = {}
        for state, frets in fret_distributions.items():
            if len(frets) == 0:
                continue
            hist, _ = np.histogram(frets, bins=bin_edges, density=True)
            hists[state] = hist
            y_max = max(y_max, np.max(hist))

        # Second pass: plot, with state labels
        for idx, (state, hist) in enumerate(hists.items()):
            plot = self.plot_widget.addPlot(row=idx, col=0)
            # — add the state as the plot title —
            plot.setTitle(f"State {state}")

            plot.setYRange(0, y_max)
            plot.setXRange(0, 1)

            x_vals = bin_edges[:-1]
            y_vals = hist
            color = (state * 50 % 255, 100, 180, 150)
            plot.plot(x_vals, y_vals, stepMode=False, fillLevel=0, brush=color)

            # clean up axes (we're showing titles instead)
            plot.hideAxis('left')
            if idx == len(hists) - 1:
                plot.setLabel('bottom', 'FRET Efficiency')
            else:
                plot.hideAxis('bottom')

            plots.append(plot)

        # Link all axes
        for plot in plots[1:]:
            plot.setXLink(plots[0])
            plot.setYLink(plots[0])


class ElbowPlotWindow(QtWidgets.QDialog):
    def __init__(self, bics, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HMM State Selection - BIC Elbow Plot")

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.plot_widget = pg.PlotWidget(title="BIC vs Number of States")
        layout.addWidget(self.plot_widget)

        states, bic_values = zip(*bics)
        bic_values = np.array(bic_values)
        valid = ~np.isnan(bic_values)
        self.plot_widget.plot(np.array(states)[valid], bic_values[valid], pen='b', symbol='o')
        self.plot_widget.setLabel('bottom', 'Number of States')
        self.plot_widget.setLabel('left', 'BIC (lower is better)')


class DwellTimeWindow(QtWidgets.QDialog):
    def __init__(self, dwell_times_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dwell Time Distributions")
        self.dwell_times_dict = dwell_times_dict

        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)

        # Controls
        control_layout = QtWidgets.QHBoxLayout()

        self.min_bin_box = QtWidgets.QDoubleSpinBox()
        self.min_bin_box.setRange(0.0, 90000.0)
        self.min_bin_box.setSuffix(" ms")
        self.min_bin_box.setValue(0.0)
        control_layout.addWidget(QtWidgets.QLabel("Min Bin:"))
        control_layout.addWidget(self.min_bin_box)

        self.max_bin_box = QtWidgets.QDoubleSpinBox()
        self.max_bin_box.setRange(0.1, 90000.0)
        self.max_bin_box.setSuffix(" ms")
        self.max_bin_box.setValue(1000.0)
        control_layout.addWidget(QtWidgets.QLabel("Max Bin:"))
        control_layout.addWidget(self.max_bin_box)

        self.bin_spinner = QtWidgets.QSpinBox()
        self.bin_spinner.setRange(10, 200)
        self.bin_spinner.setValue(31)
        control_layout.addWidget(QtWidgets.QLabel("Number of Bins:"))
        control_layout.addWidget(self.bin_spinner)

        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Histogram")
        control_layout.addWidget(self.normalize_checkbox)

        save_button = QtWidgets.QPushButton("Save Histograms and Fits")
        save_button.clicked.connect(self.save_histograms)
        control_layout.addWidget(save_button)

        main_layout.addLayout(control_layout)

        # Plot widget
        self.plot_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.plot_widget)

        # Storage for histograms and fit results
        self.histograms = {}
        self.fits = {}

        # Connect updates
        self.bin_spinner.valueChanged.connect(self.update_plots)
        self.min_bin_box.valueChanged.connect(self.update_plots)
        self.max_bin_box.valueChanged.connect(self.update_plots)
        self.normalize_checkbox.stateChanged.connect(self.update_plots)

        self.update_plots()

    @staticmethod
    def _exp_func(x, A, tau):
        return A * np.exp(-x / tau)

    def update_plots(self):
        self.plot_widget.clear()
        n_bins = self.bin_spinner.value()
        min_bin = self.min_bin_box.value()
        max_bin = self.max_bin_box.value()
        normalize = self.normalize_checkbox.isChecked()

        if max_bin <= min_bin:
            return

        bins = np.linspace(min_bin, max_bin, n_bins + 1)
        self.histograms.clear()
        self.fits.clear()
        plots = []

        for state, dwell_times in self.dwell_times_dict.items():
            if len(dwell_times) == 0:
                continue

            plot = self.plot_widget.addPlot()
            plots.append(plot)

            # Histogram data
            hist_data = np.array(dwell_times) * 1e3
            y, x = np.histogram(hist_data, bins=bins)
            if normalize:
                y = y / y.sum() if y.sum() > 0 else y
            x_center = 0.5 * (x[:-1] + x[1:])
            self.histograms[state] = (x_center, y)

            # Exponential fit
            mask = y > 0
            try:
                popt, _ = curve_fit(
                    self._exp_func,
                    x_center[mask],
                    y[mask],
                    p0=(y.max(), (x_center * y).sum() / y.sum())
                )
                A_fit, tau_fit = popt
                self.fits[state] = (A_fit, tau_fit)

                # Annotate fit on plot with margin
                fit_text = f"τ = {tau_fit:.1f} ms"
                text_item = TextItem(fit_text, anchor=(0, 0))
                plot.addItem(text_item)

                vb = plot.getViewBox()
                rect = vb.viewRect()
                x_min, x_max = rect.left(), rect.right()
                y_min, y_max = rect.bottom(), rect.top()
                # 15% margin inside the view
                x_offset = 0.15 * (x_max - x_min)
                y_offset = 0.15 * (y_max - y_min)
                text_item.setPos(x_max - x_offset, y_max - y_offset)

                # Plot fit curve
                x_fit = np.linspace(min_bin, max_bin, 200)
                y_fit = self._exp_func(x_fit, *popt)
                plot.plot(x_fit, y_fit, pen=pg.mkPen('r', width=2))
            except Exception:
                pass

            color = (state * 40 % 255, 100, 150, 150)
            plot.plot(x_center, y, stepMode=False, fillLevel=0, brush=color)
            plot.setLabel('bottom', f'Dwell Time (ms) - State {state}')
            plot.hideAxis('left')

        if len(plots) > 1:
            for p in plots[1:]:
                p.setXLink(plots[0])
            if normalize:
                for p in plots[1:]:
                    p.setYLink(plots[0])

    def save_histograms(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Dwell Time Histograms and Fits", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        with open(file_path, 'w') as f:
            f.write("State,BinCenter,Count\n")
            for state, (x_center, y) in self.histograms.items():
                for xc, yc in zip(x_center, y):
                    f.write(f"{state},{xc},{yc}\n")
            f.write("\nState,Parameter,Value\n")
            for state, (A_fit, tau_fit) in self.fits.items():
                f.write(f"{state},A,{A_fit}\n")
                f.write(f"{state},tau,{tau_fit}\n")
        print(f"Histograms and fits saved to {file_path}")


class TransitionMatrixWindow(QtWidgets.QDialog):
    def __init__(self, matrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HMM Transition Matrix")
        layout = QtWidgets.QVBoxLayout()
        plot_widget = pg.PlotWidget(title="Transition Matrix")
        img = ImageItem(matrix.T)
        cmap = colormap.get("viridis")
        img.setLookupTable(cmap.getLookupTable())
        img.setLevels([0, np.max(matrix)])
        plot_widget.addItem(img)
        plot_widget.setLabel('left', 'To State')
        plot_widget.setLabel('bottom', 'From State')
        plot_widget.getViewBox().invertY(True)
        layout.addWidget(plot_widget)
        self.setLayout(layout)

class IntensityPlotWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.plot_widget)
        try:
            self.layout().setContentsMargins(0, 0, 0, 0)
            self.layout().setSpacing(0)
            self.plot_widget.setBackground(None)
            self.plot_widget.ci.layout.setContentsMargins(0, 0, 0, 0)
            self.plot_widget.ci.layout.setSpacing(0)
        except Exception:
            pass
        self.plots = []
        # Optional global y-range override for all trace plots (histograms are Y-linked)
        self._y_range_override = None  # type: tuple[float, float] | None
        self.v_lines = []

    def _create_log_hist_plot(self, linked_y_plot, show_x_axis):
        log_axis = pg.AxisItem(orientation='bottom', logMode=True)
        hist_plot = pg.PlotItem(axisItems={'bottom': log_axis})
        hist_plot.setYLink(linked_y_plot)
        hist_plot.getViewBox().invertX(False)
        hist_plot.hideAxis('left')
        if not show_x_axis:
            hist_plot.hideAxis('bottom')
        else:
            hist_plot.setLabel('bottom', 'Counts (log)')
        return hist_plot

    def _add_fill_between_yaxis_and_curve(self, plot, x_data, y_data, color=(255, 0, 0, 80)):
        path = QtGui.QPainterPath()
        path.moveTo(0, y_data[0])
        for x, y in zip(x_data, y_data):
            path.lineTo(x, y)
        path.lineTo(0, y_data[-1])
        path.closeSubpath()

        item = QtWidgets.QGraphicsPathItem(path)
        item.setBrush(QtGui.QBrush(QtGui.QColor(*color)))
        item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        plot.addItem(item)

    def set_y_range(self, y_min: float, y_max: float):
        """Set a global Y range for all trace plots. Histograms are Y-linked and will follow."""
        print(f"set_y_range({y_min}, {y_max})")
        if y_min is None or y_max is None:
            self._y_range_override = None
            return
        a = float(y_min)
        b = float(y_max)
        if not np.isfinite(a) or not np.isfinite(b):
            return
        if a > b:
            a, b = b, a
        self._y_range_override = (a, b)
        print(f"Global Y range override: {self._y_range_override}")
        for trace_plot, _ in self.plots:
            print(f"{trace_plot.name}: {trace_plot.getYRange()}")
            trace_plot.setYRange(a, b, padding=0)
            print(f"{trace_plot.name}: {trace_plot.getYRange()}")

    def _apply_y_override_if_any(self):
        ov = getattr(self, '_y_range_override', None)
        if ov and isinstance(ov, tuple) and len(ov) == 2:
            a, b = ov
            for trace_plot, _ in self.plots:
                try:
                    trace_plot.setYRange(a, b, padding=0)
                except Exception:
                    pass

    def plot_trace_and_histogram(
        self, time_axis, traces, channel_labels=None,
        bin_count=100, time_window_ms=10.0,
        hist_min=None, hist_max=None, hmm_states=None,
        show_window_lines=False
    ):
        self.plot_widget.clear()
        self.plots.clear()
        self.v_lines.clear()

        self.plot_widget.ci.layout.setColumnStretchFactor(0, 2)
        self.plot_widget.ci.layout.setColumnStretchFactor(1, 1)

        n_channels = traces.shape[1]
        if channel_labels is None:
            channel_labels = [f"{i}" for i in range(n_channels)]

        for i in range(n_channels):
            trace = traces[:, i]
            label = channel_labels[i]
            show_x = (i == n_channels)

            trace_plot = self.plot_widget.addPlot(row=i, col=0)
            try:
                trace_plot.hideButtons()
                trace_plot.getViewBox().setPadding(0.0)
                trace_plot.layout.setContentsMargins(0, 0, 0, 0)
            except Exception:
                pass
            trace_plot.plot(time_axis, trace, pen='b', name=str(label))
            trace_plot.setLabel('left', f'{label}\nCounts / {int(time_window_ms)} ms')
            if not show_x:
                trace_plot.hideAxis('bottom')
            else:
                trace_plot.setLabel('bottom', 'Time', units='s')

            hist_plot = self._create_log_hist_plot(trace_plot, show_x)
            self.plot_widget.addItem(hist_plot, row=i, col=1)
            try:
                hist_plot.hideButtons()
                hist_plot.getViewBox().setPadding(0.0)
                hist_plot.layout.setContentsMargins(0, 0, 0, 0)
            except Exception:
                pass

            data = trace[trace > 0]
            if hist_min is not None and hist_max is not None:
                data = data[(data >= hist_min) & (data <= hist_max)]

            if len(data) > 0:
                counts, bins = np.histogram(data, bins=bin_count, density=False)
                centers = 0.5 * (bins[:-1] + bins[1:])
                hist_plot.addItem(pg.PlotCurveItem(counts, centers[1:], pen='r', stepMode=True))
                self._add_fill_between_yaxis_and_curve(hist_plot, counts, centers)

            self.plots.append((trace_plot, hist_plot))

        combined = traces.sum(axis=1)
        trace_plot = self.plot_widget.addPlot(row=n_channels, col=0)
        try:
            trace_plot.addLegend()
        except Exception:
            pass
        trace_plot.plot(time_axis, combined, pen='g', name='Sum')
        trace_plot.setLabel('left', f'Sum\nCounts / {int(time_window_ms)} ms')
        trace_plot.setLabel('bottom', 'Time', units='s')

        hist_plot = self._create_log_hist_plot(trace_plot, True)
        self.plot_widget.addItem(hist_plot, row=n_channels, col=1)

        data = combined[combined > 0]
        if hist_min is not None and hist_max is not None:
            data = data[(data >= hist_min) & (data <= hist_max)]

        if len(data) > 0:
            counts, bins = np.histogram(data, bins=bin_count, density=False)
            centers = 0.5 * (bins[:-1] + bins[1:])
            curve = pg.PlotCurveItem(counts, centers[1:], pen='r', stepMode=True)
            hist_plot.addItem(curve)
            self._add_fill_between_yaxis_and_curve(hist_plot, counts, centers)

        self.plots.append((trace_plot, hist_plot))

        # Optionally add vertical lines at time-window boundaries
        if show_window_lines and time_axis is not None and len(time_axis) > 1:
            try:
                # time_axis is assumed to be bin start times in seconds
                # Limit to at most 200 lines for performance
                n = len(time_axis)
                step = max(1, n // 200)
                positions = time_axis[::step]
                pen = pg.mkPen(color=(150, 150, 150, 120), width=1)
                for (pitem, _h) in self.plots:
                    for x in positions:
                        vline = pg.InfiniteLine(pos=x, angle=90, pen=pen)
                        pitem.addItem(vline)
            except Exception:
                pass

        if hmm_states is not None:
            state_plot = self.plot_widget.addPlot(row=n_channels+1, col=0, colspan=1)
            state_plot.setXLink(self.plots[0][0])
            state_plot.plot(time_axis, hmm_states, pen='y')
            state_plot.setLabel('left', 'Hidden States')
            state_plot.setLabel('bottom', 'Time', units='s')

            combined_vals = traces.sum(axis=1)
            means = []
            for s in range(int(np.max(hmm_states)) + 1):
                means.append(np.mean(combined_vals[hmm_states == s]))
            order = np.argsort(means)
            remap = np.zeros_like(order)
            for new_idx, old in enumerate(order):
                remap[old] = new_idx
            sorted_states = remap[hmm_states]

            state_plot.clear()
            state_plot.plot(time_axis, sorted_states, pen='y')

            counts = np.bincount(sorted_states, minlength=len(order))
            state_hist = self.plot_widget.addPlot(row=n_channels+1, col=1, colspan=1)
            bg = pg.BarGraphItem(x=np.arange(len(order)), height=counts, width=0.8, brush=(200, 200, 100, 200))
            state_hist.addItem(bg)
            state_hist.setLabel('bottom', 'State (sorted)')
            state_hist.setLabel('left', 'Count')

            if traces.shape[1] >= 2:
                ch0 = traces[:, 0]
                total = np.clip(traces.sum(axis=1), 1e-6, None)
                fret_eff = ch0 / total

                mask = ch0 >= (hist_min or 0.0)
                fret_by_state = {}

                for s in range(int(np.max(hmm_states)) + 1):
                    valid = (hmm_states == s) & mask
                    state_fret = fret_eff[valid]
                    if len(state_fret) > 0:
                        fret_by_state[s] = state_fret

                # Create combined histogram
                bins = np.linspace(0, 1, 51)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                combined_hist = np.zeros_like(bin_centers)

                fret_plot = self.plot_widget.addPlot(row=n_channels + 2, col=0)
                fret_plot.setXLink(self.plots[0][0])
                fret_plot.setLabel('left', 'FRET Hist\nper State')
                fret_plot.setLabel('bottom', 'FRET Efficiency')

                hist_plot = pg.PlotItem()
                self.plot_widget.addItem(hist_plot, row=n_channels + 2, col=1)

                for state, state_fret in fret_by_state.items():
                    hist, _ = np.histogram(state_fret, bins=bins, density=True)
                    combined_hist += hist
                    hist_plot.plot(bin_centers, hist[1:], stepMode=True, fillLevel=0,
                                   brush=(state * 50 % 255, 100, 180, 100))

                # Add combined histogram
                hist_plot.plot(bin_centers, combined_hist[1:], stepMode=True, pen='k')

                fret_plot.plot(time_axis, fret_eff, pen='m')

        for plot, _ in self.plots[1:]:
            plot.setXLink(self.plots[0][0])
        for _, hist in self.plots[1:]:
            hist.setXLink(self.plots[0][1])
        # Apply any y-range override after plots are constructed
        try:
            self._apply_y_override_if_any()
        except Exception:
            pass

class IntensityTrace(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTTR Intensity Trace Viewer")
        # 1) Instantiate the plot widget
        self.plot_widget = IntensityPlotWidget(self)

        # Detector setup storage and page
        self._detector_settings = None
        self.detector_wizard_page = DetectorWizardPage()
        self.detector_wizard_page.setParent(self)
        # Keep the page hidden; it will be shown inside a modal dialog on demand
        try:
            self.detector_wizard_page.hide()
        except Exception:
            pass

        # 2) Instantiate labels and buttons
        self.file_path_edit = QtWidgets.QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("No file selected")
        self.file_path_edit.setToolTip("Current TTTR file path (read-only; click to select and copy)")

        # Compatibility: provide a file_label-like interface expected by callers
        class _FileLabelCompat:
            def __init__(self, line_edit: QtWidgets.QLineEdit):
                self._le = line_edit
            def setText(self, text: str):
                t = str(text) if text is not None else ""
                tl = t.lower()
                if tl.startswith("selected file:"):
                    # Keep only the path part after the first ':'
                    try:
                        t = t.split(':', 1)[1].strip()
                    except Exception:
                        pass
                self._le.setText(t)
            def text(self) -> str:
                return self._le.text()

        # Expose compatibility attribute used by other modules (e.g., Trace Browser)
        self.file_label = _FileLabelCompat(self.file_path_edit)

        self.load_button    = QtWidgets.QToolButton()
        self.load_button.setText("Load TTTR")
        self.load_button.clicked.connect(self.load_file)
        self.load_button.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        self.setup_button = QtWidgets.QToolButton()
        self.setup_button.setText("Setup")
        self.setup_button.clicked.connect(self._open_setup_dialog)
        self.setup_button.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        self.save_button    = QtWidgets.QToolButton()
        self.save_button.setText("Save Traces")
        self.save_button.clicked.connect(self.save_output)
        self.save_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.dist_button    = QtWidgets.QToolButton()
        self.dist_button.setText("FRET Distributions")
        self.dist_button.clicked.connect(self.show_fret_distributions)
        self.dist_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.elbow_button   = QtWidgets.QToolButton()
        self.elbow_button.setText("BIC Elbow")
        self.elbow_button.clicked.connect(self.show_elbow_plot)
        self.elbow_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.hmm_button     = QtWidgets.QToolButton()
        self.hmm_button.setText("Compute HMM")
        self.hmm_button.clicked.connect(self.perform_hmm)
        self.hmm_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.hmm_button.setStyleSheet("font-weight: bold;")

        self.matrix_button  = QtWidgets.QToolButton()
        self.matrix_button.setText("HMM Matrix")
        self.matrix_button.clicked.connect(self.show_matrix)
        self.matrix_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.dwell_button   = QtWidgets.QToolButton()
        self.dwell_button.setText("Dwell Times")
        self.dwell_button.clicked.connect(self.show_dwell_times)
        self.dwell_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        # 3) Instantiate input fields and spin boxes
        # Time window (ms) as double spin box with adaptive step
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.1, 999.0)
        self.window_spin.setDecimals(2)
        self.window_spin.setValue(10.0)
        self.window_spin.setSuffix(" ms")
        self.window_spin.setKeyboardTracking(False)
        # Initial step
        try:
            self.window_spin.setSingleStep(0.5)
        except Exception:
            pass
        # Removed: used_channels_le and micro_ranges_le fields (no longer needed)

        self.bin_spinner = QtWidgets.QSpinBox()
        self.bin_spinner.setRange(10, 500)
        self.bin_spinner.setValue(41)

        self.hist_min_input = QtWidgets.QDoubleSpinBox()
        self.hist_min_input.setRange(0.0, 10000.0)
        self.hist_min_input.setDecimals(2)
        self.hist_min_input.setValue(0.0)

        self.hist_max_input = QtWidgets.QDoubleSpinBox()
        self.hist_max_input.setRange(0.0, 10000.0)
        self.hist_max_input.setDecimals(2)
        self.hist_max_input.setValue(1000.0)

        self.hmm_components_spinner = QtWidgets.QSpinBox()
        self.hmm_components_spinner.setRange(1, 15)
        self.hmm_components_spinner.setValue(3)

        # 4) Layout: Main vertical layout with file section on top and tabs below
        main_widget_layout = QtWidgets.QVBoxLayout()
        main_widget_layout.setContentsMargins(0, 0, 0, 0)
        main_widget_layout.setSpacing(0)

        # File loading section with Load button, filename, and Setup button (on top)
        file_layout = QtWidgets.QHBoxLayout()
        file_layout.setSpacing(0)
        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.setup_button)
        main_widget_layout.addLayout(file_layout)

        # Tab widget below file section
        tab_widget = QtWidgets.QTabWidget()

        # -- Processing Tab --
        proc_tab = QtWidgets.QWidget()
        proc_layout = QtWidgets.QVBoxLayout(proc_tab)
        proc_layout.setContentsMargins(0, 0, 0, 0)
        proc_layout.setSpacing(0)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setSpacing(0)

        time_group = QtWidgets.QGroupBox("Time Window")
        time_layout = QtWidgets.QVBoxLayout(time_group)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(0)
        time_inner = QtWidgets.QHBoxLayout()
        time_inner.addWidget(QtWidgets.QLabel("ms:"))
        time_inner.addWidget(self.window_spin)
        time_layout.addLayout(time_inner)
        time_layout.addWidget(self.save_button)
        main_layout.addWidget(time_group)

        detector_group = QtWidgets.QGroupBox("Detector Selection")
        self.detector_checkbox_layout = QtWidgets.QVBoxLayout(detector_group)
        self.detector_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.detector_checkbox_layout.setSpacing(0)
        self.detector_checkboxes = {}
        main_layout.addWidget(detector_group)

        main_layout.addStretch()

        hist_group = QtWidgets.QGroupBox("Histogram Settings")
        hist_layout = QtWidgets.QGridLayout(hist_group)
        hist_layout.setContentsMargins(0, 0, 0, 0)
        hist_layout.setSpacing(0)
        hist_layout.addWidget(QtWidgets.QLabel("Bins:"), 0, 0)
        hist_layout.addWidget(self.bin_spinner, 0, 1)
        hist_layout.addWidget(QtWidgets.QLabel("Min Counts:"), 1, 0)
        hist_layout.addWidget(self.hist_min_input, 1, 1)
        hist_layout.addWidget(QtWidgets.QLabel("Max Counts:"), 2, 0)
        hist_layout.addWidget(self.hist_max_input, 2, 1)
        hist_layout.setColumnStretch(2, 1)
        main_layout.addWidget(hist_group)

        proc_layout.addLayout(main_layout)
        proc_layout.addStretch()

        self._refresh_detector_checkboxes()

        tab_widget.addTab(proc_tab, "Processing")

        # -- HMM Tab --
        hmm_tab = QtWidgets.QWidget()
        hmm_main_layout = QtWidgets.QHBoxLayout(hmm_tab)
        hmm_main_layout.setContentsMargins(0, 0, 0, 0)
        hmm_main_layout.setSpacing(0)

        hmm_main_layout.addWidget(self.hmm_button)

        horizonalspacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        hmm_main_layout.addItem(horizonalspacer)

        hmm_right_layout = QtWidgets.QHBoxLayout()
        hmm_right_layout.setSpacing(0)

        hmm_settings_group = QtWidgets.QGroupBox("HMM Settings")
        hmm_settings_layout = QtWidgets.QVBoxLayout(hmm_settings_group)
        hmm_settings_layout.setContentsMargins(0, 0, 0, 0)
        hmm_settings_layout.setSpacing(0)
        components_layout = QtWidgets.QHBoxLayout()
        components_layout.addWidget(QtWidgets.QLabel("HMM Components:"))
        components_layout.addWidget(self.hmm_components_spinner)
        hmm_settings_layout.addLayout(components_layout)
        hmm_settings_layout.addWidget(self.elbow_button)
        hmm_right_layout.addWidget(hmm_settings_group)

        hmm_results_group = QtWidgets.QGroupBox("HMM Results")
        hmm_results_layout = QtWidgets.QGridLayout(hmm_results_group)
        hmm_results_layout.setContentsMargins(0, 0, 0, 0)
        hmm_results_layout.setSpacing(0)
        hmm_results_layout.addWidget(self.matrix_button, 0, 0)
        hmm_results_layout.addWidget(self.dwell_button, 0, 1)
        hmm_results_layout.addWidget(self.dist_button, 1, 0, 1, 2)
        hmm_right_layout.addWidget(hmm_results_group)

        hmm_main_layout.addLayout(hmm_right_layout)

        tab_widget.addTab(hmm_tab, "HMM")

        # Add tab widget and plot widget to main layout
        main_widget_layout.addWidget(tab_widget)
        main_widget_layout.addWidget(self.plot_widget)

        # 5) Set the main layout
        self.setLayout(main_widget_layout)

        self.current_data = None

        # Connect signals for live updates
        self.bin_spinner.valueChanged.connect(self.update_plot)
        self.hist_min_input.valueChanged.connect(self.update_plot)
        self.hist_max_input.valueChanged.connect(self.update_plot)
        self.window_spin.valueChanged.connect(self._on_time_window_changed)
        # Adaptive step for time window
        try:
            self.window_spin.valueChanged.connect(self._update_window_step)
        except Exception:
            pass

    def _open_setup_dialog(self):
        # Embed the DetectorWizardPage inside a standard dialog (not a Wizard)
        try:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Detector Setup")
            lay = QtWidgets.QVBoxLayout(dlg)

            # Create a fresh page instance scoped to the dialog to avoid reparent/floating issues
            page = DetectorWizardPage()
            # Preload existing settings if present
            try:
                if self._detector_settings is not None and hasattr(page, 'set_settings'):
                    page.set_settings(self._detector_settings)
            except Exception:
                pass

            lay.addWidget(page)
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            lay.addWidget(btns)
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)

            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                # Read settings back from the page
                if hasattr(page, 'get_settings'):
                    self._detector_settings = page.get_settings()
                    logging.info("IntensityTrace: Updated detector settings from embedded page")
                    # Refresh detector selection choices in UI
                    try:
                        self._refresh_detector_checkboxes()
                    except Exception:
                        pass
                # Re-process current file if any
                fp = self.file_path_edit.text().strip()
                if fp:
                    self.load_file(file_path=fp)
        except Exception as e:
            logging.error(f"IntensityTrace: Setup dialog failed: {e}")

    def _on_time_window_changed(self):
        if self.file_path_edit.text():
            self.load_file(file_path=self.file_path_edit.text())

    def _update_window_step(self, val):
        """Adapt the singleStep of the time window spinbox to the current magnitude."""
        try:
            if val < 1:
                step = 0.1
            elif val < 10:
                step = 0.5
            elif val < 100:
                step = 1.0
            else:
                step = 5.0
            self.window_spin.setSingleStep(step)
        except Exception:
            pass

    def _on_detector_select_changed(self, *args):
        # Re-process if a file is loaded
        fp = self.file_path_edit.text().strip()
        # Update info fields regardless
        self._update_detector_info_fields()
        if fp:
            self.load_file(file_path=fp)

    def _on_detector_checkbox_changed(self):
        """Handle detector checkbox state change."""
        # Re-process if a file is loaded
        fp = self.file_path_edit.text().strip()
        if fp:
            self.load_file(file_path=fp)
    
    def _on_detector_selection_changed(self):
        """Handle detector selection change in list widget (legacy method for compatibility)."""
        # Re-process if a file is loaded
        fp = self.file_path_edit.text().strip()
        # Update info fields regardless
        self._update_detector_info_fields()
        if fp:
            self.load_file(file_path=fp)

    def _refresh_detector_checkboxes(self):
        """
        Refresh the detector selection checkboxes with available detectors from DetectorWizard settings.
        """
        try:
            # Clear existing checkboxes
            for checkbox in self.detector_checkboxes.values():
                checkbox.setParent(None)
                checkbox.deleteLater()
            self.detector_checkboxes.clear()
            
            # Get detector names from DetectorWizard settings
            detector_names = []
            settings = None
            
            # Debug: Check current settings
            print(f"DEBUG: _detector_settings = {self._detector_settings}")
            
            try:
                # Try to get settings from current instance first
                if isinstance(self._detector_settings, dict) and self._detector_settings:
                    settings = self._detector_settings
                    print(f"DEBUG: Using in-memory settings")
                else:
                    # Load from central detector setups file
                    print(f"DEBUG: Loading from central setups file")
                    setups_data = load_detector_setups()
                    print(f"DEBUG: setups_data = {setups_data}")
                    if setups_data:
                        last_used = setups_data.get('last_used')
                        setups = setups_data.get('setups', {})
                        print(f"DEBUG: last_used = {last_used}, available setups = {list(setups.keys()) if setups else None}")
                        if last_used and last_used in setups:
                            settings = setups[last_used]
                            print(f"DEBUG: Loaded settings for '{last_used}'")
                            # Adopt these settings if we don't have any
                            if not isinstance(self._detector_settings, dict) or not self._detector_settings:
                                self._detector_settings = settings
                                print(f"DEBUG: Adopted settings into _detector_settings")
                
                if settings:
                    print(f"DEBUG: Settings available, building detector checkboxes directly from settings")
                    # Build detector mapping directly from settings (ignore windows, use only detectors)
                    detectors = settings.get("detectors", {})
                    print(f"DEBUG: Found {len(detectors)} detectors")
                    
                    # Use detector names directly
                    detector_names = list(detectors.keys()) if detectors else []
                    print(f"DEBUG: detector_names = {detector_names}")
                else:
                    print(f"DEBUG: No settings available")
                    
            except Exception as e:
                print(f"DEBUG: Exception in settings loading: {e}")
                logging.warning(f"IntensityTrace: Failed to get detectors from DetectorWizard: {e}")
            
            # Add individual detector checkboxes
            for dname in detector_names:
                checkbox = QtWidgets.QCheckBox(str(dname))
                checkbox.setChecked(True)  # Check all detectors by default
                checkbox.stateChanged.connect(self._on_detector_checkbox_changed)
                self.detector_checkbox_layout.addWidget(checkbox)
                self.detector_checkboxes[str(dname)] = checkbox
                print(f"DEBUG: Added detector checkbox '{dname}'")
            
            # If no detectors found, add fallback routing channel options
            if not detector_names:
                print(f"DEBUG: No detectors found, adding fallback routing channels")
                # Add common routing channels as fallback
                for ch in [0, 1, 2, 3, 4, 5, 6, 7]:
                    checkbox = QtWidgets.QCheckBox(f"Routing Channel {ch}")
                    checkbox.setChecked(True)
                    checkbox.stateChanged.connect(self._on_detector_checkbox_changed)
                    self.detector_checkbox_layout.addWidget(checkbox)
                    self.detector_checkboxes[f"routing_{ch}"] = checkbox
                    print(f"DEBUG: Added fallback routing channel checkbox {ch}")
            
            print(f"DEBUG: Total checkboxes: {len(self.detector_checkboxes)}")
            
        except Exception as e:
            print(f"DEBUG: Exception in _refresh_detector_checkboxes: {e}")
            logging.error(f"IntensityTrace: Error refreshing detector checkboxes: {e}")

    def _update_detector_info_fields(self):
        """Update detector info (no longer displays fields since they were removed)."""
        # This method is kept for compatibility but no longer updates display fields
        # since the used_channels_le and micro_ranges_le fields were removed
        pass

    def load_file(self, file_path=None, checked=False):
        # Qt clicked(bool) may pass a boolean into file_path; sanitize
        if not isinstance(file_path, (str, pathlib.Path)):
            file_path = None
        if file_path is None:
            # Use only tttrlib.get_supported_filetypes() to build the filter
            try:
                import tttrlib as _tttr
            except Exception:
                _tttr = None
            exts = []
            try:
                if _tttr is not None and hasattr(_tttr, "get_supported_filetypes"):
                    exts = list(_tttr.get_supported_filetypes())
            except Exception:
                exts = []
            # Normalize extensions to dotted lowercase
            norm = []
            for e in exts:
                s = str(e).strip().lower()
                if not s:
                    continue
                if not s.startswith('.'):
                    s = '.' + s
                if s not in norm:
                    norm.append(s)
            if norm:
                patterns = ' '.join(f"*{e}" for e in norm)
                filter_str = f"TTTR Files ({patterns});;All Files (*)"
            else:
                filter_str = "All Files (*)"

            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open TTTR File", "", filter_str)

        if not file_path:
            return

        self.file_path_edit.setText(str(file_path))
        time_window_ms = float(self.window_spin.value())
        time_window_s = time_window_ms / 1000.0

        # Get selected detectors from checkboxes
        selected_detectors = []
        try:
            for detector_name, checkbox in self.detector_checkboxes.items():
                if checkbox.isChecked():
                    selected_detectors.append(detector_name)
        except Exception:
            selected_detectors = []
        
        # Keep info fields synced
        try:
            self._update_detector_info_fields()
        except Exception:
            pass

        time_axis, padded, all_chs = self.process_ptu(
            pathlib.Path(str(file_path)), time_window_s, selected_detectors=selected_detectors
        )
        self.current_data = {'time_axis': time_axis, 'padded': padded, 'channels': all_chs, 'window_ms': time_window_ms, 'hmm_states': None}
        self.update_plot()

    def show_fret_distributions(self):
        # Requires HMM states and at least two channels
        if not getattr(self, 'current_data', None):
            return
        states = self.current_data.get('hmm_states')
        padded = self.current_data.get('padded')
        if states is None or padded is None or getattr(padded, 'shape', (0,0))[1] < 2:
            return
        # Compute FRET efficiency (ch0 / total)
        ch0 = padded[:, 0]
        total = np.clip(padded.sum(axis=1), 1e-12, None)
        fret_eff = ch0 / total
        fret_by_state = {}
        for s in range(int(np.max(states)) + 1):
            fret_by_state[s] = fret_eff[states == s]
        dlg = DistPlotWindow(fret_by_state, self)
        dlg.exec_()

    def show_elbow_plot(self):
        if not getattr(self, 'current_data', None):
            return
        padded = self.current_data.get('padded')
        if padded is None or getattr(padded, 'size', 0) == 0:
            return
        max_states = int(self.hmm_components_spinner.maximum())
        bics = compute_bic_curve(padded, max_states=max_states)
        dlg = ElbowPlotWindow(bics, self)
        dlg.exec_()

    def show_dwell_times(self):
        if not getattr(self, 'current_data', None):
            return
        states = self.current_data.get('hmm_states')
        if states is None:
            return
        time_step = float(self.current_data.get('window_ms', 0.0)) / 1000.0
        dwell_times = compute_dwell_times(states, time_step)
        dlg = DwellTimeWindow(dwell_times, self)
        dlg.exec_()

    def show_matrix(self):
        if not getattr(self, 'current_data', None):
            return
        matrix = self.current_data.get('transmat')
        if matrix is None:
            return
        dlg = TransitionMatrixWindow(matrix, self)
        dlg.exec_()

    def perform_hmm(self):
        if not self.current_data:
            return
        n_comp = self.hmm_components_spinner.value()
        traces = self.current_data['padded']
        hmm_states, transmat = self.apply_hmm(traces, n_components=n_comp)
        self.current_data['hmm_states'] = hmm_states
        self.current_data['transmat'] = transmat

        # Save outputs in structured subfolders
        try:
            tttr_path = pathlib.Path(self.file_path_edit.text()).resolve()
            # Build folder name: <filename>_HMM#<NbrState>_<XX>ms
            stem = tttr_path.stem
            time_window_ms = int(round(self.current_data.get('window_ms', 0)))
            folder_name = f"{stem}_HMM#{n_comp}_{time_window_ms}ms"
            base_dir = tttr_path.parent / folder_name
            bst_dir = base_dir / "bst"
            traces_dir = base_dir / "traces"
            hists_dir = base_dir / "hist"
            # Ensure directories
            for d in (base_dir, bst_dir, traces_dir, hists_dir):
                d.mkdir(parents=True, exist_ok=True)

            # 1) Save burst IDs (BST) under bst/
            logging.info(f"IntensityTrace: Saving burst IDs to {bst_dir}")
            save_burst_ids(
                hmm_states,
                self.current_data['time_axis'],
                self.current_data['window_ms'] / 1000.0,
                tttrlib.TTTR(str(tttr_path)),
                output_dir=str(bst_dir),
                file_path=self.file_path_edit.text()
            )

            # 2) Save traces (with optional HMM state) under traces/
            time_axis = np.asarray(self.current_data.get('time_axis'))
            padded = np.asarray(self.current_data.get('padded'))
            labels = self.current_data.get('channels') or []
            states = self.current_data.get('hmm_states')

            if time_axis.size > 0 and padded.size > 0:
                chan_labels = [str(l) for l in labels] if len(labels) == padded.shape[1] else [f"ch{i}" for i in range(padded.shape[1])]
                header_cols = ["time_s"] + chan_labels + (["HMM_State"] if states is not None else [])
                cols = [time_axis]
                for i in range(padded.shape[1]):
                    cols.append(padded[:, i])
                if states is not None:
                    cols.append(np.asarray(states, dtype=int))
                data = np.column_stack(cols)
                csv_path = traces_dir / f"{stem}_traces.csv"
                fmts = ["%.6f"] * (1 + padded.shape[1]) + (["%d"] if states is not None else [])
                np.savetxt(str(csv_path), data, delimiter=",", header=",".join(header_cols), comments="", fmt=fmts)
                logging.info(f"IntensityTrace: Saved traces to {csv_path}")

                # 2a) Save hidden state trajectory
                if states is not None:
                    state_traj = np.column_stack([time_axis, np.asarray(states, dtype=int)])
                    state_hdr = "time_s,HMM_State"
                    state_path = traces_dir / f"{stem}_state_traj.csv"
                    np.savetxt(str(state_path), state_traj, delimiter=",", header=state_hdr, comments="", fmt=["%.6f", "%d"])
                    logging.info(f"IntensityTrace: Saved state trajectory to {state_path}")

                # 2b) Save FRET trajectory if at least two channels
                if padded.shape[1] >= 2:
                    ch0 = padded[:, 0]
                    total = np.clip(padded.sum(axis=1), 1e-12, None)
                    fret_eff = ch0 / total
                    fret_traj = np.column_stack([time_axis, fret_eff])
                    fret_hdr = "time_s,FRET_efficiency"
                    fret_path = traces_dir / f"{stem}_fret_traj.csv"
                    np.savetxt(str(fret_path), fret_traj, delimiter=",", header=fret_hdr, comments="", fmt=["%.6f", "%.6f"])
                    logging.info(f"IntensityTrace: Saved FRET trajectory to {fret_path}")

            # 3) Save histograms under hist/
            # Intensity histograms per channel and combined
            bin_count = int(self.bin_spinner.value())
            hist_min = self.hist_min_input.value()
            hist_max = self.hist_max_input.value()
            if "padded" in self.current_data and np.size(self.current_data['padded']) > 0:
                traces_arr = np.asarray(self.current_data['padded'])
                # Per-channel histograms
                for i in range(traces_arr.shape[1]):
                    vals = traces_arr[:, i]
                    vals = vals[vals > 0]
                    if np.isfinite(hist_min) and np.isfinite(hist_max) and hist_max > hist_min:
                        vals = vals[(vals >= hist_min) & (vals <= hist_max)]
                    if vals.size == 0:
                        continue
                    counts, bins = np.histogram(vals, bins=bin_count, density=False)
                    centers = 0.5 * (bins[:-1] + bins[1:])
                    chan_label = str(labels[i]) if i < len(labels) else f"ch{i}"
                    out_path = hists_dir / f"{stem}_hist_{chan_label}.csv"
                    header = "bin_center,counts"
                    np.savetxt(str(out_path), np.column_stack([centers, counts]), delimiter=",", header=header, comments="", fmt=["%.6f", "%d"])
                    logging.info(f"IntensityTrace: Saved histogram for {chan_label} to {out_path}")

                # Combined histogram
                combined = traces_arr.sum(axis=1)
                vals = combined[combined > 0]
                if np.isfinite(hist_min) and np.isfinite(hist_max) and hist_max > hist_min:
                    vals = vals[(vals >= hist_min) & (vals <= hist_max)]
                if vals.size > 0:
                    counts, bins = np.histogram(vals, bins=bin_count, density=False)
                    centers = 0.5 * (bins[:-1] + bins[1:])
                    out_path = hists_dir / f"{stem}_hist_sum.csv"
                    header = "bin_center,counts"
                    np.savetxt(str(out_path), np.column_stack([centers, counts]), delimiter=",", header=header, comments="", fmt=["%.6f", "%d"])
                    logging.info(f"IntensityTrace: Saved combined histogram to {out_path}")

                # FRET histogram if applicable
                if traces_arr.shape[1] >= 2:
                    ch0 = traces_arr[:, 0]
                    total = np.clip(traces_arr.sum(axis=1), 1e-12, None)
                    fret_eff = ch0 / total
                    bins = np.linspace(0.0, 1.0, 51)
                    hist, edges = np.histogram(fret_eff[np.isfinite(fret_eff)], bins=bins, density=True)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    out_path = hists_dir / f"{stem}_fret_hist.csv"
                    header = "fret_bin_center,density"
                    np.savetxt(str(out_path), np.column_stack([centers, hist]), delimiter=",", header=header, comments="", fmt=["%.6f", "%.6f"])
                    logging.info(f"IntensityTrace: Saved FRET histogram to {out_path}")
        except Exception as e:
            logging.error(f"IntensityTrace: Failed to save HMM outputs: {e}")

        self.update_plot()

    def save_output(self):
        # Export current intensity traces (and HMM states if available) to CSV
        if not self.current_data:
            return

        # Prompt the user for a file path
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Traces", "", "CSV Files (*.csv)"
        )
        if not save_path:
            return

        try:
            time_axis = np.asarray(self.current_data.get('time_axis'))
            padded = np.asarray(self.current_data.get('padded'))
            labels = self.current_data.get('channels') or []
            states = self.current_data.get('hmm_states')

            if time_axis.size == 0 or padded.size == 0:
                logging.warning("IntensityTrace: No data to save.")
                return

            # Prepare CSV header
            chan_labels = [str(l) for l in labels] if len(labels) == padded.shape[1] else [f"ch{i}" for i in range(padded.shape[1])]
            header_cols = ["time_s"] + chan_labels
            include_states = states is not None and len(states) == len(time_axis)
            if include_states:
                header_cols.append("HMM_State")
            header = ",".join(header_cols)

            # Prepare data matrix
            cols = [time_axis]
            for i in range(padded.shape[1]):
                cols.append(padded[:, i])
            if include_states:
                cols.append(np.asarray(states, dtype=int))
            data = np.column_stack(cols)

            # Save with appropriate formats (floats for time/traces, int for states)
            fmts = ["%.6f"] * (1 + padded.shape[1]) + (["%d"] if include_states else [])
            np.savetxt(save_path, data, delimiter=",", header=header, comments="", fmt=fmts)
            logging.info(f"IntensityTrace: Saved traces to {save_path}")
        except Exception as e:
            logging.error(f"IntensityTrace: Failed to save output: {e}")

    def update_plot(self):
        if not self.current_data:
            return
        bin_count = self.bin_spinner.value()
        hist_min = self.hist_min_input.value()
        hist_max = self.hist_max_input.value()
        self.plot_widget.plot_trace_and_histogram(
            self.current_data['time_axis'], self.current_data['padded'], self.current_data['channels'],
            bin_count=bin_count, time_window_ms=self.current_data['window_ms'],
            hist_min=hist_min, hist_max=hist_max, hmm_states=self.current_data.get('hmm_states')
        )

    def process_ptu(self, ptu_file, time_window_length, selected_detectors=None):
        """
        Compute intensity traces for a TTTR file using DetectorWizard detector mapping or fallback to routing channels.
        
        Args:
            ptu_file: Path to TTTR file
            time_window_length: Time window in seconds
            selected_detectors: List of selected detector names, or None for all detectors
        """
        tttr_obj = tttrlib.TTTR(str(ptu_file))
        logging.info(f"IntensityTrace: Processing {ptu_file} with bin {time_window_length}s and detectors={selected_detectors}")

        # Get detector mapping directly from settings (ignore windows)
        detectors = None
        try:
            if isinstance(self._detector_settings, dict):
                detectors = self._detector_settings.get("detectors", {})
                logging.info(f"IntensityTrace: Found {len(detectors)} detectors")
        except Exception as e:
            logging.warning(f"IntensityTrace: Failed to get detectors: {e}")

        rc = tttr_obj.routing_channels
        mt = tttr_obj.micro_times
        labels = []
        traces = []
        
        # Handle fallback to routing channels if no detectors
        if not detectors:
            logging.info("IntensityTrace: Using fallback routing channel mode")
            # For fallback, use first selected detector or "__all__" if none selected
            fallback_channel = selected_detectors[0] if selected_detectors else "__all__"
            return self._process_routing_channels(tttr_obj, time_window_length, fallback_channel)
        
        # Determine which detectors to process
        detectors_to_process = []
        if not selected_detectors or len(selected_detectors) == 0:
            # If no detectors selected, process all available detectors
            detectors_to_process = list(detectors.keys())
        else:
            # Process only selected detectors that exist in settings
            for det_name in selected_detectors:
                if det_name in detectors:
                    detectors_to_process.append(det_name)
                else:
                    logging.warning(f"IntensityTrace: Selected detector '{det_name}' not found in settings")
        
        if not detectors_to_process:
            logging.warning("IntensityTrace: No valid detectors to process")
            return np.array([]), np.zeros((0,0)), []
            
        for dname in detectors_to_process:
            try:
                dinfo = detectors[dname]
                det_chs = dinfo.get('chs', []) or []
                micro_time_ranges = dinfo.get('micro_time_ranges', []) or []
                
                if not det_chs:
                    logging.warning(f"IntensityTrace: Detector '{dname}' has no routing channels")
                    continue
                
                # Build mask for this detector
                total_mask = np.zeros(rc.shape, dtype=bool)
                
                # Create routing channel mask
                ch_mask = np.isin(rc, np.array(det_chs, dtype=rc.dtype))
                
                # Apply microtime ranges if present
                if micro_time_ranges:
                    mt_mask = np.zeros_like(ch_mask, dtype=bool)
                    for mtr in micro_time_ranges:
                        if mtr is not None and len(mtr) == 2:
                            try:
                                ma, mb = int(mtr[0]), int(mtr[1])
                                mt_mask |= (mt >= ma) & (mt <= mb)
                            except Exception:
                                continue
                    total_mask = ch_mask & mt_mask
                else:
                    # No microtime gating, use all photons from routing channels
                    total_mask = ch_mask

                # Extract photons and create trace
                idxs = np.where(total_mask)[0]
                if idxs.size == 0:
                    logging.warning(f"IntensityTrace: Detector '{dname}' has no photons after gating; adding empty trace")
                    traces.append(np.array([], dtype=float))
                    labels.append(str(dname))
                    continue

                sub_tttr = tttr_obj[idxs]
                counts = sub_tttr.get_intensity_trace(time_window_length)
                traces.append(counts)
                labels.append(str(dname))
                logging.info(f"IntensityTrace: Processed detector '{dname}': {len(counts)} bins, {idxs.size} photons")
                
            except Exception as e:
                logging.warning(f"IntensityTrace: Error processing detector '{dname}': {e}")
                continue

        # Pad traces to common length
        num_bins = max((len(t) for t in traces), default=0)
        padded = np.zeros((num_bins, len(traces))) if num_bins > 0 else np.zeros((0,0))
        for i, t in enumerate(traces):
            if len(t) > 0:
                padded[:len(t), i] = t
        time_axis = np.arange(num_bins) * time_window_length if num_bins > 0 else np.array([])
        return time_axis, padded, labels

    def _process_routing_channels(self, tttr_obj, time_window_length, selected_channel):
        """
        Fallback method to process routing channels when DetectorWizard mapping is not available.
        """
        logging.info(f"IntensityTrace: Processing routing channels, selected={selected_channel}")
        
        rc = tttr_obj.routing_channels
        all_routing_channels = sorted(tttr_obj.get_used_routing_channels())
        
        labels = []
        traces = []
        
        # Determine which routing channels to process
        channels_to_process = []
        if selected_channel == "__all__":
            channels_to_process = all_routing_channels
            logging.info(f"IntensityTrace: Processing all routing channels: {channels_to_process}")
        elif selected_channel and selected_channel.startswith("routing_"):
            # Extract channel number from "routing_X" format
            try:
                ch_num = int(selected_channel.split("_")[1])
                if ch_num in all_routing_channels:
                    channels_to_process = [ch_num]
                    logging.info(f"IntensityTrace: Processing routing channel {ch_num}")
                else:
                    logging.warning(f"IntensityTrace: Routing channel {ch_num} not found in file")
                    return np.array([]), np.zeros((0,0)), []
            except (ValueError, IndexError):
                logging.warning(f"IntensityTrace: Invalid routing channel format: {selected_channel}")
                return np.array([]), np.zeros((0,0)), []
        else:
            logging.warning(f"IntensityTrace: Invalid channel selection for routing mode: {selected_channel}")
            return np.array([]), np.zeros((0,0)), []
        
        # Process each routing channel
        for ch in channels_to_process:
            try:
                # Create mask for this routing channel
                mask = (rc == ch)
                idxs = np.where(mask)[0]
                
                if idxs.size == 0:
                    logging.warning(f"IntensityTrace: Routing channel {ch} has no photons")
                    traces.append(np.array([], dtype=float))
                    labels.append(f"Ch{ch}")
                    continue
                
                # Create sub-TTTR and compute trace
                sub_tttr = tttr_obj[idxs]
                counts = sub_tttr.get_intensity_trace(time_window_length)
                traces.append(counts)
                labels.append(f"Ch{ch}")
                logging.info(f"IntensityTrace: Processed routing channel {ch}: {len(counts)} bins, {idxs.size} photons")
                
            except Exception as e:
                logging.warning(f"IntensityTrace: Error processing routing channel {ch}: {e}")
                continue
        
        # Pad traces to common length
        num_bins = max((len(t) for t in traces), default=0)
        padded = np.zeros((num_bins, len(traces))) if num_bins > 0 else np.zeros((0,0))
        for i, t in enumerate(traces):
            if len(t) > 0:
                padded[:len(t), i] = t
        
        time_axis = np.arange(num_bins) * time_window_length if num_bins > 0 else np.array([])
        logging.info(f"IntensityTrace: Routing channel processing complete: {len(traces)} traces, {num_bins} bins")
        return time_axis, padded, labels

    def apply_hmm(self, traces, n_components=2):
        logging.info(f"IntensityTrace: Running HMM with {n_components} components on traces shape={getattr(traces, 'shape', None)}")
        model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000)
        model.fit(traces)
        states = model.predict(traces)
        transmat = model.transmat_

        combined = traces.sum(axis=1)
        means = [np.mean(combined[states == s]) for s in range(n_components)]
        order = np.argsort(means)
        remap = np.zeros_like(order)
        for new_idx, old in enumerate(order):
            remap[old] = new_idx
        sorted_states = remap[states]
        transmat_sorted = transmat[np.ix_(order, order)]
        return sorted_states, transmat_sorted

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = IntensityTrace()
    window.show()
    sys.exit(app.exec_())

elif __name__ == "plugin":
    window = IntensityTrace()
    window.show()
